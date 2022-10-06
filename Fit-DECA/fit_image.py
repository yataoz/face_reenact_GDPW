import numpy as np 
import math
import os
import cv2
import random
from scipy.spatial.transform import Rotation
import pickle as pkl

import torch
from torch import nn
from torchvision.transforms import functional
import face_alignment
import nvdiffrast.torch as dr

from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
from decalib.models.lbs import rot_mat_to_euler, batch_rodrigues

from skimage.transform import estimate_transform, AffineTransform

import pdb 

class MeshRenderer(nn.Module):
    def __init__(self, glctx, v_template, tri):
        super(MeshRenderer, self).__init__()

        self.register_buffer('v_template', torch.tensor(v_template, dtype=torch.float32))
        self.register_buffer('tri', torch.tensor(tri, dtype=torch.int32))
    
        self.glctx = glctx

        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                           ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                           (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)

    def create_gl_context(self, device):
        self.glctx = dr.RasterizeGLContext(device=device)
    
    def add_SHlight(self, normal_images, sh_coeff):
        '''
        borrowed from DECA.utils.renderer.py

        normal_images: (b, h, w, c)
        sh_coeff: [bz, 9, 3]
        '''
        N = normal_images.permute(0, 3, 1, 2)
        sh = torch.stack([
                N[:,0]*0.+1., N[:,0], N[:,1], \
                N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
                N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
                ], 
                1) # [bz, 9, h, w]
        sh = sh*self.constant_factor[None,:,None,None]
        shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  => [bz, 3, h, w]
        return shading.permute(0, 2, 3, 1)
    
    def forward(self, proj_verts, viewport_size, world_verts, lights):
        """
        proj_verts: in NDC coordinates. x, y is transformed to [-1, 1]; z is kept as world depth
        """
        batch = proj_verts.shape[0]
        num_verts = proj_verts.shape[1]
        z = proj_verts[..., 2:3]
        min_z = z.amin(dim=1, keepdims=True)
        max_z = z.amax(dim=1, keepdims=True)
        normed_z = 2 * (z - min_z) / (max_z - min_z + 1.e-8) - 1    # normalized to [-1, 1] because nvdiffrast requires depth to be in [-1, 1]
        pos = torch.cat([proj_verts[..., :2], normed_z, torch.ones_like(proj_verts[..., 2:3])], dim=-1) 
        rast, _ = dr.rasterize(self.glctx, pos, self.tri, resolution=viewport_size[::-1])    # (b, h, w, 4)

        # back face culling
        face_verts = util.face_vertices(proj_verts, self.tri.unsqueeze(0))    # (b, num_tri, 3, 3)
        v0 = face_verts[..., 0, :]
        v1 = face_verts[..., 1, :]
        v2 = face_verts[..., 2, :]
        # not the real face normals because z is originally world depth normalized to [-1, 1]
        pseudo_face_normals = torch.cross(v1 - v0, v2 - v0)    # (b, num_tri, 3)
        face_cull_mask = pseudo_face_normals[..., 2] > 0      # (b, num_tri)
        for i in range(rast.shape[0]):
            tri_id = rast[i, ..., 3] - 1   # nvdiffrast rast triangle_id is offset by 1, so we subtract 1 here. (h, w)  
            rast_cull_mask = face_cull_mask[i][tri_id.long()]  # (h, w)
            rast[i, ..., 3][rast_cull_mask] = 0     # set to 0 means no face to be rasterized

        normals = util.vertex_normals(world_verts, self.tri.expand(batch, -1, -1))
        verts_attr = normals
        
        interp_attr, _ = dr.interpolate(verts_attr, rast, self.tri)
        render_attr = dr.antialias(interp_attr, rast, pos, self.tri)

        face_normals = render_attr
        shaded_face = self.add_SHlight(face_normals, lights)

        return shaded_face

class DECAFitting():
    """
    This class is not intended for any use by this module.
    This is mainly used by some external apps, e.g., head_edit_demo.py in face_reenact_pytorch
    """
    _det_short_size = 128

    def __init__(self, target_size, device0):
        self.target_size = target_size      # when target_size is None, we keep the original image resolution as output
        self.device0 = device0 

        #flame_model = np.load('data/generic_model.npz')    # same model as generic_model.pkl but saved in .npz format
        with open('data/generic_model.pkl', 'rb') as f:
            flame_model = pkl.load(f, encoding='latin1')
        flame_tri = flame_model['f'].astype(np.int32)
        flame_v_template = flame_model['v_template'].astype(np.float32)
        mesh_renderer = MeshRenderer(None, flame_v_template, flame_tri)
        mesh_renderer.create_gl_context(device0)
        mesh_renderer.to(device0)
        self.mesh_renderer = mesh_renderer

        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device0)

        deca_cfg.model.use_tex = False
        self.use_detail = False
        self.deca = DECA(config=deca_cfg, device=device0)
    
    def preprocess_and_encode(self, batch_imgs):
        """
        batch_imgs: (b, h, w, 3) RGB images as np.ndarray

        This function detects face and crop around face, and feed the cropped image to DECA encoder to get codes
        """
        # ================= run face detector first to get face boxes in order to crop images
        batch_transforms = [] 
        batch_recover_transforms = []
        batch_crop_imgs = []
        batch_bboxes = []
        final_sizes =[]

        h0, w0 = batch_imgs[0].shape[:2]

        # reduce image size for face alignment 
        det_imgs = torch.as_tensor(batch_imgs, device=self.device0).permute(0, 3, 1, 2)
        if min(h0, w0) > self._det_short_size: 
            det_imgs = functional.resize(det_imgs, size=self._det_short_size)  # shorter side will be resize to 'det_short_size'

        batch_dets = self.face_detector.get_landmarks_from_batch(det_imgs)
        for j, det in enumerate(batch_dets):
            if len(det) == 0:
                # if first trial fails to detect any faces, we try another time with the original image size
                det = self.face_detector.get_landmarks_from_image(batch_imgs[j])
                if det is None or len(det) == 0:
                    left = 0; right = h0-1; top=0; bottom=w0-1
                else:
                    kpt = det[0].squeeze()
                    left = np.min(kpt[:,0])
                    right = np.max(kpt[:,0])
                    top = np.min(kpt[:,1])
                    bottom = np.max(kpt[:,1])
            else:
                kpt = det.squeeze()
                scale = min(h0, w0) / self._det_short_size
                if scale > 1:
                    # resize face crop to original image space if we reduced its size in the beginning
                    kpt *= scale

                left = np.min(kpt[:,0])
                right = np.max(kpt[:,0])
                top = np.min(kpt[:,1])
                bottom = np.max(kpt[:,1])

            # crop 
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
            deca_crop_scale = 1.25      # default value by DECA
            size = int(old_size * deca_crop_scale)
            src_pts_1 = np.array([
                [center[0]-size/2, center[1]-size/2], 
                [center[0]-size/2, center[1]+size/2], 
                [center[0]+size/2, center[1]-size/2],
                ])
            dst_pts_1 = np.array([
                [0, 0], 
                [0, self.deca.image_size], 
                [self.deca.image_size, 0],
                ])      # deca.image_size is 224
            tform_1 = estimate_transform('similarity', src_pts_1, dst_pts_1)
            deca_crop_trans_matx = tform_1.params
        
            # warp 
            crop_img = cv2.warpAffine(batch_imgs[j], deca_crop_trans_matx[:2, :], (self.deca.image_size, self.deca.image_size))
            batch_crop_imgs.append(crop_img)

            if self.target_size is None:
                # keep original resolution
                final_size = (w0, h0)
                final_crop_trans_matx = np.eye(3)
                tform_2 = AffineTransform(matrix=final_crop_trans_matx)
            else:
                # find a larger crop for final image to save
                # all predicted vertices/mesh will be converted to this larger crop
                final_crop_scale = 1.35
                size = int(old_size * final_crop_scale)
                src_pts_2 = np.array([
                    [center[0]-size/2, center[1]-size*3/5], 
                    [center[0]-size/2, center[1]+size*2/5], 
                    [center[0]+size/2, center[1]-size*3/5],
                    [center[0]+size/2, center[1]+size*2/5], 
                    ])
                dst_pts_2 = np.array([
                    [0, 0], 
                    [0, self.target_size[1]-1], 
                    [self.target_size[0]-1, 0],
                    [self.target_size[0]-1, self.target_size[1]-1],
                    ])
                tform_2 = estimate_transform('similarity', src_pts_2, dst_pts_2)
                final_crop_trans_matx = tform_2.params
                final_size = self.target_size

            batch_transforms.append(final_crop_trans_matx)
            final_sizes.append(final_size)
            batch_recover_transforms.append(np.matmul(final_crop_trans_matx, np.linalg.inv(deca_crop_trans_matx)))

            # compute face alignment bbox on final image space
            bbox = np.array([(left, top), (right, bottom)])
            batch_bboxes.append(tform_2(bbox).reshape((-1,)))

        # run DECA: encode, decode and render
        with torch.no_grad():
            codedict = self.deca.encode(
                torch.as_tensor(batch_crop_imgs, device=self.device0, dtype=torch.float32).permute(0, 3, 1, 2) / 255.,
                use_detail=self.use_detail,
                )

            final_imgs = []
            for j in range(len(batch_imgs)):
                final_size = final_sizes[j]
                final_img = cv2.warpAffine(batch_imgs[j], batch_transforms[j][:2], final_size)
                final_imgs.append(final_img)

        codes = dict([(k, v.detach().cpu().numpy()) for k, v in codedict.items()])

        # interpretation of each code parameter
        # shape & exp have shape 100 and 50 respectively, and they are just PCA coefficients. Typically their value range is [-2, 2]
        # pose has shape 6, pose[:3] is global head rotation in axis-angle format; pose[3:6] is jaw pose in axis-angle format
        # cam has shape 3, consisting of [scale, translation_x, translation_y], which is used in ortho projection from world to image: proj_xy = scale * (xy + translation_xy)

        ###
        #(Pdb) codes['shape'].shape
        #(100,)
        #(Pdb) codes['exp'].shape
        #(50,)
        #(Pdb) codes['tex'].shape
        #(50,)
        #(Pdb) codes['pose'].shape
        #(6,)
        #(Pdb) codes['cam'].shape
        #(3,)
        #(Pdb) codes['light'].shape
        #(9, 3)

        preprocessings = {
            'transforms': batch_transforms,
            'recover_transforms': batch_recover_transforms,
            'bboxes': batch_bboxes,
            'final_sizes': final_sizes,
        }

        return np.stack(final_imgs, axis=0), codes, preprocessings
    
    def decode_and_post_align(self, codes, preprocessings):
        """
        This function takes the codes from encoder and get decoded outputs, which go through some post processing to align the outputs on target image space
        """
        codedict = dict([(k, torch.as_tensor(v, device=self.device0)) for k, v in codes.items()]) 

        batch_recover_transforms = preprocessings['recover_transforms']
        final_sizes = preprocessings['final_sizes']

        # run DECA: decode and render
        with torch.no_grad():
            opdict = self.deca.decode(
                codedict,
                rendering=False, iddict=None, vis_lmk=False, return_vis=False, use_detail=self.use_detail,
                )

            # align output to uncropped image space
            points_scale= [self.deca.image_size, self.deca.image_size]
            tforms = torch.as_tensor(batch_recover_transforms, device=self.device0, dtype=torch.float32)
            tforms = tforms.transpose(1,2)
            # when out_scale is None, returned points are not normalized
            opdict['trans_verts'] = transform_points(opdict['trans_verts'], tforms, points_scale, out_scale=None) 
            opdict['landmarks2d'] = transform_points(opdict['landmarks2d'], tforms, points_scale, out_scale=None)
            opdict['landmarks3d'] = transform_points(opdict['landmarks3d'], tforms, points_scale, out_scale=None)

            # render face image
            shaded_faces = []
            for j in range(opdict['trans_verts'].shape[0]):
                final_size = final_sizes[j]
                trans_verts = opdict['trans_verts'][j:j+1].clone()
                trans_verts[..., 0] = trans_verts[..., 0] / final_size[0] * 2 - 1
                trans_verts[..., 1] = trans_verts[..., 1] / final_size[1] * 2 - 1
                shaded_face = self.mesh_renderer(trans_verts, final_size, world_verts=opdict['verts'], lights=codedict['light']) 
                shaded_faces.append(shaded_face)

        outputs = dict([(k, v.detach().cpu().numpy()) for k, v in opdict.items()])
        outputs['shaded_face'] = torch.cat(shaded_faces, dim=0).detach().cpu().numpy()

        return outputs