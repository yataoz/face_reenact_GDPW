import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import json
import functools
from scipy.io import loadmat
import pickle as pkl

import numpy as np
import nvdiffrast.torch as dr

from .layers import UpBlock2D, DownBlock2D, SameBlock2D, SpadeResBlock2D, ResBlock2D
from .utils import make_coordinate_grid_2d, backward_warp, AntiAliasDownsample2d

import pdb

class MeshRenderer(nn.Module):
    """
    This class doesn't have any trainable parameters
    """
    def __init__(self, glctx_pool):
        super(MeshRenderer, self).__init__()

        #flame_model = np.load('FLAME/generic_model.npz')    # same model as generic_model.pkl but saved in .npz format
        with open('FLAME/generic_model.pkl', 'rb') as f:
            flame_model = pkl.load(f, encoding='latin1')
        tri = flame_model['f'].astype(np.int32)
        self.num_verts = flame_model['v_template'].shape[0]

        self.register_buffer('tri', torch.tensor(tri, dtype=torch.int32))
        self.glctx_pool = glctx_pool
    
    def create_gl_context_pool(self, devices):
        self.glctx_pool = dict()
        for device in devices:
            self.glctx_pool[torch.device(device)] = dr.RasterizeGLContext(device=device)

    def face_vertices(self, vertices, faces):
        """ 
        borrowed from DECA.utils.util.py

        :param vertices: [batch size, number of vertices, 3]
        :param faces: [batch size, number of faces, 3]
        :return: [batch size, number of faces, 3, 3]
        """
        assert (vertices.ndimension() == 3)
        assert (faces.ndimension() == 3)
        assert (vertices.shape[0] == faces.shape[0])
        assert (vertices.shape[2] == 3)
        assert (faces.shape[2] == 3)

        bs, nv = vertices.shape[:2]
        bs, nf = faces.shape[:2]
        device = vertices.device
        faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        vertices = vertices.reshape((bs * nv, 3))
        # pytorch only supports long and byte tensors for indexing
        return vertices[faces.long()]
    
    def find_fg_vertices(self, verts, viewport_size):
        # rasterize source mesh to find out which triangle faces are visible
        glctx = self.glctx_pool[verts.device]
        assert isinstance(glctx, dr.RasterizeGLContext)

        # must use relatively large viewport_size, otherwise many triangles will be discarded during rasterization,
        # making the visible vertices not present in the rasterized image.
        viewport_size = [max(256, s) for s in viewport_size]

        batch = verts.shape[0]
        num_verts = verts.shape[1]
        z = verts[..., 2:3]
        min_z = z.amin(dim=1, keepdims=True)
        max_z = z.amax(dim=1, keepdims=True)
        normed_z = 2 * (z - min_z) / (max_z - min_z + 1.e-8) - 1    # normalized to [-1, 1] because nvdiffrast requires depth to be in [-1, 1]
        pos = torch.cat([verts[..., :2], normed_z, torch.ones_like(normed_z)], dim=-1) 
        rast, _ = dr.rasterize(glctx, pos, self.tri, resolution=viewport_size[::-1])    # (b, h, w, 4)
        verts_fg_mask = torch.zeros_like(verts[..., :1])   # (b, num_verts, 1). Mask on vertices indicating whether it's a visible after rasterization
        for i in range(batch):
            rast_face_ids = rast[i, ..., -1].long().view(-1) - 1
            fg_faces_ids = torch.unique(rast_face_ids)     # (num_fg_faces, )
            fg_tri_verts = self.tri[fg_faces_ids]   # (num_fg_faces, 3)
            ## hard mask
            verts_fg_mask[i, :, 0] = torch.clip(
                torch.bincount(fg_tri_verts.view(-1), minlength=num_verts).float(), 
                0, 1)
            ### soft mask
            #bc = torch.bincount(fg_tri_verts.view(-1), minlength=num_verts).float()
            #maxv = bc.max()
            #verts_fg_mask[i, :, 0] = bc / (maxv + 1.e-8)    # normalize to [0, 1]
        return verts_fg_mask
    
    def render(self, verts, verts_attr, viewport_size):
        z = verts[..., 2:3]
        min_z = z.amin(dim=1, keepdims=True)
        max_z = z.amax(dim=1, keepdims=True)
        normed_z = 2 * (z - min_z) / (max_z - min_z + 1.e-8) - 1    # normalized to [-1, 1] because nvdiffrast requires depth to be in [-1, 1]
        pos = torch.cat([verts[..., :2], normed_z, torch.ones_like(normed_z)], dim=-1) 

        glctx = self.glctx_pool[verts.device]
        assert isinstance(glctx, dr.RasterizeGLContext)
        rast, _ = dr.rasterize(glctx, pos, self.tri, resolution=viewport_size[::-1])    # (b, h, w, c)

        # back face culling
        batch_size = verts.shape[0]
        face_verts = self.face_vertices(verts, self.tri.unsqueeze(0).repeat(batch_size, 1, 1))    # (b, num_tri, 3, 3)
        v0 = face_verts[..., 0, :]
        v1 = face_verts[..., 1, :]
        v2 = face_verts[..., 2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0)    # (b, num_tri, 3)
        face_cull_mask = face_normals[..., 2] > 0      # (b, num_tri)
        for i in range(rast.shape[0]):
            tri_id = rast[i, ..., 3] - 1   # nvdiffrast rast triangle_id is offset by 1, so we subtract 1 here. (h, w)  
            rast_cull_mask = face_cull_mask[i][tri_id.long()]  # (h, w)
            rast[i, ..., 3][rast_cull_mask] = 0     # set to 0 means no face to be rasterized

        interp_attr, _ =dr.interpolate(verts_attr, rast, self.tri)    # (b, view_h, view_w, c)
        render_attr = dr.antialias(interp_attr, rast, pos, self.tri)    # (b, view_h, view_w, c)
        return render_attr, rast
    
    def __call__(self, verts_s, verts_d, embeds, viewport_size):
        """
        verts_s/d: source/target mesh vertices. (b, num_verts, 3). xy in [-1, 1], z is world depth
        embeds: mesh template embeds. (b, num_verts, embed_dim)
        """
        for s in viewport_size:
            assert isinstance(s, int)
        
        if embeds is None:
            embeds = torch.zeros(verts_s.shape[0], self.num_verts, 0, dtype=verts_s.dtype, device=verts_s.device)   # a placeholder

        # render on the target image plane
        verts_fg_mask = self.find_fg_vertices(verts_s, viewport_size)
        verts_flow_d2s = verts_s[..., :2] - verts_d[..., :2]    # (b, num_verts, 2)        
        verts_attr = torch.cat([verts_flow_d2s, verts_fg_mask, embeds], dim=-1)
        render_attr, rast = self.render(verts_d, verts_attr, viewport_size)
        render_flow_d2s, render_fg_mask, render_embeds_d = torch.split(render_attr, [2, 1, embeds.shape[-1]], dim=-1)

        # render on the source image plane
        if embeds.numel() > 0:
            render_embeds_s, rast = self.render(verts_s, embeds, viewport_size)
        else:
            render_embeds_s = None
            render_embeds_d = None

        return render_flow_d2s, render_embeds_d, render_embeds_s

class Generator(nn.Module):
    embed_dim = 16
    base_chns = 64
    max_chns = 1024

    def __init__(self, model_cfg):
        super(Generator, self).__init__()
        self.num_levels = model_cfg.NUM_LEVELS      # (num_levels - 1) downsampling/upsampling layers
        self.alpha_blend = model_cfg.USE_ALPHA_BLEND
        self.shrink_ratio = model_cfg.SHRINK_RATIO
        self.guidance = model_cfg.GUIDANCE

        self.mesh_renderer = MeshRenderer(None)

        NF = lambda stride: min(self.max_chns, max(int(self.base_chns * self.shrink_ratio * stride), 16))
        flow_chns = 2
        inp_chns = 3 + (self.embed_dim if self.guidance in ['neural_codes', 'both'] else 0)
        spade_chns = 0 
        spade_chns += self.embed_dim if self.guidance in ['neural_codes', 'both'] else 0
        spade_chns += flow_chns if self.guidance in ['geom_disp', 'both'] else 0
        downsamp_chns = spade_chns

        self.down_layers = OrderedDict()
        self.shortcut_layers = OrderedDict()
        self.downsamples = OrderedDict()
        for i in range(self.num_levels):
            s = 2**i
            reso = 'x{}'.format(s)
            if reso == 'x1':
                self.down_layers[reso] = SameBlock2D(inp_chns, NF(s), ksize=3, padding=1)
            else:
                self.down_layers[reso] = DownBlock2D(NF(s // 2), NF(s), ksize=3, padding=1)
            self.shortcut_layers[reso] = SameBlock2D(NF(s), NF(s), ksize=3, padding=1)
            self.downsamples[reso] = AntiAliasDownsample2d(downsamp_chns, 1/s)
        self.down_layers = nn.ModuleDict(self.down_layers)
        self.shortcut_layers = nn.ModuleDict(self.shortcut_layers)
        self.downsamples = nn.ModuleDict(self.downsamples)

        self.up_layers = OrderedDict()
        for i in range(self.num_levels - 1, -1, -1):
            s = 2**i
            reso = 'x{}'.format(s)
            if reso == 'x1':
                self.up_layers[reso+'_out'] = nn.Conv2d(NF(s) * 2, 4, kernel_size=3, padding=1)
            else:
                self.up_layers[reso+'_spade'] = SpadeResBlock2D(NF(s) * 2, NF(s), spade_chns, NF(s))
                self.up_layers[reso+'_flow'] = nn.Conv2d(NF(s), flow_chns, kernel_size=3, padding=1)
                self.up_layers[reso+'_up'] = UpBlock2D(NF(s), NF(s//2), ksize=3, padding=1)     # upsample from reso to twice reso
        self.up_layers = nn.ModuleDict(self.up_layers)

        if self.guidance in ['neural_codes', 'both']:
            self.embeds = nn.Parameter(torch.rand(self.mesh_renderer.num_verts, self.embed_dim, dtype=torch.float32))
    
    def forward(self, img_s, verts_s, verts_d):
        """
        img_s: (b, 3, h, w)
        verts_s/d: (b, num_verts, 3)
        """
        # create mesh flow
        if self.guidance in ['neural_codes', 'both']:
            embeds = self.embeds.unsqueeze(0).repeat(verts_s.shape[0], 1, 1)
        else:
            embeds = None
        mesh_flow, face_tem_d, face_tem_s = self.mesh_renderer(verts_s, verts_d, embeds, img_s.shape[2:])
        mesh_flow_d2s = mesh_flow.permute(0, 3, 1, 2)
        if self.guidance in ['neural_codes', 'both']:
            face_tem_d = face_tem_d.permute(0, 3, 1, 2)
            face_tem_s = face_tem_s.permute(0, 3, 1, 2)

        # encoder
        feats = OrderedDict()
        if self.guidance in ['neural_codes', 'both']:
            out = torch.cat([img_s, face_tem_s], dim=1)
        else:
            out = img_s
        for reso, layer in self.down_layers.items():
            out = layer(out) 
            feats[reso] = out

        # resize flows and modulation inputs
        if self.guidance == 'geom_disp':
            mod = mesh_flow_d2s 
        elif self.guidance == 'neural_codes':
            mod = face_tem_d
        elif self.guidance == 'both':
            mod = torch.cat([face_tem_d, mesh_flow_d2s], dim=1)
        else:
            raise ValueError("Unrecognized guidance={}".format(self.guidance))
        mods = OrderedDict({'x1': mod})
        mesh_flows = OrderedDict({'x1': mesh_flow_d2s})
        for reso, layer in self.down_layers.items():
            if reso not in mods.keys():
                mods[reso] = self.downsamples[reso](mod)
            if reso not in mesh_flows.keys():
                if self.guidance == 'geom_disp':
                    mesh_flows[reso] = mods[reso]
                elif self.guidance == 'neural_codes':
                    mesh_flows[reso] = torch.zeros_like(mods[reso][:, :2])
                elif self.guidance == 'both':
                    mesh_flows[reso] = mods[reso][:, self.embed_dim:self.embed_dim+2]
                else:
                    raise ValueError("Unrecognized guidance={}".format(self.guidance))
        
        # warp the features at bottleneck (smallest resolution)
        max_stride = 2 ** (self.num_levels - 1)
        min_reso = 'x{}'.format(max_stride)
        warp_out = backward_warp(out, mesh_flows[min_reso])
        out = torch.cat([out, warp_out], dim=1)
        
        # decoder
        flow = None     # predicted flow
        flows = OrderedDict()
        for name, layer in self.up_layers.items():
            reso = name.split('_')[0]

            if 'spade' in name or 'out' in name:
                # warp encoder features
                if flow is not None:
                    sc = self.shortcut_layers[reso](feats[reso])
                    flows[reso] = flow
                    sc = backward_warp(sc, flow)
                    out = torch.cat([out, sc], dim=1)

                if 'spade' in name:
                    out = layer(out, mods[reso])
                else:
                    out = layer(out)

            elif 'flow' in name:
                tmp = layer(out)
                tmp = F.interpolate(tmp, scale_factor=2, mode='bilinear')    # flow output at reso, and upsampled to twice reso
                h, w = tmp.shape[2:]
                flow = 2 * torch.cat([tmp[:, :1] / (w - 1), tmp[:, 1:2] / (h - 1)], dim=1)
            else:
                out = layer(out)

        out_img, alpha = torch.split(out, [3, 1], dim=1)
        out_img = torch.sigmoid(out_img)
        alpha =torch.sigmoid(alpha)

        warp_img_s = backward_warp(img_s, flow)

        if self.alpha_blend:
            rec_img_d = out_img * alpha + warp_img_s * (1 - alpha)
        else:
            rec_img_d = out_img
        
        # save face template for it to be used outside forward()
        self.face_tem_d = face_tem_d
        self.face_tem_s = face_tem_s

        return mesh_flows, flows, warp_img_s, out_img, alpha, rec_img_d

class Discriminator(nn.Module):
    base_chns = 32
    max_chns = 512

    def __init__(self, logits_layers, dropout, use_bg_mask, num_blocks=4, shrink_ratio=1):
        super(Discriminator, self).__init__()
        self.dropout = dropout
        self.shrink_ratio = shrink_ratio

        NF = max(int(self.base_chns * self.shrink_ratio), 16)
        chns = [3] + [min(NF * (2**i), self.max_chns) for i in range(num_blocks)] 
        if use_bg_mask:
            chns[0] += 1
        layers = [DownBlock2D(chns[i], chns[i+1], ksize=4, padding=2) for i in range(num_blocks)]    # 'num_blocks' downsampling blocks
        self.layers = nn.ModuleList(layers)
        if self.dropout:
            self.drop_rates = list(np.linspace(0, 0.5, num_blocks))

        self.logits_layers = logits_layers
        out_convs = []
        for i in logits_layers:
            assert i < 0, "logits_layer index should be negative, counting backwards."
            out_convs.append(nn.Conv2d(chns[i], 1, kernel_size=4, padding=2))
        self.out_convs = nn.ModuleList(out_convs)

    def forward(self, x, condition):
        """
        x: img
        """
        assert condition is None, "No conditional input should be used"
        out = x
        feats = []
        for i, layer in enumerate(self.layers):
            out = layer(out)
            feats.append(out)
            if self.dropout:
                out = F.dropout(out, self.drop_rates[i])
        
        logits = []
        for i, j in enumerate(self.logits_layers):
            out = self.out_convs[i](feats[j])
            logits.append(out)
        return feats, logits

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales, logits_layers, dropout, use_bg_mask, num_blocks=4, shrink_ratio=1):
        super(MultiScaleDiscriminator, self).__init__()
        if not isinstance(scales, (tuple, list)):
            scales = [scales]
        discs = dict()
        downs = dict()
        for scale in scales:
            assert scale in ['x16', 'x8', 'x4', 'x2', 'x1']
            discs[scale] = Discriminator(logits_layers, dropout, use_bg_mask, num_blocks, shrink_ratio)
            downs[scale] = AntiAliasDownsample2d(3, 1./float(scale.replace('x', '')))
        self.discs = nn.ModuleDict(discs)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x, condition):
        assert condition is None, "No conditional input should be used"
        all_feats, all_logits =[], []
        for scale, disc in self.discs.items():
            downsample = self.downs[scale]
            x_s = downsample(x)
            condition_s = None if condition is None else downsample(condition)
            feats_s, logits_s = disc(x_s, condition_s)
            all_feats += feats_s 
            all_logits += logits_s 
        return all_feats, all_logits
    
    def __getitem__(self, k):
        return self.discs[k]

    def __contains__(self, k):
        return k in self.discs.keys()


