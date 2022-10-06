import torch
from torch import nn
import torch.nn.functional as F

import os
import cv2
import argparse
import numpy as np    
import copy
from collections import defaultdict
from tensorpack.utils import viz
from scipy.spatial.transform import Rotation
import re
import random

from config import config as cfg
from tester import Tester

import pdb

def update_codes_from_cmd(codes):
    codes0 = copy.deepcopy(codes)
    while 1:
        proceed = False
        while not proceed: 
            # =============== parse current value
            # interpretation of each code parameter
            # shape & exp have shape 100 and 50 respectively, and they are just PCA coefficients. Typically their value range is [-2, 2]
            # pose has shape 6, pose[:3] is global head rotation in axis-angle format; pose[3:6] is jaw pose in axis-angle format
            # cam has shape 3, consisting of [scale, translation_x, translation_y], which is used in ortho projection from world to image: proj_xy = scale * (xy + translation_xy)
            msg = '\033[92m\nCurrent values:\n'
            for i in range(10):
                msg += 'shape {}: {:.2f}\n'.format(i, codes['shape'][0, i])
            msg += '\n'

            for i in range(10):
                msg += 'exp {}: {:.2f}\n'.format(i, codes['exp'][0, i])
            msg += '\n' 

            global_pose = codes['pose'][0, :3]
            R = Rotation.from_rotvec(global_pose)
            global_yaw_pitch_roll = R.as_euler('yxz', degrees=True)
            msg += 'global_pose, yaw pitch roll: {}\n'.format(global_yaw_pitch_roll.round(2))

            jaw_pose = codes['pose'][0, 3:6]
            R = Rotation.from_rotvec(jaw_pose)
            jaw_yaw_pitch_roll = R.as_euler('yxz', degrees=True)
            msg += 'jaw_pose, yaw pitch roll: {}\n'.format(jaw_yaw_pitch_roll.round(2))

            leye_pose = codes['eye_pose'][0, :3]
            R = Rotation.from_rotvec(leye_pose)
            leye_yaw_pitch_roll = R.as_euler('yxz', degrees=True)
            msg += 'leye_pose, yaw pitch roll: {}\n'.format(leye_yaw_pitch_roll.round(2))

            reye_pose = codes['eye_pose'][0, 3:6]
            R = Rotation.from_rotvec(reye_pose)
            reye_yaw_pitch_roll = R.as_euler('yxz', degrees=True)
            msg += 'reye_pose, yaw pitch roll: {}\n'.format(reye_yaw_pitch_roll.round(2))

            cam = codes['cam'][0]
            msg += 'camera scale, tx, ty: {}\n'.format(cam.round(2))
            msg += '\033[0m \n' 
            print(msg)

            try:
                # ============== change value
                cmd = input("\033[93m\nChange the value \n E.g. reset or \n <shape or exp> <dim> <value> \n <global_pose or jaw_pose> <yaw, pitch or roll> <angle> or \n <cam> <scale, tx or ty> <value> \n \033[0m")
                if cmd == 'reset':
                    codes = copy.deepcopy(codes0)
                else:
                    splits = cmd.split(' ')
                    assert len(splits) % 3 == 0, "number command line inputs must be multiple of 3, separated by space ' '"
                    num_changes = len(splits) // 3
                    for i in range(num_changes):
                        param_name = splits[3*i]
                        assert param_name in ['shape', 'exp', 'global_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'cam'], "parameter name must be in ['shape', 'exp', 'global_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'cam']"
                        if param_name in ['shape', 'exp']:
                            dim = int(splits[3*i+1])
                            assert dim < codes[param_name][0].shape[0], "dim must be smaller than {} for {}".format(codes[param_name][0].shape[0], param_name)
                            value = float(splits[3*i+2])
                            assert value >= -2 and value <= 2, "provided {} value {} is out of range; must be in [-2, 2]".format(param_name, value)
                            codes[param_name][0, dim] = value
                        elif param_name in ['global_pose', 'jaw_pose']:
                            pose_ref = codes['pose'][0, :3] if param_name == 'global_pose' else codes['pose'][0, 3:6]
                            R = Rotation.from_rotvec(pose_ref)
                            yaw_pitch_roll = R.as_euler('yxz', degrees=True)
                            axis = splits[3*i+1]
                            assert axis in ['yaw', 'pitch', 'roll'], "axis must be yaw, pitch or roll for {}. But got {}".format(param_name, axis)
                            angle = float(splits[3*i+2])
                            if axis == 'yaw':
                                yaw_pitch_roll[0] = angle 
                            elif axis == 'pitch':
                                yaw_pitch_roll[1] = angle 
                            elif axis == 'roll':
                                yaw_pitch_roll[2] = angle 
                            if param_name == 'global_pose':
                                codes['pose'][0, :3] = Rotation.from_euler('yxz', yaw_pitch_roll, degrees=True).as_rotvec()
                            elif param_name == 'jaw_pose':
                                codes['pose'][0, 3:6] = Rotation.from_euler('yxz', yaw_pitch_roll, degrees=True).as_rotvec()
                        elif param_name in ['leye_pose', 'reye_pose']:
                            pose_ref = codes['eye_pose'][0, :3] if param_name == 'leye_pose' else codes['eye_pose'][0, 3:6]
                            R = Rotation.from_rotvec(pose_ref)
                            yaw_pitch_roll = R.as_euler('yxz', degrees=True)
                            axis = splits[3*i+1]
                            assert axis in ['yaw', 'pitch', 'roll'], "axis must be yaw, pitch or roll for {}. But got {}".format(param_name, axis)
                            angle = float(splits[3*i+2])
                            if axis == 'yaw':
                                yaw_pitch_roll[0] = angle 
                            elif axis == 'pitch':
                                yaw_pitch_roll[1] = angle 
                            elif axis == 'roll':
                                yaw_pitch_roll[2] = angle 
                            if param_name == 'leye_pose':
                                codes['eye_pose'][0, :3] = Rotation.from_euler('yxz', yaw_pitch_roll, degrees=True).as_rotvec()
                            elif param_name == 'reye_pose':
                                codes['eye_pose'][0, 3:6] = Rotation.from_euler('yxz', yaw_pitch_roll, degrees=True).as_rotvec()
                        elif param_name == 'cam':
                            comp = splits[3*i+1]
                            assert comp in ['scale', 'tx', 'ty'], "component must be scale, tx or ty for {}. But got {}".format(param_name, comp)
                            value = float(splits[3*i+2])
                            if comp == 'scale':
                                codes['cam'][0, 0] = value
                            elif comp == 'tx':
                                codes['cam'][0, 1] = value
                            elif comp == 'ty':
                                codes['cam'][0, 2] = value
            except AssertionError as e:
                print('\033[95mError: {}\033[0m'.format(e))
            else:
                proceed = True
        yield codes

def update_codes_from_preset(codes):

    def _gen_from_interps(c, n, target_vals):
        """
        c: codes
        n: num of interpolations
        target_vals: target values
        """
        dim = c.shape[0]
        if isinstance(target_vals, (int, float)):
            target_vals = np.ones((dim,), dtype=np.float32) * target_vals
        assert len(c) == len(target_vals)
        interp_vals = []
        for i in range(dim):
            interp_vals.append(np.linspace(c[i], target_vals[i], num_interps))
        interp_vals = np.array(interp_vals)     # (num_dims, num_interps)
        for k in range(num_interps):
            for i in range(dim):
                # codes are modified inplace
                c[i] = interp_vals[i, k]
            yield c


    codes0 = copy.deepcopy(codes)

    num_interps = 40

    # shape & exp
    num_rounds = 2
    for name in ['shape', 'exp']:
        dim = codes[name][0].shape[0]
        first_n_components = dim     # the rest components won't be that visually important because their PCA coefficients are small
        assert first_n_components <= dim
        for i in range(num_rounds):
            dims_to_update = np.arange(first_n_components)[i::num_rounds]

            # forward
            # sub_codes will be modified inplace and output as new_sub_codes
            sub_codes = codes[name][0, dims_to_update]
            for new_sub_codes in _gen_from_interps(sub_codes, num_interps, -2):
                codes[name][0, dims_to_update] = new_sub_codes
                yield codes
            # backward
            # sub_codes will be modified inplace and output as new_sub_codes
            target_codes = codes0[name][0, dims_to_update]
            for new_sub_codes in _gen_from_interps(sub_codes, num_interps, target_codes):
                codes[name][0, dims_to_update] = new_sub_codes
                yield codes
    
    # jaw pose
    max_angle = 8
    pose = codes['pose'][0, 3:6]
    R = Rotation.from_rotvec(pose)
    yaw_pitch_roll = R.as_euler('yxz', degrees=True)
    yaw_pitch_roll0 = yaw_pitch_roll.copy()
    for i in range(3):  
        for direction in [-1, 1]:
            # forward
            # yaw_pitch_roll will be modified inplace and output as new_yaw_pitch_roll
            target_angle = yaw_pitch_roll0[i:i+1] + direction * max_angle
            angle = yaw_pitch_roll[i:i+1]
            for new_angle in _gen_from_interps(angle, num_interps, target_angle):
                yaw_pitch_roll[i:i+1] = new_angle
                codes['pose'][0, 3:6] = Rotation.from_euler('yxz', yaw_pitch_roll, degrees=True).as_rotvec()
                yield codes
            # backward
            # yaw_pitch_roll will be modified inplace and output as new_yaw_pitch_roll
            target_angle = yaw_pitch_roll0[i]
            angle = yaw_pitch_roll[i:i+1]
            for new_angle in _gen_from_interps(angle, num_interps, target_angle):
                yaw_pitch_roll[i:i+1] = new_angle
                codes['pose'][0, 3:6] = Rotation.from_euler('yxz', yaw_pitch_roll, degrees=True).as_rotvec()
                yield codes

    # head pose
    max_angle = 15
    pose = codes['pose'][0, :3]
    R = Rotation.from_rotvec(pose)
    yaw_pitch_roll = R.as_euler('yxz', degrees=True)
    yaw_pitch_roll0 = yaw_pitch_roll.copy()
    for i in range(3):  
        for direction in [-1, 1]:
            # forward
            # yaw_pitch_roll will be modified inplace and output as new_yaw_pitch_roll
            target_angle = yaw_pitch_roll0[i:i+1] + direction * max_angle
            angle = yaw_pitch_roll[i:i+1]
            for new_angle in _gen_from_interps(angle, num_interps, target_angle):
                yaw_pitch_roll[i:i+1] = new_angle
                codes['pose'][0, :3] = Rotation.from_euler('yxz', yaw_pitch_roll, degrees=True).as_rotvec()
                yield codes
            # backward
            # yaw_pitch_roll will be modified inplace and output as new_yaw_pitch_roll
            target_angle = yaw_pitch_roll0[i]
            angle = yaw_pitch_roll[i:i+1]
            for new_angle in _gen_from_interps(angle, num_interps, target_angle):
                yaw_pitch_roll[i:i+1] = new_angle
                codes['pose'][0, :3] = Rotation.from_euler('yxz', yaw_pitch_roll, degrees=True).as_rotvec()
                yield codes

    ## eyeball wink
    #num_interps = 5
    #max_angle = 30
    #leye_pose = codes['eye_pose'][0, :3]
    #leye_R = Rotation.from_rotvec(leye_pose)
    #leye_yaw_pitch_roll = leye_R.as_euler('yxz', degrees=True)
    #leye_yaw_pitch_roll0 = leye_yaw_pitch_roll.copy()
    #reye_pose = codes['eye_pose'][0, 3:6]
    #reye_R = Rotation.from_rotvec(reye_pose)
    #reye_yaw_pitch_roll = reye_R.as_euler('yxz', degrees=True)
    #reye_yaw_pitch_roll0 = reye_yaw_pitch_roll.copy()

    ### wink 3 times both eyes
    ##times = 3
    ##for _ in range(times):
    ##    i = 1
    ##    direction = 1
    ##    # forward
    ##    # yaw_pitch_roll will be modified inplace and output as new_yaw_pitch_roll
    ##    leye_target_angle = leye_yaw_pitch_roll0[i:i+1] + direction * max_angle
    ##    leye_angle = leye_yaw_pitch_roll[i:i+1]
    ##    reye_target_angle = reye_yaw_pitch_roll0[i:i+1] + direction * max_angle
    ##    reye_angle = reye_yaw_pitch_roll[i:i+1]
    ##    angle = np.concatenate([leye_angle, reye_angle], axis=0)
    ##    target_angle = np.concatenate([leye_target_angle, reye_target_angle], axis=0)
    ##    for new_angle in _gen_from_interps(angle, num_interps, target_angle):
    ##        leye_yaw_pitch_roll[i:i+1] = new_angle[0]
    ##        codes['eye_pose'][0, :3] = Rotation.from_euler('yxz', leye_yaw_pitch_roll, degrees=True).as_rotvec()
    ##        reye_yaw_pitch_roll[i:i+1] = new_angle[1]
    ##        codes['eye_pose'][0, 3:6] = Rotation.from_euler('yxz', reye_yaw_pitch_roll, degrees=True).as_rotvec()
    ##        yield codes
    ##    # backward
    ##    # yaw_pitch_roll will be modified inplace and output as new_yaw_pitch_roll
    ##    leye_target_angle = leye_yaw_pitch_roll0[i:i+1]
    ##    leye_angle = leye_yaw_pitch_roll[i:i+1]
    ##    reye_target_angle = reye_yaw_pitch_roll0[i:i+1]
    ##    reye_angle = reye_yaw_pitch_roll[i:i+1]
    ##    angle = np.concatenate([leye_angle, reye_angle], axis=0)
    ##    target_angle = np.concatenate([leye_target_angle, reye_target_angle], axis=0)
    ##    for new_angle in _gen_from_interps(angle, num_interps, target_angle):
    ##        leye_yaw_pitch_roll[i:i+1] = new_angle[0]
    ##        codes['eye_pose'][0, :3] = Rotation.from_euler('yxz', leye_yaw_pitch_roll, degrees=True).as_rotvec()
    ##        reye_yaw_pitch_roll[i:i+1] = new_angle[1]
    ##        codes['eye_pose'][0, 3:6] = Rotation.from_euler('yxz', reye_yaw_pitch_roll, degrees=True).as_rotvec()
    ##        yield codes
    ## wink 1 time left eye
    #times = 1
    #is_left_eye = True
    #for _ in range(times):
    #    i = 1
    #    direction = 1
    #    # forward
    #    # yaw_pitch_roll will be modified inplace and output as new_yaw_pitch_roll
    #    if is_left_eye:
    #        target_angle = leye_yaw_pitch_roll0[i:i+1] + direction * max_angle
    #        angle = leye_yaw_pitch_roll[i:i+1]
    #    else:
    #        target_angle = reye_yaw_pitch_roll0[i:i+1] + direction * max_angle
    #        angle = reye_yaw_pitch_roll[i:i+1]
    #    for new_angle in _gen_from_interps(angle, num_interps, target_angle):
    #        if is_left_eye:
    #            leye_yaw_pitch_roll[i:i+1] = new_angle
    #            codes['eye_pose'][0, :3] = Rotation.from_euler('yxz', leye_yaw_pitch_roll, degrees=True).as_rotvec()
    #        else:
    #            reye_yaw_pitch_roll[i:i+1] = new_angle
    #            codes['eye_pose'][0, 3:6] = Rotation.from_euler('yxz', reye_yaw_pitch_roll, degrees=True).as_rotvec()
    #        yield codes
    #    # backward
    #    # yaw_pitch_roll will be modified inplace and output as new_yaw_pitch_roll
    #    if is_left_eye:
    #        target_angle = leye_yaw_pitch_roll0[i:i+1]
    #        angle = leye_yaw_pitch_roll[i:i+1]
    #    else:
    #        target_angle = reye_yaw_pitch_roll0[i:i+1]
    #        angle = reye_yaw_pitch_roll[i:i+1]
    #    for new_angle in _gen_from_interps(angle, num_interps, target_angle):
    #        if is_left_eye:
    #            leye_yaw_pitch_roll[i:i+1] = new_angle
    #            codes['eye_pose'][0, :3] = Rotation.from_euler('yxz', leye_yaw_pitch_roll, degrees=True).as_rotvec()
    #        else:
    #            reye_yaw_pitch_roll[i:i+1] = new_angle
    #            codes['eye_pose'][0, 3:6] = Rotation.from_euler('yxz', reye_yaw_pitch_roll, degrees=True).as_rotvec()
    #        yield codes

    ## eyeball rotation of all angles
    #max_angle = 30
    #leye_pose = codes['eye_pose'][0, :3]
    #leye_R = Rotation.from_rotvec(leye_pose)
    #leye_yaw_pitch_roll = leye_R.as_euler('yxz', degrees=True)
    #leye_yaw_pitch_roll0 = leye_yaw_pitch_roll.copy()
    #reye_pose = codes['eye_pose'][0, 3:6]
    #reye_R = Rotation.from_rotvec(reye_pose)
    #reye_yaw_pitch_roll = reye_R.as_euler('yxz', degrees=True)
    #reye_yaw_pitch_roll0 = reye_yaw_pitch_roll.copy()
    #for i in range(3):  
    #    for direction in [-1, 1]:
    #        # forward
    #        # yaw_pitch_roll will be modified inplace and output as new_yaw_pitch_roll
    #        leye_target_angle = leye_yaw_pitch_roll0[i:i+1] + direction * max_angle
    #        leye_angle = leye_yaw_pitch_roll[i:i+1]
    #        reye_target_angle = reye_yaw_pitch_roll0[i:i+1] + direction * max_angle
    #        reye_angle = reye_yaw_pitch_roll[i:i+1]
    #        angle = np.concatenate([leye_angle, reye_angle], axis=0)
    #        target_angle = np.concatenate([leye_target_angle, reye_target_angle], axis=0)
    #        for new_angle in _gen_from_interps(angle, num_interps, target_angle):
    #            leye_yaw_pitch_roll[i:i+1] = new_angle[0]
    #            codes['eye_pose'][0, :3] = Rotation.from_euler('yxz', leye_yaw_pitch_roll, degrees=True).as_rotvec()
    #            reye_yaw_pitch_roll[i:i+1] = new_angle[1]
    #            codes['eye_pose'][0, 3:6] = Rotation.from_euler('yxz', reye_yaw_pitch_roll, degrees=True).as_rotvec()
    #            yield codes
    #        # backward
    #        # yaw_pitch_roll will be modified inplace and output as new_yaw_pitch_roll
    #        leye_target_angle = leye_yaw_pitch_roll0[i:i+1]
    #        leye_angle = leye_yaw_pitch_roll[i:i+1]
    #        reye_target_angle = reye_yaw_pitch_roll0[i:i+1]
    #        reye_angle = reye_yaw_pitch_roll[i:i+1]
    #        angle = np.concatenate([leye_angle, reye_angle], axis=0)
    #        target_angle = np.concatenate([leye_target_angle, reye_target_angle], axis=0)
    #        for new_angle in _gen_from_interps(angle, num_interps, target_angle):
    #            leye_yaw_pitch_roll[i:i+1] = new_angle[0]
    #            codes['eye_pose'][0, :3] = Rotation.from_euler('yxz', leye_yaw_pitch_roll, degrees=True).as_rotvec()
    #            reye_yaw_pitch_roll[i:i+1] = new_angle[1]
    #            codes['eye_pose'][0, 3:6] = Rotation.from_euler('yxz', reye_yaw_pitch_roll, degrees=True).as_rotvec()
    #            yield codes

    

def main(args):
    model_cfg = getattr(cfg, args.model_version.upper())
    test_cfg = getattr(cfg, 'TEST')

    # update config
    if len(args.config) > 0:
        cfg.freeze(False)
        cfg.update_args(args.config.split(';'))
        model_cfg.MODEL_VERSION = args.model_version.lower()
        cfg.freeze()
    print("Config: ------------------------------------------\n" + str(cfg) + '\n')

    # fix random seed
    torch.manual_seed(test_cfg.SEED)
    random.seed(test_cfg.SEED)
    np.random.seed(test_cfg.SEED)

    # ============== setup model
    if args.gpus is None:
        args.gpus = [0]
    #os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in args.gpus])
    tester = Tester(test_cfg, model_cfg, args.gpus, args.init_ckpt_file)
    device0 = 'cuda:{}'.format(args.gpus[0])

    import sys
    old_cwd = os.getcwd()
    os.chdir('./Fit-DECA')
    new_cwd = os.getcwd()
    sys.path.remove(old_cwd)
    sys.path.append(new_cwd)    # need absolute path here
    from fit_image import DECAFitting
    deca = DECAFitting(test_cfg.IMG_SIZE, device0)
    os.chdir(old_cwd)
    sys.path.remove(new_cwd)
    sys.path.append(old_cwd)    # need absolute path here

    ori_img = cv2.imread(args.src_img)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    imgs, codes, preprocessings = deca.preprocess_and_encode(ori_img[np.newaxis])
    outputs = deca.decode_and_post_align(codes, preprocessings)
    outputs0 = copy.deepcopy(outputs)

    border_size = 2     # border size for combining images
    if args.interactive:
        codes_updater = update_codes_from_cmd
    else:
        codes_updater = update_codes_from_preset

        # create video writer
        fps = 25
        img_size = imgs[0].shape[:2][::-1]
        video_writer = cv2.VideoWriter(
            f'animation_{os.path.basename(args.src_img)}.mp4', 
            #cv2.VideoWriter_fourcc(*'MJPG'),   # avi
            cv2.VideoWriter_fourcc(*'MP4V'), 
            fps, 
            (img_size[0] * 2 + border_size, img_size[1] * 2 + border_size),
        )

    shade_min, shade_max = None, None   # used to normalize shaded face image

    for new_codes in codes_updater(codes):
        outputs = deca.decode_and_post_align(new_codes, preprocessings)

        src_img = torch.as_tensor(imgs).to(device0).permute([0, 3, 1, 2]).to(torch.float32) / 255
        src_verts = torch.as_tensor(outputs0['trans_verts']).to(device0)
        src_verts[..., 0] = src_verts[..., 0] / test_cfg.IMG_SIZE[0] * 2 - 1
        src_verts[..., 1] = src_verts[..., 1] / test_cfg.IMG_SIZE[1] * 2 - 1
        dst_verts = torch.as_tensor(outputs['trans_verts']).to(device0).unsqueeze(1)   # add a dimension for num_views
        dst_verts[..., 0] = dst_verts[..., 0] / test_cfg.IMG_SIZE[0] * 2 - 1
        dst_verts[..., 1] = dst_verts[..., 1] / test_cfg.IMG_SIZE[1] * 2 - 1

        with torch.no_grad():
            rec_dst_img = tester.run(src_img, src_verts, dst_verts)

        ## ============== debug
        ## get kpt color map
        #num_kpts = model_cfg.NUM_KEYPOINTS
        #cmap = plt.get_cmap('rainbow')
        #kpt_colors = np.array([cmap(i / num_kpts) for i in range(num_kpts)])
        #plt.figure() 
        #plt.imshow(data['img'][0])
        #plt.scatter(data['kpts'][0, :, 0], data['kpts'][0, :, 1], c=kpt_colors, s=1)     # first color is for background placeholder
        #plt.show()
        #continue

        rec_dst_img = np.cast[np.uint8](rec_dst_img.permute([0, 1, 3, 4, 2]).cpu().detach().numpy() * 255)
        rec_dst_img = cv2.cvtColor(rec_dst_img[0, 0], cv2.COLOR_RGB2BGR)
        src_img = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR)
        #src_shaded_face = cv2.normalize(outputs0['face_template'][0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        #dst_shaded_face = cv2.normalize(outputs['face_template'][0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        #src_shaded_face = cv2.normalize(outputs0['shaded_face'][0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        #dst_shaded_face = cv2.normalize(outputs['shaded_face'][0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        if shade_max is None or shade_min is None:
            shade_max = outputs0['shaded_face'][0].max(axis=(0, 1), keepdims=True)
            shade_min = outputs0['shaded_face'][0].min(axis=(0, 1), keepdims=True)
        src_shaded_face = cv2.cvtColor(cv2.cvtColor((np.clip((outputs0['shaded_face'][0] - shade_min) / (shade_max - shade_min), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        dst_shaded_face = cv2.cvtColor(cv2.cvtColor((np.clip((outputs['shaded_face'][0] - shade_min) / (shade_max - shade_min), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

        #src_lmks = src_img.copy()
        #dst_lmks = rec_dst_img.copy()
        #for pt in outputs0['trans_verts'][0]:
        #    cv2.circle(src_lmks, (int(pt[0]), int(pt[1])), radius=1, color=(0, 0, 255), thickness=1)
        #for pt in outputs0['landmarks2d'][0]:
        #    cv2.circle(src_lmks, (int(pt[0]), int(pt[1])), radius=1, color=(255, 0, 0), thickness=2)
        #for pt in outputs['trans_verts'][0]:
        #    cv2.circle(dst_lmks, (int(pt[0]), int(pt[1])), radius=1, color=(0, 0, 255), thickness=1)
        #for pt in outputs['landmarks2d'][0]:
        #    cv2.circle(dst_lmks, (int(pt[0]), int(pt[1])), radius=1, color=(255, 0, 0), thickness=2)
        #canvas = viz.stack_patches([src_img, src_lmks, src_shaded_face, rec_dst_img, dst_lmks, dst_shaded_face], 2, 3, border=2)

        canvas = viz.stack_patches([src_img, rec_dst_img, src_shaded_face, dst_shaded_face], 2, 2, border=border_size)

        if args.interactive:
            cv2.imwrite('face_edit_output_img.jpg', canvas)
        else:
            video_writer.write(canvas)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', default='model', help="Model version to use.")
    parser.add_argument('--gpus', nargs='+', type=int, help="GPUs to use")
    parser.add_argument('--config', default='', help="Config params. Each config param is separated by ';'.")
    parser.add_argument('--init_ckpt_file', default='', help="Init checkpoint file.")
    parser.add_argument('--src_img', default='', help="Source image")
    parser.add_argument('--interactive', type=int, required=True, help="Whether update the codes in cmd interactively or use presets to generate animation")
    parser.set_defaults(verbose=False)

    main(parser.parse_args())


