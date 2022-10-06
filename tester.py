import torch
from torch import nn
import torch.nn.functional as F

import os
import functools
from collections import OrderedDict

from common import ModelContainer

import pdb

class FullGenerator(nn.Module):
    def __init__(self, generator, test_cfg):
        super(FullGenerator, self).__init__()

        self.test_cfg = test_cfg

        self.generator = generator     # with MotionEstimator & FeatureDecoder inside already

    def forward(self, 
            src_img, 
            src_verts, dst_verts, 
        ):
        """
        src_img/dst_img: (b, 3, h, w)
        input image is must have been normalized to [0, 1] already.
        """
        num_view = dst_verts.shape[1]
        flatten_multiview_func = lambda x: torch.flatten(x.unsqueeze(1).expand(*([-1, num_view] + [-1] * (x.ndim - 1))), start_dim=0, end_dim=1)
        src_img = flatten_multiview_func(src_img)
        src_verts = flatten_multiview_func(src_verts)
        dst_verts = torch.flatten(dst_verts, 0, 1)

        # forward reconstruction
        mesh_flows_fwd, flows_fwd, warp_rgb_s, out_img_d, alpha_d, rec_dst_img = self.generator(src_img, src_verts, dst_verts)

        return rec_dst_img.view(-1, num_view, *(src_img.shape[1:]))

class Tester():
    def __init__(self, test_cfg, model_cfg, gpu_ids, init_ckpt_file=''):
        self.test_cfg = test_cfg
        self.model_cfg = model_cfg

        # create models
        self.container = ModelContainer(model_cfg, 'test')
        device0 = 'cuda:{}'.format(gpu_ids[0])
        
        if hasattr(self.container.generator, 'flow_rasterizer'):
            self.container.generator.flow_rasterizer.create_gl_context_pool([device0])
        if hasattr(self.container.generator, 'mesh_renderer'):
            self.container.generator.mesh_renderer.create_gl_context_pool([device0])
        self.full_g = FullGenerator(
            self.container.generator,
            self.test_cfg)
        self.full_g.to(device0)
        self.full_g.eval()

        # must initialize model before calling DataParallel()
        if len(init_ckpt_file) > 0:
            self.load_ckpt(init_ckpt_file, map_location=device0)

    def load_ckpt(self, ckpt_file, map_location):
        assert ckpt_file[-4:] == '.pth'
        assert os.path.isfile(ckpt_file)
        state_dict = torch.load(ckpt_file, map_location=map_location)

        ## remove spectral norm after loadking checkpoint
        ##self.full_g.apply(lambda m: nn.utils.remove_spectral_norm(m) if isinstance(m, nn.Conv2d) and m.weight.shape[0] > 10 and m.weight.shape[1] > 10 else m)
        #for m in self.full_g.modules():
        #    if isinstance(m, nn.Conv2d) and hasattr(m, 'weight_orig'):
        #        nn.utils.remove_spectral_norm(m)

        self.container.generator.load_state_dict(state_dict['generator'])

        print("Model successfully loaded from {}".format(ckpt_file))

    def run(self, 
            src_img, 
            src_verts, dst_verts, 
        ):
        """
        src_img/dst_img: (b, 3, h, w)
        input image must have been normalized to [0, 1] already.
        """
        return self.full_g(src_img, src_verts, dst_verts)
    