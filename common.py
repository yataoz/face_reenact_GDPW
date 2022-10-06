import torch 
import functools

import importlib
from Model.layers import NORM, SN, BS

class ModelContainer():
    def __init__(self, model_cfg, phase):
        assert phase in ['train', 'test']
        self.model_cfg = model_cfg
        self.phase = phase

        # check NORM & SN are consistent with preset
        assert self.model_cfg.SPECTRAL_NORM == SN
        assert self.model_cfg.NORM_METHOD == NORM
        assert self.model_cfg.ENFORCE_BIAS == BS

        # import corresponding model version
        module = importlib.import_module('Model.{}'.format(self.model_cfg.MODEL_VERSION))

        # create models
        self.generator = module.Generator(self.model_cfg)

        if self.phase == 'train':
            self.discriminator = module.MultiScaleDiscriminator(self.model_cfg.D_SCALES, self.model_cfg.D_LOGITS_LAYERS, self.model_cfg.D_DROPOUT, use_bg_mask=False)
        else:
            self.generator.eval()

    @property
    def g_params(self):
        models = [self.generator]
        g_params = functools.reduce(
            lambda x, y: x + y, 
            [list(m.parameters()) for m in models],
        )  
        return g_params
    
    @property
    def d_params(self):
        if self.phase == 'test':
            raise ValueError("d_params not available when phase='test'.")
        d_params = self.discriminator.parameters()
        return d_params

    def state_dict(self):
        res = dict()
        res['generator'] = self.generator.state_dict()
        if self.phase == 'train':
            res['discriminator'] = self.discriminator.state_dict()
        return res

