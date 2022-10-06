import os
import pprint
import socket
import platform
import copy
import pdb

class AttrDict():
    _freezed = False
    """ Avoid accidental creation of new hierarchies. """

    def __getattr__(self, name):
        if self._freezed:
            raise AttributeError(name)
        if name.startswith('_'):
            # Do not mess with internals. Otherwise copy/pickle will fail
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __setattr__(self, name, value):
        if self._freezed and name not in self.__dict__:
            raise AttributeError(
                "Config was freezed! Unknown config: {}".format(name))
        super().__setattr__(name, value)

    def __str__(self):
        return pprint.pformat(self.to_dict(), indent=1, width=100, compact=True)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__.items() if not k.startswith('_')}

    def update_args(self, args):
        """
        Update from command line args. 
        E.g., args = [TRAIN.BATCH_SIZE=1,TRAIN.INIT_LR=0.1]
        """
        assert isinstance(args, (tuple, list))
        for cfg in args:
            keys, v = cfg.split('=', maxsplit=1)
            keylist = keys.split('.')

            dic = self
            for i, k in enumerate(keylist[:-1]):
                assert k in dir(dic), "Unknown config key: {}".format(keys)
                dic = getattr(dic, k)
            key = keylist[-1]

            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                v = eval(v)
            setattr(dic, key, v)

    def freeze(self, freezed=True):
        self._freezed = freezed
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.freeze(freezed)

    # avoid silent bugs
    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()


config = AttrDict()
_C = config     # short alias to avoid coding

# training
_C.TRAIN.DATA_ROOT = 'data/sample_train'
_C.TRAIN.SEED = 100
_C.TRAIN.MAX_EPOCH = 1000
_C.TRAIN.LOSS_WEIGHTS = {
                    'scale_flow': 10,
                    'scale_occ': 1,
                    'pix': 10,
                    'vgg_feature': 10,
                    'vgg_style': 10,
                    'feat_match': 10,
                    'GAN_gen': 1,
                    'GAN_discrim': 1,
                    } 

_C.TRAIN.STEPS_PER_EPOCH = 10000
_C.TRAIN.SAVE_PER_K_EPOCHS = 1
_C.TRAIN.SUMMARY_PERIOD = 1000

_C.TRAIN.INIT_G_LR = 2.e-4
_C.TRAIN.INIT_D_LR = 2.e-4
_C.TRAIN.G_PERIOD = 1
_C.TRAIN.D_PERIOD = 1
_C.TRAIN.PARAM_UPDATE_PERIOD = 1000
_C.TRAIN.WEIGHT_DECAY = 0

_C.TRAIN.IMG_SIZE = (256, 256)
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.COLOR_JITTER = True

_C.TRAIN.PIX_LOSS = True     # pixelwise loss
_C.TRAIN.GAN_LOSS = 'LS'    # Hinge, LS, or None
_C.TRAIN.SCALE_FLOW_LOSS = False
_C.TRAIN.VGG_LOSS.NUM_PYRAMIDS = 4
_C.TRAIN.VGG_LOSS.FEATURE_LAYERS = [0, 1, 2, 3, 4]
_C.TRAIN.VGG_LOSS.STYLE_LAYERS = []

_C.MODEL.GUIDANCE = 'neural_codes'  # neural_codes, geom_disp or both
_C.MODEL.NUM_LEVELS = 6
_C.MODEL.OCCLUSION_AWARE = False
_C.MODEL.USE_ALPHA_BLEND = False
_C.MODEL.SHRINK_RATIO = 0.5
_C.MODEL.SPECTRAL_NORM = True
_C.MODEL.NORM_METHOD = 'BN'
_C.MODEL.ENFORCE_BIAS = True
_C.MODEL.D_LOGITS_LAYERS = [-1]
_C.MODEL.D_SCALES = ['x1']
_C.MODEL.D_DROPOUT = False

_C.TEST.SEED = 100
_C.TEST.IMG_SIZE = (256, 256)
_C.TEST.BATCH_SIZE = 1

_C.freeze()
