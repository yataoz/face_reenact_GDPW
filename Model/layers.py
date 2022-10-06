import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from functools import partial
import pdb

SN = True   # spectral norm
BS = True   # enforce bias regardless of normalization layer
NORM = 'BN'  # batch norm (BN) or instance norm (IN)

ACT_FN = lambda x: F.leaky_relu_(x, 0.1)

"""
BS=True will make head pose estimator converge much faster, which means a conv with bias before BN is still necessary.
BN=False will make keypoints clutered instead of scattered around.
"""



def _make_spectral_norm(layer_class):
    def wrapper(*args, **kwargs):
        layer = layer_class(*args, **kwargs)
        return nn.utils.spectral_norm(layer)
    return wrapper

class LinearBlock(nn.Module):
    def __init__(self, in_chns, out_chns, act_fn=ACT_FN, norm_fn=NORM, spectral_norm=SN):
        super(LinearBlock, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm1d
        InstNorm = nn.InstanceNorm1d

        if spectral_norm:
            Linear = _make_spectral_norm(nn.Linear)
        else:
            Linear = nn.Linear

        self.linear = Linear(in_chns, out_chns, bias=norm_fn is not None or not norm_fn != 'None' or BS)
        if norm_fn == 'BN':
            self.norm = BatchNorm(out_chns, affine=True)
        elif norm_fn == 'IN':
            self.norm = InstNorm(out_chns, affine=True)
        self.act_fn = act_fn

    def forward(self, x):
        out = self.linear(x)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        out = self.act_fn(out)
        return out

class UpBlock(nn.Module):
    def __init__(self, ndim, in_chns, out_chns, ksize=3, padding=1, act_fn=ACT_FN, norm_fn=NORM, spectral_norm=SN):
        super(UpBlock, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        if ndim == 2:
            Conv = nn.Conv2d 
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
            InstNorm = nn.InstanceNorm2d
        elif ndim == 3:
            Conv = nn.Conv3d 
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm3d
            InstNorm = nn.InstanceNorm3d
        else:
            raise NotImplementedError()

        if spectral_norm:
            Conv = _make_spectral_norm(Conv)

        self.conv = Conv(in_channels=in_chns, out_channels=out_chns, kernel_size=ksize, padding=padding, bias=norm_fn is not None or not norm_fn != 'None' or BS)
        if norm_fn == 'BN':
            self.norm = BatchNorm(out_chns, affine=True)
        elif norm_fn == 'IN':
            self.norm = InstNorm(out_chns, affine=True)

        self.act_fn = act_fn

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        out = self.act_fn(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, ndim, in_chns, out_chns, ksize=3, padding=1, act_fn=ACT_FN, norm_fn=NORM, spectral_norm=SN):
        super(DownBlock, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        if ndim == 2:
            Conv = nn.Conv2d 
            Pool = nn.AvgPool2d
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
            InstNorm = nn.InstanceNorm2d
        elif ndim == 3:
            Conv = nn.Conv3d 
            Pool = nn.AvgPool3d
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm3d
            InstNorm = nn.InstanceNorm3d
        else:
            raise NotImplementedError()

        if spectral_norm:
            Conv = _make_spectral_norm(Conv)

        self.conv = Conv(in_channels=in_chns, out_channels=out_chns, kernel_size=ksize, stride=1, padding=padding, bias=norm_fn is not None or not norm_fn != 'None' or BS)
        self.pool = Pool(kernel_size=2)
        if norm_fn == 'BN':
            self.norm = BatchNorm(out_chns, affine=True)
        elif norm_fn == 'IN':
            self.norm = InstNorm(out_chns, affine=True)

        self.act_fn = act_fn

    def forward(self, x):
        out = self.conv(x)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        out = self.act_fn(out)
        out = self.pool(out)
        return out


class SameBlock(nn.Module):
    def __init__(self, ndim, in_chns, out_chns, ksize=3, padding=1, act_fn=ACT_FN, norm_fn=NORM, spectral_norm=SN):
        super(SameBlock, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        if ndim == 2:
            Conv = nn.Conv2d 
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
            InstNorm = nn.InstanceNorm2d
        elif ndim == 3:
            Conv = nn.Conv3d 
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm3d
            InstNorm = nn.InstanceNorm3d
        else:
            raise NotImplementedError()

        if spectral_norm:
            Conv = _make_spectral_norm(Conv)

        self.conv = Conv(in_channels=in_chns, out_channels=out_chns, kernel_size=ksize, padding=padding, bias=norm_fn is not None or not norm_fn != 'None' or BS)
        if norm_fn == 'BN':
            self.norm = BatchNorm(out_chns, affine=True)
        elif norm_fn == 'IN':
            self.norm = InstNorm(out_chns, affine=True)

        self.act_fn = act_fn

    def forward(self, x):
        out = self.conv(x)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        out = self.act_fn(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, ndim, in_chns, out_chns, ksize=3, padding=1, act_fn=ACT_FN, downsample=False, norm_fn=NORM, spectral_norm=SN):
        super(ResBlock, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        if ndim == 2:
            Conv = nn.Conv2d 
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
            InstNorm = nn.InstanceNorm2d
        elif ndim == 3:
            Conv = nn.Conv3d 
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm3d
            InstNorm = nn.InstanceNorm3d
        else:
            raise NotImplementedError()

        if spectral_norm:
            Conv = _make_spectral_norm(Conv)

        stride = 2 if downsample else 1
        if stride != 1 or in_chns != out_chns:
            self.non_identity_shortcut = Conv(in_channels=in_chns, out_channels=out_chns, kernel_size=1, stride=stride, padding=0)
        else:
            self.non_identity_shortcut = None

        self.conv1 = Conv(in_channels=in_chns, out_channels=out_chns, kernel_size=ksize, stride=stride, padding=padding, bias=norm_fn is not None or not norm_fn != 'None' or BS)
        self.conv2 = Conv(in_channels=out_chns, out_channels=out_chns, kernel_size=ksize, padding=padding, bias=norm_fn is not None or not norm_fn != 'None' or BS)
        if norm_fn == 'BN':
            self.norm1 = BatchNorm(out_chns, affine=True)
            self.norm2 = BatchNorm(out_chns, affine=True)
        elif norm_fn == 'IN':
            self.norm1 = InstNorm(out_chns, affine=True)
            self.norm2 = InstNorm(out_chns, affine=True)

        self.act_fn = act_fn

    def forward(self, x):
        out = x
        if hasattr(self, 'norm1'):
            out = self.norm1(out)
        out = self.act_fn(out)
        out = self.conv1(out)
        if hasattr(self, 'norm2'):
            out = self.norm2(out)
        out = self.act_fn(out)
        out = self.conv2(out)

        if self.non_identity_shortcut:
            x = self.non_identity_shortcut(x)
        out += x

        return out


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, ndim, in_chns, out_chns, ksize=3, padding=1, act_fn=ACT_FN, downsample=False, norm_fn=NORM, spectral_norm=SN):
        super(ResBottleneck, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        if ndim == 2:
            Conv = nn.Conv2d 
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
            InstNorm = nn.InstanceNorm2d
        elif ndim == 3:
            Conv = nn.Conv3d 
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm3d
            InstNorm = nn.InstanceNorm3d
        else:
            raise NotImplementedError()

        if spectral_norm:
            Conv = _make_spectral_norm(Conv)

        stride = 2 if downsample else 1
        if stride != 1 or in_chns != out_chns:
            self.non_identity_shortcut = Conv(in_channels=in_chns, out_channels=out_chns, kernel_size=1, stride=stride, padding=0)
        else:
            self.non_identity_shortcut = None


        width = out_chns // self.expansion
        self.conv1 = Conv(in_channels=in_chns, out_channels=width, kernel_size=1, padding=0, bias=norm_fn is not None or not norm_fn != 'None' or BS)
        self.conv2 = Conv(in_channels=width, out_channels=width, kernel_size=ksize, stride=stride, padding=padding, bias=norm_fn is not None or not norm_fn != 'None' or BS)
        self.conv3 = Conv(in_channels=width, out_channels=out_chns, kernel_size=1, padding=0, bias=norm_fn is not None or not norm_fn != 'None' or BS)
        if norm_fn == 'BN':
            self.norm1 = BatchNorm(width, affine=True)
            self.norm2 = BatchNorm(width, affine=True)
            self.norm3 = BatchNorm(out_chns, affine=True)
        elif norm_fn == 'IN':
            self.norm1 = InstNorm(width, affine=True)
            self.norm2 = InstNorm(width, affine=True)
            self.norm3 = InstNorm(out_chns, affine=True)

        self.act_fn = act_fn
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        if hasattr(self, 'norm1'):
            out = self.norm1(out)
        out = self.act_fn(out)
        out = self.conv2(out)
        if hasattr(self, 'norm2'):
            out = self.norm2(out)
        out = self.act_fn(out)
        out = self.conv3(out)
        if hasattr(self, 'norm3'):
            out = self.norm3(out)

        if self.non_identity_shortcut:
            x = self.non_identity_shortcut(x)
        out += x

        return out

class Spade(nn.Module):
    def __init__(self, ndim, inp_chns, mod_chns, hid_chns, ksize=3, padding=1, act_fn=ACT_FN, norm_fn=NORM, spectral_norm=SN):
        """
        inp_chns: #channels of regular input to be normalized
        mod_chns: #channels of modulation input
        hid_chns: #channels of hidden layers for modulation input
        """
        super(Spade, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        if ndim == 2:
            Conv = nn.Conv2d 
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
            InstNorm = nn.InstanceNorm2d
        elif ndim == 3:
            Conv = nn.Conv3d 
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm3d
            InstNorm = nn.InstanceNorm3d
        else:
            raise NotImplementedError()

        if spectral_norm:
            Conv = _make_spectral_norm(Conv)
        
        if norm_fn == 'BN':
            self.norm = BatchNorm(inp_chns, affine=False)
        elif norm_fn == 'IN':
            self.norm = InstNorm(inp_chns, affine=False)

        self.shared_conv = nn.Sequential(
            Conv(mod_chns, hid_chns, kernel_size=ksize, padding=padding),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.gamma_conv = Conv(hid_chns, inp_chns, kernel_size=ksize, padding=padding)
        self.beta_conv = Conv(hid_chns, inp_chns, kernel_size=ksize, padding=padding)
    
    def forward(self, x, modulation):
        normed_x = self.norm(x)
        hid = self.shared_conv(modulation)
        gamma = self.gamma_conv(hid)
        beta = self.beta_conv(hid)
        return normed_x * (1 + gamma) + beta

class SpadeResBlock(nn.Module):
    def __init__(self, ndim, inp_chns, out_chns, mod_chns, hid_chns, ksize=3, padding=1, act_fn=ACT_FN, downsample=False, norm_fn=NORM, spectral_norm=SN):
        """
        inp_chns: #channels of regular input to be normalized
        mod_chns: #channels of modulation input
        hid_chns: #channels of hidden layers for modulation input
        """
        super(SpadeResBlock, self).__init__()

        if ndim == 2:
            Conv = nn.Conv2d 
        elif ndim == 3:
            Conv = nn.Conv3d 
        else:
            raise NotImplementedError()

        if spectral_norm:
            Conv = _make_spectral_norm(Conv)

        stride = 2 if downsample else 1
        if stride != 1 or inp_chns != out_chns:
            self.non_identity_shortcut = Conv(in_channels=inp_chns, out_channels=out_chns, kernel_size=1, stride=stride, padding=0)
        else:
            self.non_identity_shortcut = None

        self.spade1 = Spade(ndim, inp_chns, mod_chns, hid_chns, ksize, padding, act_fn, norm_fn, spectral_norm)
        self.conv1 = Conv(inp_chns, out_chns, kernel_size=ksize, padding=padding)
        self.spade2 = Spade(ndim, out_chns, mod_chns, hid_chns, ksize, padding, act_fn, norm_fn, spectral_norm)
        self.conv2 = Conv(out_chns, out_chns, kernel_size=ksize, padding=padding)
        self.act_fn = act_fn

    def forward(self, x, modulation):
        out = x
        out = self.spade1(out, modulation)
        out = self.act_fn(out)
        out = self.conv1(out)
        out = self.spade2(out, modulation)
        out = self.act_fn(out)
        out = self.conv2(out)
        if self.non_identity_shortcut:
            x = self.non_identity_shortcut(x)
        out += x
        return out

class AdaConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(AdaptiveConv2d, self).__init__()
        # Set options
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        if isinstance(self.kernel_size, int):
            self.kernel_size = tuple(self.kernel_size, self.kernel_size)

    def forward(self, inputs, weight, bias=None):
        """
        inputs: (b, cin, h, w)
        weight: (b, cout, cin, k, k)
        bias: (b, cout)
        """
        assert inputs.shape[0] == weight.shape[0]
        assert inputs.shape[1] == self.in_channels
        assert tuple(weight.shape[1:]) == (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        if self.bias:
            assert bias is not None
            assert inputs.shape[0] == bias.shape[0]
            assert bias.shape[1] == self.out_channels

        # Apply convolution
        if self.kernel_size[0] * self.kernel_size[1] > 1:
            outputs = []
            for i in range(inputs.shape[0]):
                outputs.append(F.conv2d(inputs[i:i+1], weight[i], bias[i], 
                                        self.stride, self.padding, self.dilation, self.groups))
            outputs = torch.cat(outputs, 0)

        else:
            assert self.groups == 1, "groups must be 1 if we use matrix multiplication to compute conv"
            b, c, h, w = inputs.shape
            weight = weight[:, :, :, 0, 0].transpose(1, 2)
            outputs = torch.bmm(inputs.view(b, c, -1).transpose(1, 2), weight).transpose(1, 2).view(b, -1, h, w)
            outputs = outputs + bias[..., None, None]
        
        return outputs

class AdaSpade(nn.Module):
    def __init__(self, ndim, inp_chns, mod_chns, hid_chns, ksize=3, padding=1, act_fn=ACT_FN, norm_fn=NORM):
        """
        inp_chns: #channels of regular input to be normalized
        mod_chns: #channels of modulation input
        hid_chns: #channels of hidden layers for modulation input

        Doesn't support spectral norm
        """
        super(Spade, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        if ndim == 2:
            Conv = AdaConv2d
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
            InstNorm = nn.InstanceNorm2d
        elif ndim == 3:
            Conv = AdaConv3d    # not implemented yet
            BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm3d
            InstNorm = nn.InstanceNorm3d
        else:
            raise NotImplementedError()

        if norm_fn == 'BN':
            self.norm = BatchNorm(inp_chns, affine=False)
        elif norm_fn == 'IN':
            self.norm = InstNorm(inp_chns, affine=False)

        self.shared_conv = nn.Sequential(
            Conv(mod_chns, hid_chns, kernel_size=ksize, padding=padding),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.gamma_conv = Conv(hid_chns, inp_chns, kernel_size=ksize, padding=padding)
        self.beta_conv = Conv(hid_chns, inp_chns, kernel_size=ksize, padding=padding)
    
    def forward(self, x, modulation, weights, biases):
        """
        weights/biases: list of weights/biases
        """
        normed_x = self.norm(x)
        hid = self.shared_conv(modulation, weights[0], biases[0])
        gamma = self.gamma_conv(hid, weights[1], biases[1])
        beta = self.beta_conv(hid, weights[2], biases[2])
        return normed_x * (1 + gamma) + beta

class AdaSpadeResBlock(nn.Module):
    def __init__(self, ndim, inp_chns, out_chns, mod_chns, hid_chns, ksize=3, padding=1, act_fn=ACT_FN, downsample=False, norm_fn=NORM):
        """
        inp_chns: #channels of regular input to be normalized
        mod_chns: #channels of modulation input
        hid_chns: #channels of hidden layers for modulation input

        Doesn't support spectral norm
        """
        super(SpadeResBlock, self).__init__()

        if ndim == 2:
            Conv = AdaConv2d
        elif ndim == 3:
            Conv = AdaConv3d
        else:
            raise NotImplementedError()

        stride = 2 if downsample else 1
        if stride != 1 or inp_chns != out_chns:
            self.non_identity_shortcut = Conv(in_channels=inp_chns, out_channels=out_chns, kernel_size=1, stride=stride, padding=0)
        else:
            self.non_identity_shortcut = None

        self.spade1 = Spade(ndim, inp_chns, mod_chns, hid_chns, ksize, padding, act_fn, norm_fn)
        self.conv1 = Conv(inp_chns, out_chns, kernel_size=ksize, padding=padding)
        self.spade2 = Spade(ndim, out_chns, mod_chns, hid_chns, ksize, padding, act_fn, norm_fn)
        self.conv2 = Conv(out_chns, out_chns, kernel_size=ksize, padding=padding)
        self.act_fn = act_fn

    def forward(self, x, modulation, weights, biases):
        """
        weights/biases: list of list of weights/biases
        """
        out = x
        out = self.spade1(out, modulation, weights[0], biases[0])
        out = self.act_fn(out)
        out = self.conv1(out, weights[1], biases[1])
        out = self.spade2(out, modulation, weights[2], biases[2])
        out = self.act_fn(out)
        out = self.conv2(out, weights[3], biases[3])
        if self.non_identity_shortcut:
            x = self.non_identity_shortcut(x, weights[4], biases[4])
        out += x
        return out

######################################## 2D modules

class UpBlock2D(UpBlock):
    def __init__(self, *args, **kwargs):
        super(UpBlock2D, self).__init__(2, *args, **kwargs)

class DownBlock2D(DownBlock):
    def __init__(self, *args, **kwargs):
        super(DownBlock2D, self).__init__(2, *args, **kwargs)

class SameBlock2D(SameBlock):
    def __init__(self, *args, **kwargs):
        super(SameBlock2D, self).__init__(2, *args, **kwargs)

class ResBlock2D(ResBlock):
    def __init__(self, *args, **kwargs):
        super(ResBlock2D, self).__init__(2, *args, **kwargs)

class ResBottleneck2D(ResBottleneck):
    def __init__(self, *args, **kwargs):
        super(ResBottleneck2D, self).__init__(2, *args, **kwargs)

class Spade2D(Spade):
    def __init__(self, *args, **kwargs):
        super(Spade2D, self).__init__(2, *args, **kwargs)

class SpadeResBlock2D(SpadeResBlock):
    def __init__(self, *args, **kwargs):
        super(SpadeResBlock2D, self).__init__(2, *args, **kwargs)

class AdaSpade2D(AdaSpade):
    def __init__(self, *args, **kwargs):
        super(AdaSpade2D, self).__init__(2, *args, **kwargs)

class AdaSpadeResBlock2D(AdaSpadeResBlock):
    def __init__(self, *args, **kwargs):
        super(AdaSpadeResBlock2D, self).__init__(2, *args, **kwargs)

#########################################  3D modules

class UpBlock3D(UpBlock):
    def __init__(self, *args, **kwargs):
        super(UpBlock3D, self).__init__(3, *args, **kwargs)

class DownBlock3D(DownBlock):
    def __init__(self, *args, **kwargs):
        super(DownBlock3D, self).__init__(3, *args, **kwargs)

class SameBlock3D(SameBlock):
    def __init__(self, *args, **kwargs):
        super(SameBlock3D, self).__init__(3, *args, **kwargs)

class ResBlock3D(ResBlock):
    def __init__(self, *args, **kwargs):
        super(ResBlock3D, self).__init__(3, *args, **kwargs)

class ResBottleneck3D(ResBottleneck):
    def __init__(self, *args, **kwargs):
        super(ResBottleneck3D, self).__init__(3, *args, **kwargs)

class Spade3D(Spade):
    def __init__(self, *args, **kwargs):
        super(Spade3D, self).__init__(3, *args, **kwargs)

class SpadeResBlock3D(SpadeResBlock):
    def __init__(self, *args, **kwargs):
        super(SpadeResBlock3D, self).__init__(3, *args, **kwargs)