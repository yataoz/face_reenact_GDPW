import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad

import pdb

def make_coordinate_grid_2d(spatial_shape, dtype, device, normalize):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_shape.

    The reason to use normalize=True is that F.grid_sample() requires input sampling gird to be in [-1, 1]
    """
    h, w = spatial_shape
    y = torch.arange(h, dtype=dtype, device=device)
    x = torch.arange(w, dtype=dtype, device=device)

    if normalize: 
        x = (2 * (x / (w - 1)) - 1)
        y = (2 * (y / (h - 1)) - 1)
       
    yy, xx = torch.meshgrid(y, x)
    meshed = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2)
    return meshed

class AntiAliasDownsample2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasDownsample2d, self).__init__()
        assert scale <= 1
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out

class ImagePyramid(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, num_pyramids, num_channels):
        super(ImagePyramid, self).__init__()
        downs = []
        for p in range(num_pyramids):
            s = 1 / (2 ** p)
            downs.append(AntiAliasDownsample2d(num_channels, s))
        self.downs = nn.ModuleList(downs)

    def forward(self, x):
        res = []
        for down in self.downs:
            res.append(down(x))
        return res

def backward_warp(feat, flow_d2s, occlusion_mask=None, resize_flow=False):
    """
    warp using backward flow

    feat: (b, h, w, c)
    flow_d2s: (b, 2, h', w')
    """
    try:
        assert flow_d2s.ndim == 4 and flow_d2s.shape[1] == 2
    except:
        pdb.set_trace()
    spatial_shape = feat.shape[2:]
    if resize_flow and flow_d2s.shape[2:] != spatial_shape:
        flow = F.interpolate(flow_d2s, spatial_shape, mode='area')
    else:
        flow = flow_d2s
    grid_d = make_coordinate_grid_2d(spatial_shape, feat.dtype, feat.device, normalize=True)  # xy of shape (h, w, 2)
    #debug_print(grid_d, 'grid_d', 92, 'stats')
    #debug_print(flow, 'flow', 94, 'stats')
    grid_s = grid_d.unsqueeze(0) + flow.permute(0, 2, 3, 1)
    #debug_print(grid_s, 'grid_s', 93, 'stats')
    warp_feat = F.grid_sample(feat, grid_s)
    
    if occlusion_mask is not None:
        if resize_flow and occlusion_mask.shape[2:] != spatial_shape:
            mask = F.interpolate(occlusion_mask, spatial_shape, mode='area')
        else:
            mask = occlusion_mask
        warp_feat *= mask

    return  warp_feat

