import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19, vgg16

import pdb

class PerceptualLoss(nn.Module):
    """
    reference: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def forward(self, X, Y, feature_layers, style_layers, pred_is_list):
        """
        X, Y must be normalized to [0, 1] already.

        If pred_is_list=True, then X is a list of predictions; each entry is used to compute loss with Y and the computation on Y is shared
        """
        #feature_loss_weights = [1, 1, 1, 1, 10]
        #style_loss_weights = [1, 1, 1, 20, 500]

        feature_loss_weights = [1, 1, 1, 1, 1]
        style_loss_weights = [1, 1, 1, 1, 1]

        if len(feature_layers) > 0:
            assert max(feature_layers) < len(feature_loss_weights), "Feature layer index mustn't exceed total number of blocks"
        if len(style_layers) > 0:
            assert max(style_layers) < len(style_loss_weights), "Style layer index mustn't exceed total number of blocks"
        
        # run network on Y and cache layer outputs
        if not isinstance(Y, (tuple, list)):
            Y = [Y]

        num_pyramids = len(Y)
        cached_outputs_y = [None for _ in range(num_pyramids)]
        for p in range(num_pyramids):
            cached_outputs_y[p] = self.net(Y[p])

        # run network on each X and compute loss with Y
        X_list = X if pred_is_list else [X]
        feature_losses_list = []
        style_losses_list = []
        for X in X_list:
            if not isinstance(X, (tuple, list)):
                X = [X]

            feature_losses = [0] * len(feature_layers)
            style_losses = [0] * len(style_layers)
            for p in range(num_pyramids):
                layer_outputs_x = self.net(X[p])
                layer_outputs_y = cached_outputs_y[p]
                
                # perceptual loss
                for k, i in enumerate(feature_layers):
                    feature_losses[k] += F.l1_loss(layer_outputs_x[i], layer_outputs_y[i]) * feature_loss_weights[i]
                for k, i in enumerate(style_layers):
                    act_x = layer_outputs_x[i].reshape(layer_outputs_x[i].shape[0], layer_outputs_x[i].shape[1], -1)
                    act_y = layer_outputs_y[i].reshape(layer_outputs_y[i].shape[0], layer_outputs_y[i].shape[1], -1)
                    norm_factor = act_x.shape[1] * act_x.shape[2]
                    gram_x = act_x @ act_x.permute(0, 2, 1) / norm_factor   # averaged by H*W*C
                    gram_y = act_y @ act_y.permute(0, 2, 1) / norm_factor   # averaged by H*W*C
                    style_losses[k] += F.l1_loss(gram_x, gram_y) * style_loss_weights[i]

            feature_losses_list.append(feature_losses)
            style_losses_list.append(style_losses)
        
        if not pred_is_list:
            feature_losses_list = feature_losses_list[0]
            style_losses_list = style_losses_list[0]

        return feature_losses_list, style_losses_list 

class vgg_face_wrapper(nn.Module):
    """
    reference: https://www.robots.ox.ac.uk/~albanie/pytorch-models.html
    """
    def __init__(self, use_pool_output, num_blocks):
        super(vgg_face_wrapper, self).__init__()

        assert num_blocks > 0 and num_blocks <= 5
        self.num_blocks = num_blocks

        self.mean = torch.tensor([129.186279296875, 104.76238250732422, 93.59396362304688]).view(1, 3, 1, 1)
        self.mean = torch.nn.Parameter(data=self.mean, requires_grad=False)

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

        self.blocks = [None] * 5
        if use_pool_output:
            self.blocks[0] = nn.Sequential(
                self.conv1_1, 
                self.relu1_1, 
                self.conv1_2, 
                self.relu1_2, 
                self.pool1,
                )
            self.blocks[1] = nn.Sequential(
                self.conv2_1, 
                self.relu2_1, 
                self.conv2_2, 
                self.relu2_2, 
                self.pool2,
                )
            self.blocks[2] = nn.Sequential(
                self.conv3_1, 
                self.relu3_1, 
                self.conv3_2, 
                self.relu3_2, 
                self.conv3_3, 
                self.relu3_3, 
                self.pool3,
                )
            self.blocks[3] = nn.Sequential(
                self.conv4_1, 
                self.relu4_1, 
                self.conv4_2, 
                self.relu4_2, 
                self.conv4_3, 
                self.relu4_3, 
                self.pool4,
                )
            self.blocks[4] = nn.Sequential(
                self.conv5_1, 
                self.relu5_1, 
                self.conv5_2, 
                self.relu5_2, 
                self.conv5_3, 
                self.relu5_3, 
                self.pool5,
                )
        else:
            self.blocks[0] = nn.Sequential(
                self.conv1_1, 
                self.relu1_1, 
                self.conv1_2, 
                self.relu1_2, 
                )
            self.blocks[1] = nn.Sequential(
                self.pool1,
                self.conv2_1, 
                self.relu2_1, 
                self.conv2_2, 
                self.relu2_2, 
                )
            self.blocks[2] = nn.Sequential(
                self.pool2,
                self.conv3_1, 
                self.relu3_1, 
                self.conv3_2, 
                self.relu3_2, 
                self.conv3_3, 
                self.relu3_3, 
                )
            self.blocks[3] = nn.Sequential(
                self.pool3,
                self.conv4_1, 
                self.relu4_1, 
                self.conv4_2, 
                self.relu4_2, 
                self.conv4_3, 
                self.relu4_3, 
                )
            self.blocks[4] = nn.Sequential(
                self.pool4,
                self.conv5_1, 
                self.relu5_1, 
                self.conv5_2, 
                self.relu5_2, 
                self.conv5_3, 
                self.relu5_3, 
                )

    def forward(self, x):
        #mean = self.mean.to(x.device)
        x = 255 * x - self.mean
        outs = []
        for i, block in enumerate(self.blocks):
            if i >= self.num_blocks:
                break
            x = block(x)
            outs.append(x)
        return outs

def vgg_face_dag(use_pool_output, num_blocks, weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = vgg_face_wrapper(use_pool_output, num_blocks)
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

class VggFaceLoss(PerceptualLoss):
    def __init__(self, use_pool_output, num_blocks):
        super(VggFaceLoss, self).__init__()
        self.net = vgg_face_dag(use_pool_output, num_blocks, 'models/vgg_face_dag.pth')
        self.net.requires_grad_(False)
        self.net.eval()    # set to eval mode to remove dropout behavior

class vgg19_wrapper(nn.Module):
    def __init__(self, use_pool_output, num_blocks):
        super(vgg19_wrapper, self).__init__()

        assert num_blocks > 0 and num_blocks <= 5
        self.num_blocks = num_blocks

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.mean = torch.nn.Parameter(data=self.mean, requires_grad=False)
        self.std = torch.nn.Parameter(data=self.std, requires_grad=False)

        vgg = vgg19(pretrained=True)
        blocks = []
        if use_pool_output:
            blocks.append(vgg.features[:5])
            blocks.append(vgg.features[5:10])
            blocks.append(vgg.features[10:19])
            blocks.append(vgg.features[19:28])
            blocks.append(vgg.features[28:37])
        else:
            blocks.append(vgg.features[:4])
            blocks.append(vgg.features[4:9])
            blocks.append(vgg.features[9:18])
            blocks.append(vgg.features[18:27])
            blocks.append(vgg.features[27:36])
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x):
        #mean = self.mean.to(x.device)
        #std = self.std.to(x.device)
        x = (x - self.mean) / self.std
        outs = []
        for i, block in enumerate(self.blocks):
            if i >= self.num_blocks:
                break
            x = block(x)
            outs.append(x)
        return outs

class vgg16_wrapper(nn.Module):
    def __init__(self, use_pool_output, num_blocks):
        super(vgg16_wrapper, self).__init__()

        assert num_blocks > 0 and num_blocks <= 5
        self.num_blocks = num_blocks

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.mean = torch.nn.Parameter(data=self.mean, requires_grad=False)
        self.std = torch.nn.Parameter(data=self.std, requires_grad=False)

        vgg = vgg16(pretrained=True)
        blocks = []
        if use_pool_output:
            blocks.append(vgg.features[:5])
            blocks.append(vgg.features[5:10])
            blocks.append(vgg.features[10:17])
            blocks.append(vgg.features[17:24])
            blocks.append(vgg.features[23:31])
        else:
            blocks.append(vgg.features[:4])
            blocks.append(vgg.features[4:9])
            blocks.append(vgg.features[9:16])
            blocks.append(vgg.features[16:23])
            blocks.append(vgg.features[23:30])
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x):
        #mean = self.mean.to(x.device)
        #std = self.std.to(x.device)
        x = (x - self.mean) / self.std
        outs = []
        for i, block in enumerate(self.blocks):
            if i >= self.num_blocks:
                break
            x = block(x)
            outs.append(x)
        return outs

class vgg19_fomm_wrapper(nn.Module):
    def __init__(self, use_pool_output, num_blocks):
        super(vgg19_fomm_wrapper, self).__init__()

        assert num_blocks > 0 and num_blocks <= 5
        self.num_blocks = num_blocks

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.mean = torch.nn.Parameter(data=self.mean, requires_grad=False)
        self.std = torch.nn.Parameter(data=self.std, requires_grad=False)

        vgg = vgg19(pretrained=True)
        blocks = []
        if use_pool_output:
            raise NotImplementedError()
        else:
            blocks.append(vgg.features[:2])
            blocks.append(vgg.features[2:7])
            blocks.append(vgg.features[7:12])
            blocks.append(vgg.features[12:21])
            blocks.append(vgg.features[21:30])
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x):
        #mean = self.mean.to(x.device)
        #std = self.std.to(x.device)
        x = (x - self.mean) / self.std
        outs = []
        for i, block in enumerate(self.blocks):
            if i >= self.num_blocks:
                break
            x = block(x)
            outs.append(x)
        return outs

class VggLoss(PerceptualLoss):
    """
    reference: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    """
    def __init__(self, which_vgg, use_pool_output, num_blocks):
        super(VggLoss, self).__init__()
        self.net = eval('{}_wrapper'.format(which_vgg))(use_pool_output, num_blocks)
        self.net.requires_grad_(False)
        self.net.eval()    # set to eval mode to remove dropout behavior

if __name__ == '__main__':
    VggLoss('vgg16')