import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.transforms import ColorJitter
import torch.distributed as dist

import os
import functools

from Model.utils import ImagePyramid, backward_warp
from utils import SummaryCollector
from common import ModelContainer
from Losses.perceptual import VggLoss, VggFaceLoss

import pdb

class FullDiscriminator(nn.Module):
    def __init__(self, discriminator, train_cfg, summary_collector):
        super(FullDiscriminator, self).__init__()

        self.train_cfg = train_cfg
        self.discriminator = discriminator
        self.summary_collector = summary_collector

    def D_loss_fn(self, real_logit, fake_logit):
        if self.train_cfg.GAN_LOSS == 'Hinge':
            return F.relu(1 + fake_logit) + F.relu(1 - real_logit)      # convergence value ~2
        elif self.train_cfg.GAN_LOSS == 'LS':
            return torch.square(1 - real_logit) + torch.square(fake_logit)  # convergence value ~1
        else:
            raise NotImplementedError("Unrecognized GAN loss: {}".format(self.train_cfg.GAN_LOSS))

    def forward(self, img_pd, img_gt, cond, reso=None, name_scope='discrim'):
        loss_weights = self.train_cfg.LOSS_WEIGHTS

        if reso is None:
            # use multi-scale discriminator
            real_feats, real_logits = self.discriminator(img_gt, cond)
            fake_feats, fake_logits = self.discriminator(img_pd.detach(), cond)
        else:
            # use a discriminator at specific resolution
            real_feats, real_logits = self.discriminator[reso](img_gt, cond)
            fake_feats, fake_logits = self.discriminator[reso](img_pd.detach(), cond)

        # discriminator hinge loss 
        losses = []
        for real_logit, fake_logit in zip(real_logits, fake_logits):
            loss = torch.mean(self.D_loss_fn(real_logit, fake_logit)) * loss_weights['GAN_discrim']
            losses.append(loss)
            self.summary_collector.add('scalar', '{}/gan_loss_for_discrim'.format(name_scope), loss)
        gan_loss_for_discrim = sum(losses)
        return gan_loss_for_discrim

class FullGenerator(nn.Module):
    def __init__(self, generator, discriminator, train_cfg, summary_collector):
        super(FullGenerator, self).__init__()

        self.train_cfg = train_cfg
        self.summary_collector = summary_collector

        self.enable_d = self.train_cfg.GAN_LOSS not in ['None', None, False]

        self.generator = generator
        self.discriminator = discriminator
        if train_cfg.COLOR_JITTER:
            self.color_jit = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1) 

        self.vgg_loss = VggLoss('vgg19_fomm', False, 5)
        self.img_pyramid = ImagePyramid(self.train_cfg.VGG_LOSS.NUM_PYRAMIDS, 3)

    def G_loss_fn(self, fake_logit):
        if self.train_cfg.GAN_LOSS == 'Hinge':
            return -fake_logit      # convergence value ~0
        elif self.train_cfg.GAN_LOSS == 'LS':
            return torch.square(1 - fake_logit)     # convergence value ~0.5
        else:
            raise NotImplementedError("Unrecognized GAN loss: {}".format(self.train_cfg.GAN_LOSS))
    
    def compute_recon_loss(self, 
            img_pd_dict, img_gt,
            name_scope='losses',
        ):
        loss_weights = self.train_cfg.LOSS_WEIGHTS
        g_loss_list = []

        key_list = [k for k in img_pd_dict.keys()]
        img_pd_list = [img_pd_dict[k] for k in key_list]

        # pixelwise loss
        if self.train_cfg.PIX_LOSS:
            for k, v in img_pd_dict.items():
                pix_loss = F.l1_loss(v, img_gt) * loss_weights['pix']
                name = '{}/{}/pix_loss'.format(name_scope, k)
                self.summary_collector.add('scalar', name, pix_loss)
                g_loss_list.append(pix_loss)

        # vgg perceptual loss
        pyr_gt = [x.detach() for x in self.img_pyramid(img_gt)]
        pyr_pd_list = [self.img_pyramid(img_pd) for img_pd in img_pd_list]

        vgg_feature_losses_list, vgg_style_losses_list = self.vgg_loss(
            pyr_pd_list, pyr_gt, 
            self.train_cfg.VGG_LOSS.FEATURE_LAYERS, 
            self.train_cfg.VGG_LOSS.STYLE_LAYERS, 
            pred_is_list=True,
        )
        for k in range(len(img_pd_list)): 
            vgg_feature_losses = vgg_feature_losses_list[k]
            for i, loss in enumerate(vgg_feature_losses):
                loss *= loss_weights['vgg_feature']
                vgg_feature_losses[i] = loss
                name = '{}/{}/block_{}_vgg_feature_loss'.format(name_scope, key_list[k], i)
                self.summary_collector.add('scalar', name, loss)
            vgg_feature_loss = sum(vgg_feature_losses)
            g_loss_list.append(vgg_feature_loss)

            vgg_style_losses = vgg_style_losses_list[k]
            for i, loss in enumerate(vgg_style_losses):
                loss *= loss_weights['vgg_style']
                vgg_style_losses[i] = loss
                name = '{}/{}/block_{}_vgg_style_loss'.format(name_scope, key_list[k], i)
                self.summary_collector.add('scalar', name, loss)
            vgg_style_loss = sum(vgg_style_losses)
            g_loss_list.append(vgg_style_loss)

        ## VGG face perceptual loss
        #vgg_face_feature_losses_list, vgg_face_style_losses_list = self.vgg_face_loss(
        #    pyr_pd_list, pyr_gt, 
        #    self.train_cfg.VGG_FACE_LOSS.FEATURE_LAYERS, 
        #    self.train_cfg.VGG_FACE_LOSS.STYLE_LAYERS, 
        #    pred_is_list=True,
        #)
        #for k in range(len(img_pd_list)): 
        #    vgg_face_feature_losses = vgg_face_feature_losses_list[k]
        #    for i, loss in enumerate(vgg_face_feature_losses):
        #        loss *= loss_weights['vgg_face_feature']
        #        vgg_face_feature_losses[i] = loss
        #        name = '{}/{}/block_{}_vgg_face_feature_loss'.format(name_scope, key_list[k], i)
        #        self.summary_collector.add('scalar', name, loss)
        #    vgg_face_feature_loss = sum(vgg_face_feature_losses)
        #    g_loss_list.append(vgg_face_feature_loss)

        #    vgg_face_style_losses = vgg_face_style_losses_list[k]
        #    for i, loss in enumerate(vgg_face_style_losses):
        #        loss *= loss_weights['vgg_face_style']
        #        vgg_face_style_losses[i] = loss
        #        name = '{}/{}/block_{}_vgg_face_style_loss'.format(name_scope, key_list[k], i)
        #        self.summary_collector.add('scalar', name, loss)
        #    vgg_face_style_loss = sum(vgg_face_style_losses)
        #    g_loss_list.append(vgg_face_style_loss)

        # discriminator 
        if self.enable_d:
            cond = None
            real_feats, real_logits = self.discriminator(img_gt, cond)
            for k, img_pd in enumerate(img_pd_list):
                fake_feats, fake_logits = self.discriminator(img_pd, cond)

                # feature matching loss
                feat_match_loss = 0
                for real_feat, fake_feat in zip(real_feats, fake_feats):
                    loss = F.l1_loss(real_feat, fake_feat) * loss_weights['feat_match']
                    name = '{}/{}/feat_match_loss'.format(name_scope, key_list[k])
                    self.summary_collector.add('scalar', name, loss)
                    feat_match_loss += loss
                g_loss_list.append(feat_match_loss)

                # GAN loss for generator
                losses = []
                for real_logit, fake_logit in zip(real_logits, fake_logits):
                    loss = torch.mean(self.G_loss_fn(fake_logit)) * loss_weights['GAN_gen']
                    losses.append(loss)
                    name = '{}/{}/gan_loss_for_gen'.format(name_scope, key_list[k])
                    self.summary_collector.add('scalar', name, loss)
                    self.summary_collector.add('histogram', '{}/{}/fake_logit'.format(name_scope, key_list[k]), fake_logit)
                    self.summary_collector.add('histogram', '{}/{}/real_logit'.format(name_scope, key_list[k]), real_logit)
                gan_loss_for_gen = sum(losses)
                g_loss_list.append(gan_loss_for_gen)

        g_loss = sum(g_loss_list)

        return g_loss

    def compute_warp_loss(self, 
            flows, 
            src_img, dst_img,
            name_scope='losses',
        ):
        loss_weights = self.train_cfg.LOSS_WEIGHTS
        g_loss_list = []

        # ensure consistent flow output across resolutions
        if self.train_cfg.SCALE_FLOW_LOSS:
            max_reso = sorted([reso for reso in flows.keys()], key=lambda r: int(r.replace('x', '')))[0]
            for reso in flows.keys():
                if reso!= max_reso:
                    flow = flows[reso]
                    flow_loss = F.mse_loss(flow, F.interpolate(flows[max_reso], flow.shape[2:], mode='area')) * loss_weights['scale_flow']
                    name = '{}/{}/scale_flow_loss'.format(name_scope, reso)
                    self.summary_collector.add('scalar', name, flow_loss)
                    g_loss_list.append(flow_loss)

        # perceptual loss across resolutions for warped image
        wp_imgs = []
        ms_imgs = []    # multi scale gt images
        key_list = []
        for reso, flow in flows.items():
            stride = int(reso.replace('x', '')) 
            if stride >= 32:
                continue
            if src_img.shape[2:] != flow.shape[2:]:
                img_s = F.interpolate(src_img, flow.shape[2:], mode='area')
                img_d = F.interpolate(dst_img, flow.shape[2:], mode='area')
            else:
                img_s = src_img
                img_d = dst_img
            warp_img_s = backward_warp(img_s, flow)     # no occlusion mask applied here otherwise there'll be holes
            wp_imgs.append(warp_img_s)
            ms_imgs.append(img_d)
            key_list.append(reso)

        vgg_feature_losses, vgg_style_losses = self.vgg_loss(
            wp_imgs,
            ms_imgs,
            self.train_cfg.VGG_LOSS.FEATURE_LAYERS, 
            self.train_cfg.VGG_LOSS.STYLE_LAYERS, 
            pred_is_list=False,
        )
        for i, loss in enumerate(vgg_feature_losses):
            loss *= loss_weights['vgg_feature']
            vgg_feature_losses[i] = loss
            name = '{}/block_{}_vgg_feature_loss'.format(name_scope, i)
            self.summary_collector.add('scalar', name, loss)
        vgg_feature_loss = sum(vgg_feature_losses)
        g_loss_list.append(vgg_feature_loss)

        for i, loss in enumerate(vgg_style_losses):
            loss *= loss_weights['vgg_style']
            vgg_style_losses[i] = loss
            name = '{}/block_{}_vgg_style_loss'.format(name_scope, i)
            self.summary_collector.add('scalar', name, loss)
        vgg_style_loss = sum(vgg_style_losses)
        g_loss_list.append(vgg_style_loss)

        # discriminator loss across resolutions
        if self.enable_d:
            for reso, pd_img, ms_img in zip(key_list, wp_imgs, ms_imgs):
                if reso not in self.discriminator:
                    continue
                cond = None
                real_feats, real_logits = self.discriminator[reso](ms_img, cond)
                fake_feats, fake_logits = self.discriminator[reso](pd_img, cond)

                # feature matching loss
                feat_match_loss = 0
                for real_feat, fake_feat in zip(real_feats, fake_feats):
                    loss = F.l1_loss(real_feat, fake_feat) * loss_weights['feat_match']
                    name = '{}/{}/feat_match_loss'.format(name_scope, reso)
                    self.summary_collector.add('scalar', name, loss)
                    feat_match_loss += loss
                g_loss_list.append(feat_match_loss)

                # GAN loss for generator
                losses = []
                for real_logit, fake_logit in zip(real_logits, fake_logits):
                    loss = torch.mean(self.G_loss_fn(fake_logit)) * loss_weights['GAN_gen']
                    losses.append(loss)
                    name = '{}/{}/gan_loss_for_gen'.format(name_scope, reso)
                    self.summary_collector.add('scalar', name, loss)
                    self.summary_collector.add('histogram', '{}/{}/fake_logit'.format(name_scope, reso), fake_logit)
                    self.summary_collector.add('histogram', '{}/{}/real_logit'.format(name_scope, reso), real_logit)
                gan_loss_for_gen = sum(losses)
                g_loss_list.append(gan_loss_for_gen)
        
        
        g_loss = sum(g_loss_list)

        return g_loss, dict([(k, v) for k, v in zip(key_list, wp_imgs)]), dict([(k, v) for k, v in zip(key_list, ms_imgs)])
    
    def forward(self, 
            src_img, dst_img, 
            src_verts, dst_verts, 
        ):
        """
        src_img/dst_img: (b, 3, h, w)
        src_verts/dst_verts: (b, num_verts, 3)
        input image must have been normalized to [0, 1] already.
        """
        if hasattr(self, 'color_jit'):
            for b in range(src_img.shape[0]):
                img = torch.stack([src_img[b], dst_img[b]], dim=0)
                img = self.color_jit(img)
                src_img[b], dst_img[b] = img[0], img[1]
        #print("\n rank={}, batch={}".format(dist.get_rank(), src_img.shape[0]))

        # forward reconstruction
        mesh_flows_fwd, flows_fwd, warp_rgb_s, out_img_d, alpha_d, rec_dst_img = self.generator(src_img, src_verts, dst_verts)

        #import cv2
        #from utils import colorize_uv
        #flow = flow_fwd.detach().cpu().numpy()[0]
        #flow_color = colorize_uv(flow)
        #cv2.imwrite('img.jpg', flow_color)
        #pdb.set_trace()

        # ============= compute image loss 
        #fwd_recon_imgs = {'warp_rgb': warp_rgb_s, 'rec_rgb': rec_dst_img}
        fwd_recon_imgs = {'rec_dst': rec_dst_img}
        fwd_g_loss = self.compute_recon_loss(
            fwd_recon_imgs, dst_img,
            name_scope='fwd_gen/recon_loss',
        )
        fwd_wp_loss, fwd_wp_imgs, fwd_ms_imgs = self.compute_warp_loss(
            flows_fwd,
            src_img, dst_img,
            name_scope='fwd_gen/warp_loss',
        )
        fwd_g_loss += fwd_wp_loss

        g_loss = fwd_g_loss

        # ============== regularization
        if self.train_cfg.WEIGHT_DECAY >= 0:
            reg_loss = 0
            for name, param in self.generator.named_parameters():
                if 'weight' in name:
                    reg_loss = reg_loss + torch.sum(torch.square(param))
            reg_loss = reg_loss * self.train_cfg.WEIGHT_DECAY
            name = 'regularization/reg_loss'
            self.summary_collector.add('scalar', name, reg_loss)
            g_loss += reg_loss

        ## create real/fake image pairs for discriminator training step
        #fwd_real_imgs = dict()
        #for key in fwd_ms_imgs.keys():
        #    fwd_real_imgs['warp/' + key] = fwd_ms_imgs[key]
        #fwd_fake_imgs = dict()
        #for key in fwd_wp_imgs.keys():
        #    fwd_fake_imgs['warp/' + key] = fwd_wp_imgs[key]
        #for key in fwd_recon_imgs.keys():
        #    fwd_fake_imgs[key] = fwd_recon_imgs[key]
        #    fwd_real_imgs[key] = dst_img

        #bwd_real_imgs = dict()
        #for key in bwd_ms_imgs.keys():
        #    bwd_real_imgs['warp/' + key] = bwd_ms_imgs[key]
        #bwd_fake_imgs = dict()
        #for key in bwd_wp_imgs.keys():
        #    bwd_fake_imgs['warp/' + key] = bwd_wp_imgs[key]
        #for key in bwd_recon_imgs.keys():
        #    bwd_fake_imgs[key] = bwd_recon_imgs[key]
        #    bwd_real_imgs[key] = dst_img

        src_face_tem = self.generator.face_tem_s
        dst_face_tem = self.generator.face_tem_d

        name_scope = ''
        self.summary_collector.add('image', '{}/src_img'.format(name_scope), src_img)
        self.summary_collector.add('image', '{}/dst_img'.format(name_scope), dst_img)
        self.summary_collector.add('image', '{}/warp_rgb_s'.format(name_scope), warp_rgb_s)
        self.summary_collector.add('image', '{}/rec_dst_img'.format(name_scope), rec_dst_img)
        self.summary_collector.add('image', '{}/out_img_d'.format(name_scope), out_img_d)
        self.summary_collector.add('image', '{}/alpha_d'.format(name_scope), alpha_d)
        for reso in flows_fwd.keys():
            self.summary_collector.add('image', '{}/{}/mesh_flow_fwd'.format(name_scope, reso), mesh_flows_fwd[reso])
            self.summary_collector.add('image', '{}/{}/flow_fwd'.format(name_scope, reso), flows_fwd[reso])
        if self.generator.guidance in ['neural_codes', 'both']:
            self.summary_collector.add('image', '{}/src_face_tem'.format(name_scope), src_face_tem[:, :3])  # only visualize the first 3 channels if more than 3 channels
            self.summary_collector.add('image', '{}/dst_face_tem'.format(name_scope), dst_face_tem[:, :3])
            self.summary_collector.add('histogram', '{}/embeds'.format(name_scope), self.generator.embeds)

        
        return fwd_recon_imgs, fwd_wp_imgs, fwd_ms_imgs, g_loss

class Trainer():
    def __init__(self, train_cfg, model_cfg, rank, init_ckpt_file=''):
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.rank = rank
        self.distributed = dist.is_available() and dist.is_initialized()

        self.enable_d = self.train_cfg.GAN_LOSS not in ['None', None, False]

        self.g_period = train_cfg.G_PERIOD
        self.d_period = train_cfg.D_PERIOD

        self.global_step = 0

        # create models
        self.summary_collector = SummaryCollector()
        self.container = ModelContainer(model_cfg, 'train')
        device0 = 'cuda:{}'.format(rank)

        if hasattr(self.container.generator, 'flow_rasterizer'):
            self.container.generator.flow_rasterizer.create_gl_context_pool([device0])
        if hasattr(self.container.generator, 'mesh_renderer'):
            self.container.generator.mesh_renderer.create_gl_context_pool([device0])
        self.full_g = FullGenerator(
            self.container.generator, 
            self.container.discriminator, 
            train_cfg, self.summary_collector)
        self.full_g.to(device0)

        self.full_d = FullDiscriminator(
            self.container.discriminator, 
            train_cfg, self.summary_collector)
        self.full_d.to(device0)

        #if self.train_cfg.FREEZE_BN:
        #    self.full_g.apply(self._freeze_bn)
        #    self.full_d.apply(self._freeze_bn)

        # create optimizers
        self.g_optimizer = torch.optim.Adam(self.container.g_params, lr=train_cfg.INIT_G_LR, betas=(0.9, 0.999))
        self.d_optimizer = torch.optim.Adam(self.container.d_params, lr=train_cfg.INIT_D_LR, betas=(0.9, 0.999))

        # must initialize model before calling DataParallel()
        if len(init_ckpt_file) > 0:
            self.load_ckpt(init_ckpt_file, map_location=device0)

        if self.distributed:
            self.full_g = nn.parallel.DistributedDataParallel(self.full_g, device_ids=[rank], find_unused_parameters=True)
            self.full_d = nn.parallel.DistributedDataParallel(self.full_d, device_ids=[rank], find_unused_parameters=True)
    
    #def _freeze_bn(self, module):
    #    if isinstance(module, nn.modules.batchnorm._BatchNorm):
    #        module.eval()

    def load_ckpt(self, ckpt_file, map_location):
        assert ckpt_file[-3:] == 'pth'
        assert os.path.isfile(ckpt_file)
        state_dict = torch.load(ckpt_file, map_location=map_location)

        self.container.generator.load_state_dict(state_dict['generator'])

        self.container.discriminator.load_state_dict(state_dict['discriminator'])

        self.global_step = state_dict['global_step']
        #self.g_optimizer.load_state_dict(state_dict['g_optimizer'])
        #self.d_optimizer.load_state_dict(state_dict['d_optimizer'])

        print("Model successfully loaded from {}".format(ckpt_file))

    def save_ckpt(self, ckpt_file):
        assert ckpt_file[-3:] == 'pth'
        state_dict = self.container.state_dict() 

        state_dict['global_step'] = self.global_step
        #state_dict['g_optimizer'] = self.g_optimizer.state_dict() 
        #state_dict['d_optimizer'] = self.d_optimizer.state_dict() 

        torch.save(state_dict, ckpt_file)
        print("Model successfully saved to {}".format(ckpt_file))

    def step(self, 
            src_img, dst_img, 
            src_verts, dst_verts, 
        ):
        """
        src_img/dst_img: (b, 3, h, w)
        src_verts/dst_verts: (b, num_verts, 3)
        input image must have been normalized to [0, 1] already.
        """
        self.summary_collector.clear()
        update_g = self.global_step % self.g_period == 0
        update_d = self.global_step % self.d_period == 0

        if update_g:
            fwd_recon_imgs, fwd_wp_imgs, fwd_ms_imgs, g_loss = self.full_g(src_img, dst_img, src_verts, dst_verts)
            self.g_optimizer.zero_grad()
            g_loss.backward()    
            self.g_optimizer.step()
        if update_d and self.enable_d:
            cond = None
            d_loss = 0
            # D loss on reconstructed images. Losses can be on multiple resolutions using multi-scale discriminator
            for key in fwd_recon_imgs.keys():
                fwd_fake_img = fwd_recon_imgs[key]
                fake_img = fwd_fake_img
                real_img = dst_img
                d_loss = self.full_d(fake_img, real_img, cond, name_scope='discrim/recon/{}'.format(key))

            # D loss on warped images. For each warped image, loss can only be computed on the same resolution using corresponding discriminator
            for reso in fwd_wp_imgs.keys():
                disc = self.full_d.module.discriminator if self.distributed else self.full_d.discriminator
                if reso not in disc:
                    continue
                fwd_fake_img = fwd_wp_imgs[reso]
                fwd_real_img = fwd_ms_imgs[reso]
                fake_img = fwd_fake_img
                real_img = fwd_real_img

                d_loss += self.full_d(fake_img, real_img, cond, reso=reso, name_scope='discrim/warp/{}'.format(reso))

            self.d_optimizer.zero_grad()
            d_loss.backward()    
            self.d_optimizer.step()

        self.global_step += 1
        g_lr, d_lr = self.get_learning_rate()
        self.summary_collector.add('scalar', 'learning_rate/g_lr', g_lr)
        self.summary_collector.add('scalar', 'learning_rate/d_lr', d_lr)
    
    def set_learning_rate(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def get_learning_rate(self):
        for param_group in self.g_optimizer.param_groups:
            g_lr = param_group['lr']
        for param_group in self.d_optimizer.param_groups:
            d_lr = param_group['lr']
        return g_lr, d_lr