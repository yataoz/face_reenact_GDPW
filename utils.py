from torch.utils.tensorboard import SummaryWriter
import torch
import os
import cv2 
from shutil import rmtree, copytree
import sys
import glob
import random
import numpy as np
from math import cos, sin
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorpack.utils.utils import get_tqdm_kwargs
from tensorpack.utils.viz import stack_patches
import tqdm

import pdb

matplotlib.use('Agg')   # for plt to work with ssh

def create_tqdm_bar(total, **kwargs):
    tqdm_args = get_tqdm_kwargs(leave=True, **kwargs)
    bar = tqdm.trange(total, **tqdm_args)
    return bar

def get_latest_checkpoint(log_dir):
    ckpt_files =[f for f in glob.glob(os.path.join(log_dir, '*.pth'))]
    ckpt_files =sorted(ckpt_files)
    return ckpt_files[-1]

def clean_folders(folders, force=False):
    for folder in folders:
        if os.path.exists(folder):
            if force:
                rmtree(folder)
            else:
                print("Are you sure you want to delete/overwrite the folder {}?".format(folder))
                choice = input("\033[93mPress y to delete/overwrite and n to cancel.\033[0m\n")
                if choice == 'y':
                    rmtree(folder)
                else:
                    sys.exit()

def set_rand_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

def colorize_uv(uv):
    hsv = np.zeros(uv.shape[:2]+(3,), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(uv[..., 0], uv[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

class SummaryCollector():
    def __init__(self):
        self.collections = defaultdict(list)

    def add(self, summary_type, name, value, append_if_exists=True):
        assert summary_type in ['scalar', 'image', 'histogram', 'tensor', 'meta']
        if name in self.collections.keys() and not append_if_exists:
            raise RuntimeError("An item with name={} already exists in the collection. You can't add a new item with the same name when 'append_if_exists' is disable.".format(name))
        self.collections[name].append({'summary_type': summary_type, 'value': value})

    def clear(self):
        self.collections.clear()

class BaseSummaryLogger():
    def __init__(self, log_dir):
        self.logger = SummaryWriter(log_dir)
    
    def write_graph(self, graph, inputs):
        self.logger.add_graph(graph, inputs)

    def scale_xy(self, normed_xy, wh):
        return np.stack([
            wh[0] / 2 * (normed_xy[..., 0] + 1), 
            wh[1] / 2 * (normed_xy[..., 1] + 1),
        ], axis=-1)

    def scale_flow(self, normed_flow, wh):
        return np.stack([
            wh[0] / 2 * normed_flow[..., 0], 
            wh[1] / 2 * normed_flow[..., 1],
        ], axis=-1)

    def draw_flow_on_ax(self, ax, mot_mask, xy_d, flow_d2s, color='r'):
        if mot_mask is None:
            mot_mask = np.zeros_like(xy_d[..., :1])
        ax.imshow(mot_mask.squeeze(-1), cmap='gray')
        flow_size = min(16, min(flow_d2s.shape[:2]))   # only show 16x16 flows
        sample_rate_x = xy_d.shape[1] // flow_size
        sample_rate_y = xy_d.shape[0] // flow_size
        mx = xy_d[::sample_rate_x, ::sample_rate_x, 0]
        my = xy_d[::sample_rate_y, ::sample_rate_y, 1]
        mu = flow_d2s[::sample_rate_x, ::sample_rate_x, 0]
        mv = flow_d2s[::sample_rate_y, ::sample_rate_y, 1]
        ax.quiver(mx, my, mu, mv, color=color, angles='xy', scale_units='xy', scale=1, linewidths=3) 
        ax.set_aspect('equal')

    def write_summaries(self, *args, **kwargs):
        raise NotImplementedError("To be implemented in sublcass.")

    def close(self):
        self.logger.close()

    def __del__(self):
        self.logger.close()

class SummaryLogger(BaseSummaryLogger):
    def write_summaries(self, collector, global_step, first_num_imgs=None):
        """
        collector: collector created in the forward pass of a model

        first_num_img: only show first several imgs
        """
        assert isinstance(collector, SummaryCollector)

        # simple logging for existing primitives such as scalar and histogram
        data = collector.collections

        for name, item_list in data.items():
            for i, item in enumerate(item_list):
                summary_type, value = item['summary_type'], item['value']
                unq_name = '{}_{}'.format(name, i)
                if summary_type == 'scalar':
                    self.logger.add_scalar(unq_name, value, global_step=global_step)
                elif summary_type == 'histogram':
                    self.logger.add_histogram(unq_name, value, global_step=global_step)
                elif summary_type == 'image':
                    item['value'] = value.permute(0, 2, 3, 1)   # channel first to channel last
                    #self.logger.add_image(unq_name, value)
                #elif summary_type == 'tensor':
                #    other_tensors[unq_name] = value

        bat_src_img = data['/src_img'][0]['value'].cpu().detach().numpy()
        bat_dst_img = data['/dst_img'][0]['value'].cpu().detach().numpy()
        bat_rec_dst_img = data['/rec_dst_img'][0]['value'].cpu().detach().numpy()
        bat_warp_rgb_s = data['/warp_rgb_s'][0]['value'].cpu().detach().numpy()
        bat_out_img_d = data['/out_img_d'][0]['value'].cpu().detach().numpy()
        bat_alpha_d = data['/alpha_d'][0]['value'].cpu().detach().numpy()

        bat_all_flows = dict()
        bat_all_occ_masks = dict()
        all_flow_resos = set()      # collect all available flow resolutions
        for reso in ['x1', 'x2', 'x4', 'x8', 'x16', 'x32', 'x64']:
            for drc in ['fwd']:
                for flow_type in ['mesh_flow', 'flow']:
                    name = '/{}/{}_{}'.format(reso, flow_type, drc)
                    if name not in data.keys():
                        continue
                    flow = data[name][0]['value'].cpu().detach().numpy()
                    bs, h, w, _ = flow.shape
                    bat_all_flows[name] = self.scale_flow(flow, (w, h))
                    all_flow_resos.add(reso)
            
                for occ_type in ['mesh_occ_mask', 'occ_mask']:
                    name = '/{}/{}_{}'.format(reso, occ_type, drc)
                    if name not in data.keys():
                        continue
                    bat_all_occ_masks[name] = data[name][0]['value'].cpu().detach().numpy()
        all_flow_resos = sorted(list(all_flow_resos), key=lambda r: int(r.replace('x', '')))

        show_face_tem = '/src_face_tem' in data.keys() and '/dst_face_tem' in data.keys()
        if show_face_tem:
            bat_src_face_tem = data['/src_face_tem'][0]['value'].cpu().detach().numpy()
            bat_dst_face_tem = data['/dst_face_tem'][0]['value'].cpu().detach().numpy()
        

        show_face_symmetry = '/symmetry_warp_rgb_s' in data.keys() and \
                            '/symmetry_mesh_flow_fwd' in data.keys()
        if show_face_symmetry:
            bat_symmetry_warp_rgb_s = data['/symmetry_warp_rgb_s'][0]['value'].cpu().detach().numpy()
            bat_symmetry_mesh_flow_fwd = data['/symmetry_mesh_flow_fwd'][0]['value'].cpu().detach().numpy()

            # scale motion
            bs, h, w, _ = bat_symmetry_mesh_flow_fwd.shape
            bat_symmetry_mesh_flow_fwd = self.scale_flow(bat_symmetry_mesh_flow_fwd, (w, h))

        h, w = bat_src_img.shape[1:3]
        my, mx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        mxy = np.stack([mx, my], axis=-1)

        rec_figs = []
        flw_figs = []
        symm_flw_figs = []
        show_bs = min(first_num_imgs, bs) if first_num_imgs else bs
        for i in range(show_bs):
            src_img = bat_src_img[i]
            dst_img = bat_dst_img[i]
            rec_dst_img = bat_rec_dst_img[i]
            warp_rgb_s = bat_warp_rgb_s[i]
            out_img_d = bat_out_img_d[i]
            alpha_d = bat_alpha_d[i]
            all_flows = dict([(k, v[i]) for k, v in bat_all_flows.items()])
            all_occ_masks = dict([(k, v[i]) for k, v in bat_all_occ_masks.items()])

            ax_rows = 2
            ax_cols = 5
            col_idx = dict()
            if show_face_symmetry:
                col_idx['face_symmetry'] = ax_cols
                ax_cols += 1
            if show_face_tem:
                col_idx['face_tem'] = ax_cols
                ax_cols += 1
            fig, axarr = plt.subplots(ax_rows, ax_cols)
            fig.set_size_inches(3*ax_cols, 3*ax_rows, forward=True)

            if show_face_symmetry:
                symmetry_warp_rgb_s = bat_symmetry_warp_rgb_s[i]
                axarr[1, col_idx['face_symmetry']].imshow(symmetry_warp_rgb_s)

            if show_face_tem:
                src_face_tem = bat_src_face_tem[i]
                src_face_tem = cv2.normalize(src_face_tem, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
                dst_face_tem = bat_dst_face_tem[i]
                dst_face_tem = cv2.normalize(dst_face_tem, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

                axarr[0, col_idx['face_tem']].imshow(src_face_tem)
                axarr[1, col_idx['face_tem']].imshow(dst_face_tem)
            
            # reconstruction visualization
            fig.suptitle('img pair {}'.format(i))
            h, w = src_img.shape[-2:]
            axarr[0, 0].imshow(src_img)
            axarr[1, 0].imshow(dst_img)
            axarr[1, 1].imshow(rec_dst_img)
            axarr[1, 2].imshow(alpha_d.squeeze(-1) if alpha_d.shape[-1] == 1 else alpha_d)
            axarr[1, 3].imshow(out_img_d)
            axarr[1, 4].imshow(warp_rgb_s)
            rec_figs.append(fig)

            # motion flow
            ncols = 2 * len(all_flow_resos)
            nrows = 2
            fig, axarr = plt.subplots(nrows, ncols)
            fig.suptitle('img pair {}'.format(i))
            fig.set_size_inches(3 * ncols, 3 * nrows, forward=True)
            for j1, drc in enumerate(['fwd']):
                for j2, flow_type in enumerate(['mesh_flow', 'flow']):
                    for j3, reso in enumerate(all_flow_resos):
                        flow_name = '/{}/{}_{}'.format(reso, flow_type, drc)
                        if flow_name not in all_flows.keys():
                            continue
                        occ_name = flow_name.replace('flow', 'occ_mask')
                        occ_mask = all_occ_masks.get(occ_name, None)
                        r = j1 + j2
                        c = j3
                        flow = all_flows[flow_name]
                        h, w = flow.shape[:2]
                        my, mx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                        mxy = np.stack([mx, my], axis=-1)
                        #flow = cv2.resize(flow, tuple(mxy.shape[:2][::-1]), interpolation=cv2.INTER_NEAREST)    # upsampled to full resolution for visualization
                        #if flow_type == 'mesh_flow' and reso == 'x1':
                        #    print("mesh_flow, min, max", flow.min(), flow.max())
                        #    cv2.imwrite('img.jpg', colorize_uv(flow))
                        #    pdb.set_trace()
                        axarr[r, 2*c].imshow(colorize_uv(flow))
                        axarr[r, 2*c].set_title(flow_name)
                        self.draw_flow_on_ax(axarr[r, 2*c+1], occ_mask, mxy, flow)
                        axarr[r, 2*c+1].set_title(flow_name)

            flw_figs.append(fig)

            # symmetry flow
            if show_face_symmetry:
                symmetry_mesh_flow_fwd = bat_symmetry_mesh_flow_fwd[i]

                fig, axarr = plt.subplots(2, 4)
                fig.suptitle('img pair {}'.format(i))
                fig.set_size_inches(3 * 4, 3 * 2, forward=True)

                axarr[0, 0].imshow(colorize_uv(symmetry_mesh_flow_fwd))
                self.draw_flow_on_ax(axarr[0, 1], None, mxy, symmetry_mesh_flow_fwd)
                axarr[0, 2].hist(symmetry_mesh_flow_fwd.reshape([-1, 2]), bins=20, density=True, label=['flow x', 'flow y'])
                axarr[0, 2].legend()

                symm_flw_figs.append(fig)

        if torch.__version__ < '1.2.9':
            # This is a workaround for torch 1.2.x 
            # when using a list of images as input to add_images() or add_figure(), tensorboard shows complementary colors of the images, which is wrong.
            # but add_image() or add_figure() with a single image input works fine
            rec_rgbs = [fig2rgb_array(x) for x in rec_figs]
            flw_rgbs = [fig2rgb_array(x) for x in flw_figs]
            self.logger.add_image('rec', stack_patches(rec_rgbs, 1, len(rec_rgbs)), dataformats='HWC')
            self.logger.add_image('flw', stack_patches(flw_rgbs, 1, len(flw_rgbs)), dataformats='HWC')
            if show_face_symmetry:
                symm_flw_rgbs = [fig2rgb_array(x) for x in symm_flw_figs]
                self.logger.add_image('symm_flw', stack_patches(symm_flw_rgbs, 1, len(symm_flw_rgbs)), dataformats='HWC')
        else:
            self.logger.add_figure('rec', rec_figs, global_step=global_step)
            self.logger.add_figure('flw', flw_figs, global_step=global_step)
            if show_face_symmetry:
                self.logger.add_figure('symm_flw', symm_flw_figs, global_step=global_step)

        plt.close('all')
        self.logger.flush()