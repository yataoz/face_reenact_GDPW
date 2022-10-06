import torch
from torch import nn
import torch.nn.functional as F

import os
import cv2
import argparse
import numpy as np    
import copy
from collections import defaultdict
import random
import tqdm
import glob

from config import config as cfg
from tester import Tester

import pdb

class VideoFrameIterator():
    def __init__(self, src_video):
        self.src_video = src_video

        # src_video must be a video file, or a dir with many video files, or a dir with many images, or an integer representing camera device
        if os.path.isfile(src_video) and src_video[-4:] == '.mp4' or isinstance(src_video, int):
            cap = cv2.VideoCapture(src_video)
            assert cap.isOpened()
            self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
        elif os.path.isdir(src_video):
            img_files = [f for f in glob.glob(os.path.join(src_video, '*.jpg'))]
            self.num_frames = len(img_files)
            img0 = cv2.imread(img_files[0])
            self.frame_height, self.frame_width = img0.shape[:2]
        elif os.path.isfile(src_video) and src_video[-4:] in ['.jpg', '.png']:
            self.num_frames = 1
            img0 = cv2.imread(src_video)
            self.frame_height, self.frame_width = img0.shape[:2]
        else:
            raise ValueError("src_video is not a camera input, a video file or a folder with images")
    
    def __len__(self):
        return self.num_frames
    
    @property
    def frame_size(self):
        return self.frame_width, self.frame_height
    
    def __iter__(self):
        src_video = self.src_video
        # src_video must be a video file, or a dir with many video files, or a dir with many images, or an integer representing camera device
        if os.path.isfile(src_video) and src_video[-4:] == '.mp4' or isinstance(src_video, int):
            cap = cv2.VideoCapture(src_video)
            assert cap.isOpened()
            while 1:
                ret, frame = cap.read()
                if ret is None or ret is False or frame is None:
                    break
                yield frame
        elif os.path.isdir(src_video):
            img_files = [f for f in glob.glob(os.path.join(src_video, '*.jpg'))]
            vid_files = [f for f in glob.glob(os.path.join(src_video, '*.mp4'))]

            if len(img_files) > 0:
                assert len(vid_files) == 0
                for img_file in sorted(img_files):
                    img = cv2.imread(img_file)
                    yield img
            elif len(vid_files) > 0:
                assert len(img_files) == 0
                for vid_file in vid_files:
                    cap = cv2.VideoCapture(vid_file)
                    assert cap.isOpened()
                    while 1:
                        ret, frame = cap.read()
                        if ret is None or ret is False or frame is None:
                            break
                        yield frame
        elif os.path.isfile(src_video) and src_video[-4:] in ['.jpg', '.png']:
            img_file = src_video
            yield cv2.imread(img_file)
        else:
            raise ValueError("src_video is not a camera input, a video file or a folder with images")

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev
        assert t_e > 0

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

def reenact(src_img, driving_video, fit_model, tester, output_video, device0):
    if isinstance(src_img, np.ndarray):
        ori_img = src_img
    elif os.path.isfile(src_img):
        ori_img = cv2.imread(src_img)
    else:
        raise ValueError("src_img is invalid...")
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    src_imgs, src_codes, src_preprocessings = fit_model.preprocess_and_encode(ori_img[np.newaxis])
    src_outputs = fit_model.decode_and_post_align(src_codes, src_preprocessings)

    # create video writer
    fps = 25
    img_size = src_imgs[0].shape[:2][::-1]
    assert output_video[-3:] == 'mp4'
    video_writer = cv2.VideoWriter(
        output_video, 
        #cv2.VideoWriter_fourcc(*'MJPG'),   # avi
        cv2.VideoWriter_fourcc(*'MP4V'), 
        fps, 
        (img_size[0] * 3, img_size[1]),
    )

    src_img = torch.as_tensor(src_imgs).to(device0).permute([0, 3, 1, 2]).to(torch.float32) / 255
    src_verts = torch.as_tensor(src_outputs['trans_verts']).to(device0)
    src_verts[..., 0] = src_verts[..., 0] / img_size[0] * 2 - 1
    src_verts[..., 1] = src_verts[..., 1] / img_size[1] * 2 - 1

    src_img_to_save = (src_img[0].permute([1, 2, 0]).detach().cpu().numpy() * 255).astype(np.uint8)

    vid_iter = VideoFrameIterator(driving_video)
    filters = dict()
    for i, frame in enumerate(tqdm.tqdm(vid_iter)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dst_imgs, dst_codes, dst_preprocessings = fit_model.preprocess_and_encode(frame[np.newaxis])

        for k in ['exp', 'pose', 'cam']:
            if k not in filters.keys():
                filters[k] = OneEuroFilter(0, dst_codes[k], min_cutoff=0.5, beta=0)
            else:
                dst_codes[k] = filters[k](i, dst_codes[k])

        # keep the source shape
        dst_codes['shape'] = src_codes['shape']
        #dst_codes['light'] = src_codes['light']

        dst_outputs = fit_model.decode_and_post_align(dst_codes, dst_preprocessings)

        dst_img = torch.as_tensor(dst_imgs).to(device0).permute([0, 3, 1, 2]).to(torch.float32) / 255
        dst_verts = torch.as_tensor(dst_outputs['trans_verts']).to(device0)
        dst_verts[..., 0] = dst_verts[..., 0] / img_size[0] * 2 - 1
        dst_verts[..., 1] = dst_verts[..., 1] / img_size[1] * 2 - 1

        # add a dim for num_views dim
        dst_img = dst_img.unsqueeze(1)
        dst_verts = dst_verts.unsqueeze(1)

        with torch.no_grad():
            rec_dst_img = tester.run(src_img, src_verts, dst_verts)

        dst_img_to_save = (dst_img[0, 0].permute([1, 2, 0]).cpu().detach().numpy() * 255).astype(np.uint8)
        rec_dst_img_to_save = (rec_dst_img[0, 0].permute([1, 2, 0]).cpu().detach().numpy() * 255).astype(np.uint8)
        canvas = np.concatenate([src_img_to_save, rec_dst_img_to_save, dst_img_to_save], axis=1)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        video_writer.write(canvas)
        #cv2.imwrite('viz.jpg', canvas)
        #pdb.set_trace()
    
    video_writer.release()

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

    # ============== fit face model on src image
    import sys
    old_cwd = os.getcwd()
    os.chdir('./Fit-DECA')
    new_cwd = os.getcwd()
    sys.path.remove(old_cwd)
    sys.path.append(new_cwd)    # need absolute path here
    from fit_image import DECAFitting
    fit_model = DECAFitting(test_cfg.IMG_SIZE, device0)
    os.chdir(old_cwd)
    sys.path.remove(new_cwd)
    sys.path.append(old_cwd)    # need absolute path here

    reenact(args.src_img, args.driving_video, fit_model, tester, 'face_reenact_demo_video.mp4', device0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', default='model', help="Model version to use.")
    parser.add_argument('--gpus', nargs='+', type=int, help="GPUs to use")
    parser.add_argument('--config', default='', help="Config params. Each config param is separated by ';'.")
    parser.add_argument('--init_ckpt_file', default='', help="Init checkpoint file.")
    parser.add_argument('--src_img', required=True, help="Source image")
    parser.add_argument('--driving_video', required=True, help="Driving video path")
    parser.set_defaults(verbose=False)

    main(parser.parse_args())


