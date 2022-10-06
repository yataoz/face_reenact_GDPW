import torch
import tensorpack as tp
import numpy as np
import cv2 
import os
import functools
import itertools
from scipy.spatial.transform import Rotation
import random
from collections import defaultdict

from .utils import MultiProcessVoxCelebData
from .img_proc import ImgProcessor

import pdb

def create_dataflow(data_root, img_size):
    df = MultiProcessVoxCelebData(data_root)
    img_processor = ImgProcessor(img_size)
    df = tp.dataflow.MapData(df, img_processor)
    return df

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, data_root, img_size):
        self.df = create_dataflow(data_root, img_size)
    
    def __iter__(self):
        self.df.reset_state()
        yield from self.df

def debug_Dataset():
    data_root = 'data/sample_train'
    img_size = (256, 256)
    ds = Dataset(data_root, img_size)
    for x in ds:
        img = np.concatenate(x['img'], axis=0)
        #fg_mask = np.concatenate(x['fg_mask'], axis=0)

        w, h = img_size
        verts = np.concatenate(x['posed_verts'], axis=0)
        verts[:, 0] = (verts[:, 0] + 1) * w / 2
        verts[:, 1] = (verts[:, 1] + 1) * h / 2
        verts[verts.shape[0]//2:, 1] += h
        for pt in verts:
            #if pt[0] >= img_size[0] or pt[0] < 0 or pt[1] >= 2*img_size[1] or pt[1] < 0:
            #    pdb.set_trace()
            cv2.circle(img, (int(pt[0]), int(pt[1])), radius=1, color=(0, 0, 255), thickness=1)
        
        cv2.imwrite('tmp/img.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        pdb.set_trace()

if __name__ == '__main__':
    seed = 200
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    debug_Dataset()