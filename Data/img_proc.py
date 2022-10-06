import numpy as np
import cv2 
import os

import pdb

class ImgProcessor():
    _rng = np.random.default_rng()

    _flip_chance = 0.5

    def __init__(self, img_size):
        self.img_size = img_size
        self.verts_flip_mapping = np.load('FLAME/flame_verts_flip_mapping.npy')
    
    def __call__(self, dp):
        img_list = []
        verts_list = []

        do_flip = self._rng.random() < self._flip_chance

        for i in range(2):
            img = dp['img{}'.format(i)]
            verts = dp['verts{}'.format(i)]

            assert img.shape[:2][::-1] == self.img_size

            if do_flip:
                img = cv2.flip(img, 1)      # code 1 means horizontal flip
                verts = verts[self.verts_flip_mapping]
                verts[:, 0] = self.img_size[0] - verts[:, 0]

            img_list.append(img)

            verts[:, :2] = 2 * verts[:, :2] / np.array(self.img_size, dtype=np.float32)[np.newaxis] - 1     # normalize verts
            verts_list.append(verts)

        res = {
            'img': np.array(img_list), 
            'posed_verts': np.array(verts_list),
        }


        return res

