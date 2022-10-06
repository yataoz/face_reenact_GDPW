import torch
import torch.distributed as dist

import numpy as np
import lz4.frame
import cv2
import math
import os
import struct
from collections import defaultdict
import random
import glob
import json

from tensorpack.dataflow.base import DataFlow

import pdb

CODE2DTYPE = dict([(np.dtype(d).num, np.dtype(d)) for d in ['float16', 'uint8', 'bool', 'int16', 'float32']])

def deserialize_ndarray(buf):
    buf = lz4.frame.decompress(buf)
    dtype_code, ndim = struct.unpack('ii', buf[:8])
    dtype = CODE2DTYPE[dtype_code]
    unpack_format = 'i'*ndim
    shp = struct.unpack(unpack_format, buf[8: 8+4*ndim])
    arr = np.frombuffer(buf, dtype=dtype, offset=8+4*ndim)
    return arr.reshape(shp)

def read_data(img_path, return_lmks_codes=False):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # BGR to RGB

    vert_path = img_path.replace('_img.jpg', '_posed_vertx.lz4')
    with open(vert_path, 'rb') as f:
        vert_data = f.read()
        verts = deserialize_ndarray(vert_data).astype(np.float32)
    
    data = dict()
    data['img'] = img
    data['verts'] = verts

    return data

def distributed_partitioner(dataset):
    if dist.is_available() and dist.is_initialized():
        replica_id = dist.get_rank()
        num_replicas = dist.get_world_size()
    else:
        return 

    overall_start = dataset.start
    overall_end = dataset.end
    per_replica = int(math.ceil((overall_end - overall_start) / float(num_replicas)))
    dataset.start = overall_start + replica_id * per_replica
    dataset.end = min(dataset.start + per_replica, overall_end)

    print("{} has been partitioned by {} distributed training replicas. This instance is from {} to {}.".format(dataset.source, num_replicas, dataset.start, dataset.end))

def multi_process_load_partitioner(dataset):
    data_worker_info = torch.utils.data.get_worker_info()
    if data_worker_info is None:
        return

    overall_start = dataset.start
    overall_end = dataset.end

    # handle multi process data loading
    data_worker_id = data_worker_info.id
    num_data_workers = data_worker_info.num_workers

    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(num_data_workers)))
    dataset.start = overall_start + data_worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

    print("{} has been partitioned by {} data loading processes. This instance is from {} to {}.".format(dataset.source, num_data_workers, dataset.start, dataset.end))

class MultiProcessVoxCelebData(DataFlow):
    source = 'VoxCelebData'

    def __init__(self, root, id_sampling=True):
        self.root = root
        self.id_sampling = id_sampling

        vid_list = []
        for folder in glob.glob(os.path.join(root, '**/*/*/*/'), recursive=True):
            rel_path = os.path.relpath(folder, root).strip('/')
            vid_list.append(rel_path)

        vid_tree = defaultdict(list)
        for vid in vid_list:
            person_id, video_id, clip_id = vid.split('/')
            if self.id_sampling:
                vid_tree[person_id].append((video_id, clip_id))
            else:
                vid_tree[vid] = []
        self.vid_tree = vid_tree

        print("Found {} person ids, {} total video clips in {}.".format(len(self.vid_tree), len(vid_list), self.root))

        # distributed partitioner must be called in initializer, otherwise dist.is_initialized() always returns False even if distributed training is enabled
        self.start = 0
        self.end = len(self.vid_tree)
        distributed_partitioner(self)
    
    def __len__(self):
        raise NotImplementedError("This dataflow is indefinite and does not have a __len__() method.")

    def __iter__(self):
        # convert dict to list of (key, val) pairs in order to indexing and splitting
        sampling_source = [(k, v) for k, v in self.vid_tree.items()]
        multi_process_load_partitioner(self)
        worker_sampling_source = sampling_source[self.start:self.end]

        while 1:
            random.shuffle(worker_sampling_source)
            if self.id_sampling:
                # randomly select a video from each person id.
                # each entry in sampled_video is a tuple of (video_id, start#end)
                sampled_videos = [(id, random.choice(videos)) for id, videos in worker_sampling_source]    
                # now each entry becomes person_id/video_id/start#end
                sampled_videos = [os.path.join(id, *vid_start_end) for id, vid_start_end in sampled_videos]
            else:
                # convert each entry from person_id#video_id#start#end to person_id/video_id/start#end
                sampled_videos = [vid.replace('#', '/', 2) for vid, _ in worker_sampling_source]
            
            for vid_name in sampled_videos:
                vid_dir = os.path.join(self.root, vid_name)
                img_paths = [x for x in glob.glob(os.path.join(vid_dir, '*_img.jpg'))]
                num_frames = len(img_paths)
                if num_frames < 2:
                    #print("num_frames less than 2 for {}".format(vid_dir))
                    continue

                selected_pair = np.sort(np.random.choice(num_frames, replace=False, size=2))

                res = dict()
                for i in range(2):
                    img_path = img_paths[selected_pair[i]]
                    data = read_data(img_path)
                    for k, v in data.items():
                        res[k+str(i)] = v

                yield res