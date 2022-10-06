import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

import os
import cv2
import argparse
from shutil import rmtree, copytree
import numpy as np    
import re

from Data.train_data import Dataset
from config import config as cfg
from trainer import Trainer
from utils import SummaryLogger, create_tqdm_bar, clean_folders, set_rand_seed

import traceback
import pdb

def setup_log_dir(args):
    # ============= set up logger
    train_dir = args.train_dir
    clean_folders([train_dir])
    log_dir = os.path.join(train_dir, 'logger')
    return log_dir


def main(rank, args):
    # init process group
    if args.world_size > 1:
        dist.init_process_group(                                   
            backend='nccl',                                         
            world_size=args.world_size,                              
            rank=rank                                               
        )                   

    model_cfg = getattr(cfg, args.model_version.upper())
    train_cfg = getattr(cfg, 'TRAIN')

    # update config
    cfg.freeze(False)
    if len(args.config) > 0:
        cfg.update_args(args.config.split(';'))
    model_cfg.MODEL_VERSION = args.model_version.lower()
    cfg.freeze()
    if args.debug_run:
        cfg.freeze(False)
        train_cfg.MAX_EPOCH = 10
        train_cfg.STEPS_PER_EPOCH = 100
        train_cfg.SAVE_PER_K_EPOCHS = 1
        train_cfg.SUMMARY_PERIOD = 5
        train_cfg.LR_UPDATE_PERIOD = 5
        cfg.freeze()


    # fix random seed
    set_rand_seed(train_cfg.SEED + rank)

    # ============== setup data loader
    ds = Dataset(train_cfg.DATA_ROOT, train_cfg.IMG_SIZE)
    data_loader = torch.utils.data.DataLoader(ds, batch_size=train_cfg.BATCH_SIZE // args.world_size)   # GPUs are fully occupied; num_worker > 1 doesn't help

    # ============== setup model
    trainer = Trainer(train_cfg, model_cfg, rank, args.init_ckpt_file)
    device0 = 'cuda:{}'.format(rank)

    if args.reset_global_step >= 0:
        trainer.global_step = args.reset_global_step
    if args.reset_learning_rate is not None:
        g_lr, d_lr = args.reset_learning_rate
        trainer.set_learning_rate(g_lr, d_lr)
    
    if rank == 0:
        print("Config: ------------------------------------------\n" + str(cfg) + '\n')
        summary = SummaryLogger(args.log_dir) 
        pbar = create_tqdm_bar(total=train_cfg.STEPS_PER_EPOCH, desc='epoch 1')

    local_step = 0
    try:
        for data in data_loader:
            src_img = torch.as_tensor(data['img'][:, 0], device=device0, dtype=torch.float32).permute([0, 3, 1, 2]) / 255.
            dst_img = torch.as_tensor(data['img'][:, 1], device=device0, dtype=torch.float32).permute([0, 3, 1, 2]) / 255.
            src_verts = torch.as_tensor(data['posed_verts'][:, 0], device=device0, dtype=torch.float32)
            dst_verts = torch.as_tensor(data['posed_verts'][:, 1], device=device0, dtype=torch.float32)

            with torch.autograd.set_detect_anomaly(args.debug_run):
                trainer.step(src_img, dst_img, src_verts, dst_verts)

            global_step = trainer.global_step
            epoch = global_step // train_cfg.STEPS_PER_EPOCH
            local_step = global_step % train_cfg.STEPS_PER_EPOCH
            epoch_done = (local_step == 0)

            # update learning rate and other hyper params if human_hyper_param_setter.txt is present
            # this block should be executed for all gpu workers
            hyper_param_file = os.path.join(args.log_dir, 'human_hyper_param_setter.txt')
            if (global_step -1) % train_cfg.PARAM_UPDATE_PERIOD == 0 and os.path.isfile(hyper_param_file):
                hyper_params = {}
                try:
                    with open(hyper_param_file, 'r') as f:
                        for line in f:
                            key, val = line.strip('\n').split(':')
                            hyper_params[key] = val
                    if 'g_lr' in hyper_params.keys() and 'd_lr' in hyper_params.keys():
                        trainer.set_learning_rate(float(hyper_params['g_lr']), float(hyper_params['d_lr'])) 
                        #print("learning rate has been updated with g_lr={}, d_lr={}".format(hyper_params['g_lr'], hyper_params['d_lr']))
                    if 'TRAIN.G_PERIOD' in hyper_params.keys():
                        cfg.freeze(False)
                        train_cfg.TRAIN.G_PERIOD = int(hyper_params['TRAIN.G_PERIOD'])
                        cfg.freeze()
                    if 'TRAIN.D_PERIOD' in hyper_params.keys():
                        cfg.freeze(False)
                        train_cfg.TRAIN.D_PERIOD = int(hyper_params['TRAIN.D_PERIOD'])
                        cfg.freeze()
                except:
                    traceback.print_exc()
                    print("An error happened when trying to update hyper parameters. The update will be ignored and this won't affect the training process.")

            if rank == 0:
                pbar.update()

                # save summaries to tensorboard
                if (global_step + 1) % train_cfg.SUMMARY_PERIOD == 0:
                    summary.write_summaries(trainer.summary_collector, trainer.global_step, 5)

                if epoch_done:     #  current epoch done. next step will be new epoch
                    # save ckpt
                    if epoch % train_cfg.SAVE_PER_K_EPOCHS == 0:
                        ckpt_file = os.path.join(args.log_dir, '{:010d}.pth'.format(global_step))
                        trainer.save_ckpt(ckpt_file)

                    pbar.close()
                    pbar = create_tqdm_bar(total=train_cfg.STEPS_PER_EPOCH, desc='epoch {}'.format(epoch + 1))

                # save final ckpt before termination
                if epoch_done and epoch + 1 >= train_cfg.MAX_EPOCH:
                    ckpt_file = os.path.join(args.log_dir, '{:010d}.pth'.format(global_step))
                    trainer.save_ckpt(ckpt_file)
                    pbar.close()
                    print("Max #epochs reached. Training finished.")
                    break

    except KeyboardInterrupt:
        if rank == 0:
            # always save ckpt upon key interruption
            print("Don't interrupt. Waiting for model to be saved...")
            ckpt_file = os.path.join(args.log_dir, '{:010d}-force_terminated.pth'.format(global_step))
            trainer.save_ckpt(ckpt_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', default='model', help="Model version to use.")
    parser.add_argument('--debug_run', default=False, type=bool, help="Whether to run the training for debug purposes.")
    parser.add_argument('--gpus', nargs='+', type=int, help="GPUs to use")
    parser.add_argument('--train_dir', default='./train_log', help="Directory where to write training log.")
    parser.add_argument('--config', default='', help="Config params. Each config param is separated by ';'.")
    parser.add_argument('--init_ckpt_file', default='', help="Init checkpoint file.")
    parser.add_argument('--reset_global_step', default=-1, type=int, help="Whether to reset global step.")
    parser.add_argument('--reset_learning_rate', nargs=2, type=float, help="Whether to reset learning rate. First being g_lr, second being d_lr.")
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    args.log_dir = setup_log_dir(args)

    # ============== set up for distributed trainining
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in args.gpus])
    args.world_size = len(args.gpus)
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'             
        os.environ['MASTER_PORT'] = '1235'      
        mp.spawn(main, nprocs=args.world_size, args=(args,))   
    else:
        main(0, args)