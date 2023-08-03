import os
import time
import logging
import math
import argparse
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from utils.util import setup_logger, print_args
from models import Trainer

class SR():
    def __init__(self):
        parser = argparse.ArgumentParser(description='referenceSR Testing')
        parser.add_argument('--random_seed', default=0, type=int)
        parser.add_argument('--name', default='masa_rec_TestSet', type=str)
        parser.add_argument('--phase', default='test', type=str)

        ## device setting
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                            help='job launcher')
        parser.add_argument('--local_rank', type=int, default=0)

        ## network setting
        parser.add_argument('--net_name', default='MASA', type=str, help='')
        parser.add_argument('--sr_scale', default=4, type=int)
        parser.add_argument('--input_nc', default=3, type=int)
        parser.add_argument('--output_nc', default=3, type=int)
        parser.add_argument('--nf', default=64, type=int)
        parser.add_argument('--n_blks', default='4, 4, 4', type=str)
        parser.add_argument('--nf_ctt', default=32, type=int)
        parser.add_argument('--n_blks_ctt', default='2, 2, 2', type=str)
        parser.add_argument('--num_nbr', default=1, type=int)
        parser.add_argument('--n_blks_dec', default=10, type=int)
        parser.add_argument('--ref_level', default=1, type=int)

        ## dataloader setting
        parser.add_argument('--data_root', default='/data/wjk/SR/', type=str)
        parser.add_argument('--dataset', default='CUFED', type=str, help='CUFED')
        parser.add_argument('--crop_size', default=256, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--num_workers', default=1, type=int)
        parser.add_argument('--data_augmentation', default=False, type=bool)

        parser.add_argument('--resume', default='checkpoints/masa_rec.pth', type=str)
        parser.add_argument('--testset', default='TestSet', type=str, help='Sun80 | Urban100 | TestSet_multi')
        parser.add_argument('--save_folder', default='./test_results/', type=str)

        ## setup training environment
        args = parser.parse_args()
        ## setup training device
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpu_ids.append(id)

        #### distributed training settings
        if args.launcher == 'none':  # disabled distributed training
            args.dist = False
            args.rank = -1
            print('Disabled distributed training.')
        else:
            args.dist = True
            init_dist()
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()

        ## test model
        self.trainer = Trainer(args)
        self.args = args

    def inference(self, LR, Ref):
        scale = self.args.sr_scale
        h, w, c = LR.shape
        LR = np.array(Image.fromarray(LR).resize((w // scale, h // scale), Image.BICUBIC))
        LR = LR.astype(np.float32) / 255.
        LR = torch.from_numpy(LR).permute(2, 0, 1).float().unsqueeze(0)

        h, w, c = Ref.shape
        Ref_down = np.array(Image.fromarray(Ref).resize((w // scale, h // scale), Image.BICUBIC))
        Ref_down = Ref_down.astype(np.float32) / 255.
        Ref_down = torch.from_numpy(Ref_down).permute(2, 0, 1).float().unsqueeze(0)

        Ref = Ref.astype(np.float32) / 255.
        Ref = torch.from_numpy(Ref).permute(2, 0, 1).float().unsqueeze(0)

        return self.trainer.inference(LR, Ref, Ref_down)
