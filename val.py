#!/usr/bin/env  python3

""" valuate network using pytorch
    Junde Wu
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function


def main():
    args = cfg.parse_args()
    if args.dataset == 'refuge' or args.dataset == 'refuge2':
        args.data_path = '../dataset'

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

    '''load pretrained model'''
    assert args.weights != 0
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    
    # 載入權重檔案
    checkpoint = torch.load(checkpoint_file, map_location=loc)

    # --- 核心相容性修改開始 ---
    # 使用 .get() 抓取，如果找不到 key 就不會報錯 KeyError
    start_epoch = checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0
    best_tol = checkpoint.get('best_tol', 1e4) if isinstance(checkpoint, dict) else 1e4

    # 自動判斷結構：是包含多個標籤的字典，還是純參數字典
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # 如果檔案本身就是參數字典 (例如 KiTS23 的 pth)
        state_dict = checkpoint
    # --- 核心相容性修改結束 ---

    if args.distributed != 'none':
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:] # remove `module.`
            name = 'module.' + k
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict

    # 載入參數到網路中
    net.load_state_dict(new_state_dict)
    print(f'=> Successfully loaded checkpoint (epoch {start_epoch})')

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    '''segmentation data'''
    nice_train_loader, nice_test_loader = get_dataloader(args)

    '''begain valuation'''
    best_acc = 0.0
    best_tol = 1e4

    if args.mod == 'sam_adpt':
        net.eval()

        if args.dataset != 'REFUGE':
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, start_epoch, net)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {start_epoch}.')
        else:
            tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function.validation_sam(args, nice_test_loader, start_epoch, net)
            logger.info(f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {start_epoch}.')


if __name__ == '__main__':
    main()