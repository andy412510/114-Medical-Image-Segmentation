# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cfg
import function
from conf import settings
from dataset import *
from utils import *


def plot_training_curves(train_losses, val_dices, val_ious, save_path,
                          best_dice, best_iou, best_epoch, moving_avg_window=15):
    """
    畫出訓練曲線：
    上半部：Training Loss（含移動平均）
    下半部：Validation Dice 和 IoU
    底部標注 Best Dice、Best IoU、Best Epoch
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    epochs = list(range(len(train_losses)))

    # 上半部：Training Loss
    ax1.plot(epochs, train_losses, color='#AED6F1', alpha=0.5, linewidth=0.8, label='Train loss')
    if len(train_losses) >= moving_avg_window:
        moving_avg = np.convolve(train_losses,
                                  np.ones(moving_avg_window) / moving_avg_window,
                                  mode='valid')
        ma_epochs = list(range(moving_avg_window - 1, len(train_losses)))
        ax1.plot(ma_epochs, moving_avg, color='#2980B9', linewidth=1.5,
                 label=f'Train loss ({moving_avg_window}-epoch moving avg)')
    ax1.set_title('Training Loss')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(train_losses))

    # 下半部：Validation Dice & IoU
    val_epochs  = sorted(val_dices.keys())
    dice_values = [val_dices[e] for e in val_epochs]
    iou_values  = [val_ious[e]  for e in val_epochs]

    ax2.plot(val_epochs, dice_values, color='#27AE60', linewidth=1.5, label='DICE')
    ax2.plot(val_epochs, iou_values,  color='#E67E22', linewidth=1.5, label='IOU')
    ax2.set_title('Validation Metrics (DICE and IOU)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric Value')
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(train_losses))

    # 底部文字標注
    info_text = (
        f'Best DICE: {best_dice:.4f} @ checkpoint epoch {best_epoch}\n'
        f'Best IOU:  {best_iou:.4f} @ checkpoint epoch {best_epoch}'
    )
    fig.text(0.02, 0.01, info_text, fontsize=8, verticalalignment='bottom',
             color='#2C3E50')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():

    args = cfg.parse_args()

    seed = args.seed
    set_seed(seed)

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights, strict=False)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']

        net.load_state_dict(checkpoint['state_dict'], strict=False)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    nice_train_loader, nice_test_loader = get_dataloader(args)

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_acc  = 0.0
    best_tol  = 1e4
    best_dice = 0.0
    best_iou  = 0.0
    best_epoch_record = 0

    # 記錄每個 epoch 的數值
    train_losses = []
    val_dices    = {}   # {epoch: dice}
    val_ious     = {}   # {epoch: iou}

    for epoch in range(settings.EPOCH):

        if epoch < 5:
            if args.dataset != 'REFUGE':
                tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            else:
                tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {epoch}.')
            val_dices[epoch] = edice if args.dataset != 'REFUGE' else edice_cup
            val_ious[epoch]  = eiou  if args.dataset != 'REFUGE' else eiou_cup

        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis=args.vis)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        # 記錄 train loss
        train_losses.append(loss)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH - 1:
            if args.dataset != 'REFUGE':
                tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            else:
                tol, (eiou_cup, eiou_disc, edice_cup, edice_disc) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                logger.info(f'Total score: {tol}, IOU_CUP: {eiou_cup}, IOU_DISC: {eiou_disc}, DICE_CUP: {edice_cup}, DICE_DISC: {edice_disc} || @ epoch {epoch}.')

            cur_dice = edice if args.dataset != 'REFUGE' else edice_cup
            cur_iou  = eiou  if args.dataset != 'REFUGE' else eiou_cup
            val_dices[epoch] = cur_dice
            val_ious[epoch]  = cur_iou

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if cur_dice > best_dice:
                best_dice  = cur_dice
                best_iou   = cur_iou
                best_tol   = tol
                best_epoch_record = epoch
                is_best = True

                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.net,
                    'state_dict': sd,
                    'optimizer': optimizer.state_dict(),
                    'best_tol': best_dice,
                    'path_helper': args.path_helper,
                }, is_best, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth")
            else:
                is_best = False

    # 訓練結束後存一張完整的 1~500 曲線圖
    curve_path = os.path.join(args.path_helper['sample_path'], 'training_curve.png')
    plot_training_curves(
        train_losses, val_dices, val_ious,
        curve_path,
        best_dice=best_dice,
        best_iou=best_iou,
        best_epoch=best_epoch_record,
    )
    print(f'[Plot] Training curve saved to {curve_path}')

    writer.close()


if __name__ == '__main__':
    main()