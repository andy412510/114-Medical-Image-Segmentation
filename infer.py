#!/usr/bin/env python3
"""
單一 NIfTI 推論腳本
對整個 volume 的每一張 slice 做分割並輸出視覺化結果
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

import cfg
from utils import get_network, set_log_dir, create_logger, generate_click_prompt
from dataset.btcv import BTCVInference


def main():
    args = cfg.parse_args()
    args.thd = False

    GPUdevice = torch.device('cuda', args.gpu_device)

    # 載入模型
    net = get_network(args, args.net, use_gpu=args.gpu,
                      gpu_device=GPUdevice, distribution=args.distributed)

    # 載入權重
    assert args.weights != 0, "請指定 -weights"
    checkpoint = torch.load(args.weights, map_location=f'cuda:{args.gpu_device}')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    print(f"=> Loaded weights from {args.weights}")

    net.eval()

    # 設定輸出路徑
    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    save_dir = args.path_helper['sample_path']

    # 載入整個 volume
    dataset = BTCVInference(args, nii_path=args.nii_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print(f"Total slices: {len(dataset)}")
    print(f"Output will be saved to: {save_dir}")

    with torch.no_grad():
        for ind, pack in enumerate(tqdm(loader, desc='Inferring')):
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']

            # 直接用 dataset 裡設定好的固定 prompt，不用 generate_click_prompt
            pt = pack['pt'].float()
            point_labels = pack['p_label'].float()

            # 座標縮放（pt 已經是 image_size 的座標，不需要再縮放）

            point_coords = pt[..., [1, 0]]
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            if coords_torch.dim() == 2: coords_torch = coords_torch.unsqueeze(1)
            if labels_torch.dim() == 1: labels_torch = labels_torch.unsqueeze(1)
            pt_model = (coords_torch, labels_torch)

            # 推論
            origin_imgs = imgs.clone()  # 1024x1024，用於視覺化
            imgs_input = net.preprocess(imgs)
            imge = net.image_encoder(imgs_input)
            se, de = net.prompt_encoder(points=pt_model, boxes=None, masks=None)
            pred, _ = net.mask_decoder(
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
            )
            pred = F.interpolate(pred, size=(args.out_size, args.out_size),
                                 mode="bilinear", align_corners=False)

            pred_prob = torch.sigmoid(pred)
            pred_binary = (pred_prob > 0.5).float()

            # 視覺化：用 origin_imgs（1024x1024）做 overlay，跟 val.py 一致
            b, c, h, w = origin_imgs.shape

            # 把原圖 normalize 到 0~1
            img_vis = origin_imgs[0].cpu()
            img_min, img_max = img_vis.min(), img_vis.max()
            img_vis = (img_vis - img_min) / (img_max - img_min + 1e-6)

            # 把 mask resize 到原圖大小
            pred_vis = F.interpolate(pred_binary, size=(h, w), mode='nearest')

            # 轉 numpy 做 overlay
            img_np = img_vis.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            p = pred_vis[0, 0].cpu().numpy().astype(np.uint8)

            overlay = img_bgr.copy()
            overlay[p == 1] = [0, 255, 0]
            combined = cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0)

            contours, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(combined, contours, -1, (0, 255, 255), 1)

            save_path = os.path.join(save_dir, f"{name[0]}.jpg")
            cv2.imwrite(save_path, combined)

    print(f"\n推論完成！結果存放在: {save_dir}")


if __name__ == '__main__':
    main()