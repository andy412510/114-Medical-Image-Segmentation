import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
# 引入 PIL 用於繪製 Times New Roman 字體
from PIL import Image, ImageDraw, ImageFont 
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2 

import cfg
import models.sam.utils.transforms as samtrans
import pytorch_ssim
from conf import settings
from utils import *

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
KIDNEY_LABELS = [2, 3]  # 右腎=2, 左腎=3
POS_WEIGHT_VALUE = 80.0
# ... 前面的 import 與全域變數保持不變 ...

def train_sam(args, net: nn.Module, optimizer, train_loader,
              epoch, writer, schedulers=None, vis=50):
    net.train()
    optimizer.zero_grad()
    epoch_loss = 0.0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # BCE 高權重
    criterion_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT_VALUE]).to(GPUdevice)
    )

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for ind, pack in enumerate(train_loader, 1):
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)

            # 只保留腎臟
            masks_kidney = torch.zeros_like(masks)
            for lbl in KIDNEY_LABELS:
                masks_kidney += (masks == lbl).float()
            masks = masks_kidney

            imgs, pt, masks = generate_click_prompt(imgs, masks)

            # 加負點
            point_labels = torch.ones(pt.shape[0], pt.shape[2], device=pt.device)
            num_points = point_labels.shape[-1]
            num_neg = max(1, num_points // 3)
            neg_idx = torch.randperm(num_points)[:num_neg]
            point_labels[0, neg_idx] = 0

            name = pack['image_meta_dict']['filename_or_obj']

            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w')
                point_labels = rearrange(point_labels, 'b d -> (b d)')

                # ★ 強制通道為 3（解決 342/96 等異常） ★
                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                elif imgs.shape[1] != 3:
                    print(f"[Train thd] Abnormal channels after rearrange: {imgs.shape[1]}, forcing to 3")
                    if imgs.shape[1] > 3:
                        imgs = imgs[:, :3, :, :]  # 取前3通道
                    else:
                        imgs = imgs.repeat(1, 3 - imgs.shape[1], 1, 1)

                imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
                showp = pt.clone()
            else:
                orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
                imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
                scale_h = args.image_size / orig_h
                scale_w = args.image_size / orig_w
                pt = pt.float()
                pt[..., 0] *= scale_h
                pt[..., 1] *= scale_w
                showp = pt.clone()

            point_coords = pt[..., [1, 0]]
            coords_torch = point_coords.to(dtype=torch.float, device=GPUdevice)
            labels_torch = point_labels.to(dtype=torch.int, device=GPUdevice)
            if coords_torch.dim() == 2: coords_torch = coords_torch.unsqueeze(1)
            if labels_torch.dim() == 1: labels_torch = labels_torch.unsqueeze(1)
            pt_model = (coords_torch, labels_torch)

            # 凍結/解凍參數
            if args.mod == 'sam_adpt':
                for n, p in net.image_encoder.named_parameters():
                    p.requires_grad = "Adapter" in n
            elif args.mod in ['sam_lora', 'sam_adalora']:
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    rankallocator = lora.RankAllocator(net.image_encoder, lora_r=4, target_rank=8,
                                                       init_warmup=500, final_warmup=1500,
                                                       mask_interval=10, total_step=3000,
                                                       beta1=0.85, beta2=0.85)
            else:
                for p in net.image_encoder.parameters():
                    p.requires_grad = True

            origin_imgs = imgs.clone()

            # Debug 印出 shape（在 preprocess 前）
            print(f"[Train Batch {ind}] imgs shape before preprocess: {imgs.shape}, dtype: {imgs.dtype}")

            # 強制通道為 3（防呆）
            if imgs.shape[1] != 3:
                print(f"[Train preprocess] Forcing channels to 3 from {imgs.shape[1]}")
                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                elif imgs.shape[1] > 3:
                    imgs = imgs[:, :3, :, :]
                else:
                    imgs = imgs.repeat(1, 3 - imgs.shape[1], 1, 1)

            imgs_processed = net.preprocess(imgs)
            imge = net.image_encoder(imgs_processed)

            # 訓練推論
            if args.net in ['sam', 'mobile_sam']:
                if isinstance(pt_model, tuple):
                    pt_model = (pt_model[0][:, :10, :], pt_model[1][:, :10])
                se, de = net.prompt_encoder(points=pt_model, boxes=None, masks=None)
            elif args.net == "efficient_sam":
                h = w = args.image_size
                coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                se = net.prompt_encoder(coords=coords_torch, labels=labels_torch)

            if args.net == 'sam':
                pred, _ = net.mask_decoder(image_embeddings=imge,
                                           image_pe=net.prompt_encoder.get_dense_pe(),
                                           sparse_prompt_embeddings=se,
                                           dense_prompt_embeddings=de,
                                           multimask_output=(args.multimask_output > 1))
            elif args.net == 'mobile_sam':
                pred, _ = net.mask_decoder(image_embeddings=imge,
                                           image_pe=net.prompt_encoder.get_dense_pe(),
                                           sparse_prompt_embeddings=se,
                                           dense_prompt_embeddings=de,
                                           multimask_output=False)
            elif args.net == "efficient_sam":
                se = se.view(se.shape[0], 1, se.shape[1], se.shape[2])
                pred, _ = net.mask_decoder(image_embeddings=imge,
                                           image_pe=net.prompt_encoder.get_dense_pe(),
                                           sparse_prompt_embeddings=se,
                                           multimask_output=False)

            pred = F.interpolate(pred, size=(args.out_size, args.out_size),
                                 mode="bilinear", align_corners=False)

            # 混合 loss
            loss_dice = lossfunc(pred, masks)
            loss_bce = criterion_bce(pred, masks)
            loss = 0.4 * loss_dice + 0.6 * loss_bce

            pbar.set_postfix(loss=loss.item())
            epoch_loss += loss.item()

            if args.mod == 'sam_adalora':
                (loss + lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            if vis and ind % vis == 0:
                namecat = 'Train_' + '_'.join([na.split('/')[-1].split('.')[0] for na in name[:2]])
                save_path = os.path.join(args.path_helper['sample_path'], f"{namecat}_epoch{epoch}.jpg")
                show_labels = point_labels.clone().squeeze() if point_labels.dim() > 1 else point_labels
                try:
                    vis_image(origin_imgs, pred, masks, save_path, reverse=False, points=showp)
                except Exception as e:
                    print(f"[Train Vis Error] {e}")

            pbar.update()

    return epoch_loss / len(train_loader)

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    net.eval()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_slices_processed = 0
    noise_filtered_count = 0

    criterion_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT_VALUE]).to(GPUdevice)
    )

    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean') if args.thd else criterion_G

    with tqdm(total=len(val_loader), desc='Validation', unit='batch', leave=False) as pbar:
        for batch_idx, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['label'].to(dtype=torch.float32, device=GPUdevice)

            print(f"[Val Debug - Batch {batch_idx}]")
            print(f"  Original label dtype: {masksw.dtype}")
            print(f"  Original unique values: {torch.unique(masksw).cpu().numpy().tolist()}")
            kidney_orig = (torch.isclose(masksw, torch.tensor(2.0, device=GPUdevice)).sum() + 
                           torch.isclose(masksw, torch.tensor(3.0, device=GPUdevice)).sum()).item()
            print(f"  Original kidney pixels (2+3): {kidney_orig:.0f}")

            # 只保留腎臟
            masksw_kidney = torch.zeros_like(masksw)
            for lbl in KIDNEY_LABELS:
                masksw_kidney += (masksw == lbl).float()
            masksw = masksw_kidney

            print(f"  After kidney filter - GT pixels sum: {masksw.sum().item():.0f}")
            print(f"  After filter unique: {torch.unique(masksw).cpu().numpy().tolist()}")

            cur_bsz = imgsw.shape[0]

            imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)

            point_labels = torch.ones(ptw.shape[0], ptw.shape[2], dtype=torch.int, device=GPUdevice)
            num_points = point_labels.shape[-1]
            num_neg = max(1, num_points // 3)
            neg_idx = torch.randperm(num_points)[:num_neg]
            point_labels[0, neg_idx] = 0

            name = pack['image_meta_dict']['filename_or_obj']

            buoy = 0
            evl_ch = int(args.evl_chunk) if args.evl_chunk else imgsw.size(-1)

            while buoy + evl_ch <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:, :, buoy:buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch

                origin_imgs = imgs.clone()

                print(f"[Val Batch {batch_idx} Chunk {buoy//evl_ch}] imgs shape before preprocess: {imgs.shape}, dtype: {imgs.dtype}")

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w')
                    labels_torch = rearrange(point_labels, 'b d -> (b d)')

                    if imgs.shape[1] == 1:
                        imgs = imgs.repeat(1, 3, 1, 1)
                    elif imgs.shape[1] != 3:
                        print(f"[Val thd] Abnormal channels: {imgs.shape[1]}, forcing to 3")
                        if imgs.shape[1] > 3:
                            imgs = imgs[:, :3, :, :]
                        else:
                            imgs = imgs.repeat(1, 3 - imgs.shape[1], 1, 1)

                    imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
                    masks = (masks > 0.0001).float()  # 極低門檻，保留稀釋後的邊界像素
                    showp = pt.clone()
                else:
                    orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
                    imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
                    masks = (masks > 0.0001).float()  # 極低門檻，保留稀釋後的邊界像素
                    scale_h = args.image_size / orig_h
                    scale_w = args.image_size / orig_w
                    pt = pt.float()
                    pt[..., 0] *= scale_h
                    pt[..., 1] *= scale_w
                    showp = pt.clone()
                    labels_torch = point_labels.clone()

                point_coords = pt[..., [1, 0]]
                coords_torch = point_coords.to(dtype=torch.float, device=GPUdevice)
                labels_torch = labels_torch.to(dtype=torch.int, device=GPUdevice)
                if coords_torch.dim() == 2: coords_torch = coords_torch.unsqueeze(1)
                if labels_torch.dim() == 1: labels_torch = labels_torch.unsqueeze(1)
                pt_model = (coords_torch, labels_torch)

                imgs = imgs.to(dtype=torch.float32, device=GPUdevice)

                with torch.no_grad():
                    imgs_processed = net.preprocess(imgs)
                    image_embeddings = net.image_encoder(imgs_processed)

                    if args.net in ['sam', 'mobile_sam']:
                        if isinstance(pt_model, tuple):
                            pt_model = (pt_model[0][:, :10, :], pt_model[1][:, :10])
                        se, de = net.prompt_encoder(points=pt_model, boxes=None, masks=None)
                    elif args.net == "efficient_sam":
                        h = w = args.image_size
                        coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                        se = net.prompt_encoder(coords=coords_torch, labels=labels_torch)

                    if args.net == 'sam':
                        pred, _ = net.mask_decoder(image_embeddings=image_embeddings,
                                                   image_pe=net.prompt_encoder.get_dense_pe(),
                                                   sparse_prompt_embeddings=se,
                                                   dense_prompt_embeddings=de,
                                                   multimask_output=(args.multimask_output > 1))
                    elif args.net == 'mobile_sam':
                        pred, _ = net.mask_decoder(image_embeddings=image_embeddings,
                                                   image_pe=net.prompt_encoder.get_dense_pe(),
                                                   sparse_prompt_embeddings=se,
                                                   dense_prompt_embeddings=de,
                                                   multimask_output=False)
                    elif args.net == "efficient_sam":
                        se = se.view(se.shape[0], 1, se.shape[1], se.shape[2])
                        pred, _ = net.mask_decoder(image_embeddings=image_embeddings,
                                                   image_pe=net.prompt_encoder.get_dense_pe(),
                                                   sparse_prompt_embeddings=se,
                                                   multimask_output=False)

                    if pred.dim() == 5:
                        b, c, d, h, w = pred.shape
                        pred = rearrange(pred, 'b c d h w -> (b d) c h w')
                        pred = F.interpolate(pred, size=(args.out_size, args.out_size), mode="bilinear", align_corners=False)
                        pred = rearrange(pred, '(b d) c h w -> b c d h w', b=b)
                    else:
                        pred = F.interpolate(pred, size=(args.out_size, args.out_size),
                                             mode="bilinear", align_corners=False)

                    loss_dice = lossfunc(pred, masks)
                    loss_bce = criterion_bce(pred, masks)
                    loss = 0.4 * loss_dice + 0.6 * loss_bce
                    total_loss += loss.item() * cur_bsz

                    pred_prob = torch.sigmoid(pred)
                    pred_binary = (pred_prob > 0.5).float()
                    masks_binary = (masks > 0.5).float()

                    current_batch_size = pred_binary.shape[0]
                    total_slices_processed += current_batch_size

                    for i in range(current_batch_size):
                        p_i = pred_binary[i]
                        t_i = masks_binary[i]

                        if p_i.sum() < 10:
                            if p_i.sum() > 0:
                                noise_filtered_count += 1
                            p_i = torch.zeros_like(p_i)

                        intersection = (p_i * t_i).sum()
                        total_area = p_i.sum() + t_i.sum()
                        union_area = total_area - intersection

                        dice_score = 1.0 if total_area < 1 else float((2.0 * intersection / (total_area + 1e-6)).item())
                        iou_score = 1.0 if union_area < 1 else float((intersection / (union_area + 1e-6)).item())

                        total_dice += dice_score
                        total_iou += iou_score

                    # 強制產生 debug 圖（每個 batch 都執行）
                    namecat = 'Val'
                    img_name = f"{namecat}_epoch{epoch}_batch{batch_idx}.jpg"
                    save_path = os.path.join(args.path_helper['sample_path'], img_name)

                    show_labels = labels_torch.clone().squeeze(-1) if labels_torch.dim() > 1 else labels_torch

                    try:
                        vis_image(origin_imgs, pred, masks, save_path, reverse=False, points=showp)
                    except Exception as e:
                        print(f"[Val Vis Error] {e}")

                    debug_save_path = os.path.join(
                        args.path_helper['sample_path'],
                        f"{namecat}_DEBUG_dice_epoch{epoch}_batch{batch_idx}.jpg"
                    )
                    os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
                    print(f"[Val Debug - Batch {batch_idx} Chunk {buoy//evl_ch}]")
                    print(f"  GT kidney pixels sum: {masks.sum().item():.0f}")
                    print(f"  GT shape: {masks.shape}, unique values: {torch.unique(masks).tolist()}")
                    debug_dice_visualization(origin_imgs[0], pred_prob[0], masks[0], debug_save_path, threshold=0.5)

            pbar.update(1)

    if args.evl_chunk and evl_ch > 0:
        slices_per_case = imgsw.size(-1) // evl_ch
        n_val = len(val_loader) * slices_per_case

    total_slices_processed = max(total_slices_processed, 1)
    avg_loss = total_loss / total_slices_processed
    avg_iou = total_iou / total_slices_processed
    avg_dice = total_dice / total_slices_processed

    print(f"Validation noise filtered: {noise_filtered_count} slices")

    return avg_loss, (avg_iou, avg_dice)


def debug_dice_visualization(origin_img, pred_prob, true_mask, save_path, threshold=0.5):
    """
    最終修正版：尺寸對齊 + Canny 強制青色邊界 + 完整 debug + 路徑確認
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    import os

    print(f"[ENTER debug_dice_visualization] Processing for {os.path.basename(save_path)}")

    # 確保儲存資料夾存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Part 1: 處理 origin_img
    if torch.is_tensor(origin_img):
        img_tensor = origin_img.detach().cpu()
        while img_tensor.dim() > 3:
            img_tensor = img_tensor[0]

        if img_tensor.dim() == 2:
            img = img_tensor.numpy()
        elif img_tensor.dim() == 3:
            c, h, w = img_tensor.shape
            if c == 1:
                img = img_tensor.squeeze(0).numpy()
            elif c == 3:
                img = img_tensor.permute(1, 2, 0).numpy()
            else:
                print(f"[DEBUG] Abnormal channels: {c}, shape={img_tensor.shape}, min/max={img_tensor.min():.2f}/{img_tensor.max():.2f}")
                mid = c // 2
                img = img_tensor[mid].numpy()
        else:
            raise ValueError(f"Unsupported tensor dim: {img_tensor.dim()}")
    else:
        img = np.asarray(origin_img)
        if img.ndim == 3 and img.shape[-1] not in (1, 3, 4):
            img = img[..., 0]
        if img.ndim == 2:
            img = np.stack([img]*3, -1)

    img = img.astype(np.float32)
    if img.max() <= 1.01:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h_img, w_img = img_bgr.shape[:2]

    # Part 2: 處理 true_mask (t)
    t = true_mask.detach().cpu().squeeze().numpy() if torch.is_tensor(true_mask) else np.squeeze(np.asarray(true_mask))

    # 加強 debug 印出
    print(f"[Debug Vis - {os.path.basename(save_path)}]")
    print(f"  GT raw min: {t.min():.6f}, max: {t.max():.6f}, mean: {t.mean():.6f}")
    print(f"  GT shape: {t.shape}")
    print(f"  Image shape: {h_img}x{w_img}")
    print(f"  GT non-zero pixels (>0): {np.sum(t > 0)}")
    print(f"  GT pixels >0.01: {np.sum(t > 0.01)}")
    print(f"  GT pixels >0.001: {np.sum(t > 0.001)}")

    # Canny 邊緣檢測（先計算，再 resize 到原圖大小）
    gray_t = (t * 255).astype(np.uint8)
    edges = cv2.Canny(gray_t, 10, 50)  # 低門檻確保邊緣出現

    # 關鍵修正：resize edges 到與 img_bgr 相同尺寸
    edges_resized = cv2.resize(edges, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

    combined = img_bgr.copy()
    combined[edges_resized > 0] = [0, 255, 255]  # 青色邊緣

    # overlay FP（透明度極低，讓青色突出）
    pred_prob_np = pred_prob.detach().cpu().squeeze().numpy() if torch.is_tensor(pred_prob) else np.squeeze(np.asarray(pred_prob))
    pred_bin = (pred_prob_np > 0.5).astype(np.uint8)

    if pred_bin.shape[:2] != (h_img, w_img):
        pred_bin = cv2.resize(pred_bin, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

    overlay = img_bgr.copy()
    overlay[pred_bin == 1] = [0, 0, 255]  # 紅色 FP
    combined = cv2.addWeighted(overlay, 0.05, combined, 0.95, 0)  # FP 幾乎透明

    # 簡單 legend
    pil_img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

    legend_h = 80
    new_img = Image.new('RGB', (pil_img.width, pil_img.height + legend_h), (255, 255, 255))
    new_img.paste(pil_img, (0, 0))
    draw = ImageDraw.Draw(new_img)

    font = ImageFont.load_default()
    items = [
        ("TP (Hit)", (0, 255, 0)),
        ("FP (Over)", (255, 0, 0)),
        ("FN (Miss)", (0, 0, 255)),
        ("GT Boundary (青色)", (0, 255, 255))
    ]

    y = pil_img.height + 10
    for text, color in items:
        draw.rectangle([10, y, 30, y+20], fill=color)
        draw.text((40, y), text, fill="black", font=font)
        y += 25

    new_img.save(save_path)

    print(f"  Saved debug image with forced Canny GT edges: {save_path}")
    print(f"  Absolute path: {os.path.abspath(save_path)}")