import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

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

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    net.train()
    optimizer.zero_grad()
    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            masks = (masks > 0).float()
            
            imgs, pt, masks, generated_labels = generate_click_prompt(imgs, masks)
            point_labels = generated_labels
            
            name = pack['image_meta_dict']['filename_or_obj']

            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                generated_labels = rearrange(generated_labels, 'b d -> (b d)')
                point_labels = generated_labels 
                imgs = imgs.repeat(1,3,1,1)
                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                showp = pt.clone()
            else:
                orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
                imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
                scale_h = args.image_size / orig_h
                scale_w = args.image_size / orig_w
                pt = pt.float()
                pt[..., 0] = pt[..., 0] * scale_h
                pt[..., 1] = pt[..., 1] * scale_w 
                showp = pt.clone()

            mask_type = torch.float32
            ind += 1
            point_coords = pt[..., [1, 0]] 
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            if coords_torch.dim() == 2: coords_torch = coords_torch.unsqueeze(1)
            if labels_torch.dim() == 1: labels_torch = labels_torch.unsqueeze(1)
            pt_model = (coords_torch, labels_torch)

            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters(): 
                    if "Adapter" not in n: value.requires_grad = False
                    else: value.requires_grad = True
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    rankallocator = lora.RankAllocator(net.image_encoder, lora_r=4, target_rank=8, init_warmup=500, final_warmup=1500, mask_interval=10, total_step=3000, beta1=0.85, beta2=0.85)
            else:
                for n, value in net.image_encoder.named_parameters(): value.requires_grad = True

            origin_imgs = imgs.clone()        
            imgs = net.preprocess(imgs)        
            imge= net.image_encoder(imgs)

            with torch.no_grad():
                if args.net == 'sam' or args.net == 'mobile_sam':
                    se, de = net.prompt_encoder(points=pt_model, boxes=None, masks=None)
                elif args.net == "efficient_sam":
                    coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                    se = net.prompt_encoder(coords=coords_torch, labels=labels_torch)
            
            if args.net == 'sam':
                pred, _ = net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=(args.multimask_output > 1))
            elif args.net == 'mobile_sam':
                pred, _ = net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=False)
            elif args.net == "efficient_sam":
                se = se.view(se.shape[0], 1, se.shape[1], se.shape[2])
                pred, _ = net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, multimask_output=False)

            pred = F.interpolate(pred,size=(args.out_size,args.out_size), mode="bilinear", align_corners=False)
            loss = lossfunc(pred, masks)
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            if args.mod == 'sam_adalora':
                (loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            if vis and ind % vis == 0:
                namecat = 'Train'
                for na in name[:2]: namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                show_labels = point_labels.clone()
                if show_labels.dim() > 1: show_labels = show_labels.squeeze()
                vis_image(origin_imgs, pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp, point_labels=show_labels)

            pbar.update()

    return epoch_loss / len(train_loader)

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)
    
    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_slices_processed = 0
    noise_filtered_count = 0
    
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            masksw = (masksw > 0).float()
            cur_bsz = imgsw.shape[0]
            
            # Force generate prompts
            imgsw, ptw, masksw, generated_labels_w = generate_click_prompt(imgsw, masksw)
            point_labels = generated_labels_w
            name = pack['image_meta_dict']['filename_or_obj']
            
            buoy = 0
            evl_ch = int(args.evl_chunk) if args.evl_chunk else int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                batch_labels = generated_labels_w[:, buoy: buoy + evl_ch] if args.thd else point_labels
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    labels_torch = rearrange(batch_labels, 'b d -> (b d)')
                    imgs = imgs.repeat(1,3,1,1)
                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                    showp = pt.clone()
                else:
                    orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                    scale_h = args.image_size / orig_h
                    scale_w = args.image_size / orig_w
                    pt = pt.float()
                    pt[..., 0] = pt[..., 0] * scale_h
                    pt[..., 1] = pt[..., 1] * scale_w
                    showp = pt.clone()
                    current_point_labels = batch_labels
                
                ind += 1
                point_coords = pt[..., [1, 0]]
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                if not args.thd: labels_torch = torch.as_tensor(current_point_labels, dtype=torch.int, device=GPUdevice)
                if coords_torch.dim() == 2: coords_torch = coords_torch.unsqueeze(1)
                if labels_torch.dim() == 1: labels_torch = labels_torch.unsqueeze(1)
                pt_model = (coords_torch, labels_torch)

                mask_type = torch.float32
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                with torch.no_grad():
                    origin_imgs = imgs.clone()
                    imgs = net.preprocess(imgs)
                    imge= net.image_encoder(imgs)
                    
                    if args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net.prompt_encoder(points=pt_model, boxes=None, masks=None)
                    elif args.net == "efficient_sam":
                        coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                        se = net.prompt_encoder(coords=coords_torch, labels=labels_torch)

                    if args.net == 'sam':
                        pred, _ = net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=(args.multimask_output > 1))
                    elif args.net == 'mobile_sam':
                        pred, _ = net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=False)
                    elif args.net == "efficient_sam":
                        se = se.view(se.shape[0], 1, se.shape[1], se.shape[2])
                        pred, _ = net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, multimask_output=False)

                    pred = F.interpolate(pred,size=(args.out_size,args.out_size), mode="bilinear", align_corners=False)
                    total_loss += lossfunc(pred, masks) * cur_bsz

                    # Dice/IoU Calculation with Noise Filter
                    pred_prob = torch.sigmoid(pred)
                    pred_binary = (pred_prob > 0.5).float()
                    masks_binary = (masks > 0.5).float()
                    # 不再使用黑盒的 eval_seg()，改成手動計算每張切片的指標，方便之後加入 noise filter 與 slice-level debug
                    
                    current_batch_size = pred_binary.shape[0]
                    total_slices_processed += current_batch_size 

                    for i in range(current_batch_size):
                        p_i = pred_binary[i]
                        t_i = masks_binary[i]
                        
                        # Noise Filter (Threshold = 50)
                        if p_i.sum() < 50:
                            if p_i.sum() > 0: noise_filtered_count += 1
                            p_i = torch.zeros_like(p_i)

                        intersection = (p_i * t_i).sum()
                        total_area = p_i.sum() + t_i.sum()
                        union_area = total_area - intersection
                        
                        if total_area < 1: dice_score = 1.0
                        else: dice_score = float(((2.0 * intersection) / (total_area + 1e-6)).item())
                        
                        if union_area < 1: iou_score = 1.0
                        else: iou_score = float((intersection / (union_area + 1e-6)).item())
                            
                        total_dice += dice_score
                        total_iou += iou_score

                    # Visualizations
                    if args.vis and ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name[:2]: namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                        show_labels = labels_torch.clone().view(-1)

                        vis_image(origin_imgs, pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp, point_labels=show_labels)
                        
                        debug_save_path = os.path.join(args.path_helper['sample_path'], namecat + 'DEBUG_dice_' + str(epoch) + '.jpg')
                        debug_dice_visualization(origin_imgs[0], pred_prob[0], masks[0], debug_save_path, threshold=0.5)
                        
            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    if total_slices_processed == 0: total_slices_processed = 1
    avg_loss = total_loss / total_slices_processed
    avg_iou = total_iou / total_slices_processed
    avg_dice = total_dice / total_slices_processed

    return avg_loss, (avg_iou, avg_dice)

def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1); label = label.transpose(0,1)
    coord = coord.unsqueeze(1); label = label.unsqueeze(1)
    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)
    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[:, :, : decoder_max_num_input_points, :]
        label = label[:, :, : decoder_max_num_input_points]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(rescaled_batched_points, (0, 0, 0, decoder_max_num_input_points - num_pts), value=-1.0)
        label = F.pad(label, (0, decoder_max_num_input_points - num_pts), value=-1.0)
    rescaled_batched_points = rescaled_batched_points.reshape(batch_size * max_num_queries, decoder_max_num_input_points, 2)
    label = label.reshape(batch_size * max_num_queries, decoder_max_num_input_points)
    return rescaled_batched_points,label

def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
    return torch.stack([
        torch.where(batched_points[..., 0] >= 0, batched_points[..., 0] * 1024 / input_w, -1.0),
        torch.where(batched_points[..., 1] >= 0, batched_points[..., 1] * 1024 / input_h, -1.0),
    ], dim=-1)

def debug_dice_visualization(origin_img, pred_prob, true_mask, save_path, threshold=0.5):
    """
    Overlay visualization with High-DPI Times New Roman legend.
    Uses super-sampling (scaling up) to make text sharp.
    Layout: 2 Rows (2 Cols)
    Legend Height: 40 (Base)
    Border: Thinner (1px)
    Position: Adjusted lower
    """
    # ---------------------------------------------------------
    # Part 1: 數據處理與 OpenCV 繪圖
    # ---------------------------------------------------------
    if torch.is_tensor(origin_img):
        img = origin_img.detach().cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 255).astype(np.uint8)
    else:
        img = origin_img

    if torch.is_tensor(pred_prob):
        p = pred_prob.detach().cpu().squeeze().numpy()
    else:
        p = pred_prob
    p_bin = (p > threshold).astype(np.uint8) 

    if torch.is_tensor(true_mask):
        t = true_mask.detach().cpu().squeeze().numpy()
    else:
        t = true_mask
    t_bin = (t > 0.5).astype(np.uint8)

    if img.shape[0] != p_bin.shape[0] or img.shape[1] != p_bin.shape[1]:
        img = cv2.resize(img, (p_bin.shape[1], p_bin.shape[0]))

    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = p_bin.shape
    TP = (p_bin == 1) & (t_bin == 1)
    FP = (p_bin == 1) & (t_bin == 0)
    FN = (p_bin == 0) & (t_bin == 1)
    
    overlay = img_bgr.copy()
    overlay[TP] = [0, 255, 0]   
    overlay[FP] = [0, 0, 255]   
    overlay[FN] = [255, 0, 0]   
    
    alpha = 0.4
    combined = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    
    t_bin_cv = t_bin.astype(np.uint8)
    contours, _ = cv2.findContours(t_bin_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(combined, contours, -1, (0, 255, 255), 1) 

    # ---------------------------------------------------------
    # Part 2: Pillow 高解析度文字繪製
    # ---------------------------------------------------------
    
    # 1. 轉為 PIL 圖片
    pil_img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    
    # ★★★ 放大圖片以提高文字 DPI ★★★
    scale_factor = 3
    
    w, h = pil_img.size
    new_w, new_h = w * scale_factor, h * scale_factor
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 2. 設置 Legend 區域參數
    legend_h = 40 * scale_factor       
    font_size = 13 * scale_factor      
    color_box_size = 12 * scale_factor 
    padding = 10 * scale_factor        
    
    # 3. 建立含 Legend 的新畫布
    new_img = Image.new('RGB', (new_w, new_h + legend_h), (255, 255, 255))
    new_img.paste(pil_img, (0, 0))
    
    draw_legend = ImageDraw.Draw(new_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
        print("Warning: fallback font not found, using default font.")

    items = [
        ("TP (Hit)", (0, 255, 0)),
        ("FP (Over)", (255, 0, 0)),
        ("FN (Miss)", (0, 0, 255)),
        ("GT Boundary", (255, 255, 0))
    ]
    
    # 5. 排版 (2 cols x 2 rows)
    cols = 2
    step_x = new_w // cols
    row_height = legend_h // 2
    
    start_x = 19 * scale_factor
    
    # ★★★ [修改 1] 增加頂部間距 (5 -> 9.5)，讓圖示整體往下移 ★★★
    start_y_base = new_h + (9.5 * scale_factor) 
    
    for i, (text, color) in enumerate(items):
        row = i // cols
        col = i % cols
        
        x = start_x + col * step_x
        # 計算色塊的 Y 座標
        y = start_y_base + (row * row_height) - (5 * scale_factor) # 稍微往上拉一點以置中於 row
        
        # 畫色塊 (邊框 1 * scale)
        draw_legend.rectangle(
            [x, y, x + color_box_size, y + color_box_size], 
            fill=color, 
            outline="black", 
            width=1 * scale_factor
        )
        
        # 寫字
        text_x = x + color_box_size + (5 * scale_factor)
        
        # ★★★ [修改 2] 文字垂直位置微調 ★★★
        text_y = y - 3.8
        
        draw_legend.text((text_x, text_y), text, fill="black", font=font)

    new_img.save(save_path)

    intersection = np.sum(TP)
    sum_pred = np.sum(p_bin)
    sum_gt = np.sum(t_bin)
    calc_dice = (2.0 * intersection) / (sum_pred + sum_gt + 1e-6)
    if sum_pred == 0 and sum_gt == 0: calc_dice = 1.0

    msg = (f"\n[Dice Diagnosis] {save_path.split('/')[-1]}\n"
           f"  Intersection(TP): {intersection}, Pred: {sum_pred}, GT: {sum_gt}\n"
           f"  Calculated Dice: {calc_dice:.4f}")
    
    return msg
