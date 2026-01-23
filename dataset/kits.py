import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import cv2  # 需要安裝: pip install opencv-python
from utils import random_click

class KITS(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click'):
        self.args = args
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.out_size = args.out_size  # 確保 args 裡有這個參數，通常 sam 輸出是 256
        
        # 讀取上一輪生成的 dataset.json
        json_path = os.path.join(data_path, "dataset.json")
        with open(json_path, 'r') as f:
            data_info = json.load(f)
            
        # 根據 mode 選擇對應的列表
        key = 'training' if mode == 'Training' else 'validation'
        self.data_list = data_info[key]
        
        print(f"[{mode}] Loaded {len(self.data_list)} samples from {json_path}")

    def __len__(self):
        return len(self.data_list)

    def _preprocess_ct(self, img_slice):
        """
        CT 影像預處理: Windowing + Normalization
        Target: Kidney/Tumor/Abdomen
        Window Level: 50, Window Width: 400 (範圍約 -150 ~ 250)
        或是使用較寬的範圍: -200 ~ 400
        """
        min_bound = -200.0
        max_bound = 400.0
        
        # Clip 數值
        img_slice = np.clip(img_slice, min_bound, max_bound)
        
        # Normalize 到 [0, 1]
        img_slice = (img_slice - min_bound) / (max_bound - min_bound)
        
        return img_slice

    def __getitem__(self, index):
        # 1. 取得檔案路徑 (修正相對路徑問題)
        item = self.data_list[index]
        # dataset.json 裡的像 "./imagesTr/xxx.nii.gz"，需要去掉開頭的 ./
        img_rel_path = item['image'].replace('./', '')
        lbl_rel_path = item['label'].replace('./', '')
        
        img_path = os.path.join(self.data_path, img_rel_path)
        mask_path = os.path.join(self.data_path, lbl_rel_path)

        # 2. 讀取 3D NIfTI
        # 載入後的 shape 通常是 (H, W, D)
        img_vol = nib.load(img_path).get_fdata()
        mask_vol = nib.load(mask_path).get_fdata()

        # 3. 3D -> 2D 切片選擇策略
        # 我們不能一次把整個 3D 丟進去，必須選一張 Slice
        h, w, d = mask_vol.shape
        
        if self.mode == 'Training':
            # 訓練時：為了讓模型學到東西，我們儘量選「有標註」的切片
            # 找出所有有標籤的 slice index (沿著 z 軸 summation > 0)
            foreground_slices = np.where(np.sum(mask_vol, axis=(0, 1)) > 0)[0]
            
            if len(foreground_slices) > 0:
                # 80% 機率選有器官的切片，20% 機率隨機選 (包含背景)
                if np.random.rand() < 0.8:
                    slice_idx = np.random.choice(foreground_slices)
                else:
                    slice_idx = np.random.randint(0, d)
            else:
                slice_idx = np.random.randint(0, d)
        else:
            # 驗證時：可以固定取中間，或者簡單隨機 (這裡示範隨機，實際驗證通常會跑全 volume)
            slice_idx = d // 2 

        # 取出 2D Slice
        img = img_vol[:, :, slice_idx]
        mask = mask_vol[:, :, slice_idx]

        # 4. CT 數值預處理 (Windowing)
        img = self._preprocess_ct(img)

        # 5. Resize (使用 cv2 進行正確的插值)
        # 影像用線性插值，Mask 用最近鄰插值 (避免產生小數點標籤)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.out_size, self.out_size), interpolation=cv2.INTER_NEAREST)

        # 6. 轉為 Tensor 並處理通道
        img = torch.tensor(img).float().unsqueeze(0) 
        img = img.repeat(3, 1, 1) # [3, 1024, 1024]

        # === 修改這裡 ===
        mask = torch.tensor(mask).unsqueeze(0).float() # [1, 256, 256]

        # 7. 生成 Prompt (Click)
        pt = None
        point_label = None
        
        if self.prompt == 'click':
            # 注意：這裡要轉回 numpy 且去掉 channel 維度才能給 random_click 用
            binary_mask = (mask[0] > 0).int().numpy()
            # 修正上一輪的參數命名問題
            point_label, pt = random_click(binary_mask, 1)

        image_meta_dict = {'filename_or_obj': f"{item['image']}_slice{slice_idx}"}
        
        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
        }