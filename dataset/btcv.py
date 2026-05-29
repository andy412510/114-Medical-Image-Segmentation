import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import cv2
import scipy.ndimage as ndimage
from utils import random_click


def resample_volume(vol, original_spacing, target_spacing=(1.0, 1.0, 1.0), order=1):
    """
    把 volume 重採樣到目標 spacing（預設 1mm isotropic）。
    vol: numpy array, shape (H, W, D)
    original_spacing: (sp_h, sp_w, sp_d)，單位 mm
    order: 1=bilinear（影像用），0=nearest（mask 用）
    """
    zoom_factors = [o / t for o, t in zip(original_spacing, target_spacing)]
    resampled = ndimage.zoom(vol, zoom_factors, order=order)
    return resampled


class BTCVInference(Dataset):
    def __init__(self, args, nii_path):
        self.img_size = args.image_size
        self.out_size = args.out_size
        self.nii_path = nii_path

        img_obj = nib.load(nii_path)
        img_vol = img_obj.get_fdata()
        self.img_vol = img_vol
        self.d = img_vol.shape[2]

        print(f"[Inference] Loaded {nii_path}, total slices: {self.d}")

    def __len__(self):
        return self.d

    def _preprocess_ct(self, img_slice):
        min_bound = -200.0
        max_bound = 400.0
        img_slice = np.clip(img_slice, min_bound, max_bound)
        img_slice = (img_slice - min_bound) / (max_bound - min_bound)
        return img_slice

    def __getitem__(self, index):
        img = self.img_vol[:, :, index]
        img = self._preprocess_ct(img)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.tensor(img).float().unsqueeze(0)
        img_tensor = img_tensor.repeat(3, 1, 1)
        mask = torch.zeros(1, self.out_size, self.out_size).float()
        scale = self.img_size / 512.0
        pt = np.array([int(161 * scale), int(206 * scale)])
        point_label = 1
        image_meta_dict = {'filename_or_obj': f"{os.path.basename(self.nii_path)}_slice{index:04d}"}
        return {
            'image': img_tensor,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
            'pixel_spacing': (1.0, 1.0),
        }


class BTCV(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click'):
        self.args = args
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.out_size = args.out_size
        self.target_labels = [2]  # 左腎
        self.axis = getattr(args, 'axis', 'axial')
        print(f"[{mode}] Slicing axis: {self.axis}")

        json_path = os.path.join(data_path, "dataset.json")
        with open(json_path, 'r') as f:
            data_info = json.load(f)

        key = 'training' if mode == 'Training' else 'validation'
        self.data_list = data_info[key]
        print(f"[{mode}] Loaded {len(self.data_list)} samples from {json_path}")

    def __len__(self):
        return len(self.data_list)

    def _preprocess_ct(self, img_slice):
        min_bound = -200.0
        max_bound = 400.0
        img_slice = np.clip(img_slice, min_bound, max_bound)
        img_slice = (img_slice - min_bound) / (max_bound - min_bound)
        return img_slice

    def _merge_label_mask(self, mask_vol):
        binary_mask = np.zeros_like(mask_vol, dtype=np.uint8)
        for label in self.target_labels:
            binary_mask[np.abs(mask_vol - label) < 0.5] = 1
        return binary_mask

    def _get_spacing(self, nib_obj):
        zooms = nib_obj.header.get_zooms()[:3]
        return tuple(float(z) for z in zooms)

    def _get_slice(self, vol, idx):
        if self.axis == 'axial':
            return vol[:, :, idx]
        elif self.axis == 'coronal':
            return vol[:, idx, :]
        elif self.axis == 'sagittal':
            return vol[idx, :, :]
        else:
            raise ValueError(f"Unknown axis: {self.axis}")

    def _get_num_slices(self, vol):
        if self.axis == 'axial':
            return vol.shape[2]
        elif self.axis == 'coronal':
            return vol.shape[1]
        elif self.axis == 'sagittal':
            return vol.shape[0]
        else:
            raise ValueError(f"Unknown axis: {self.axis}")

    def _get_foreground_slices(self, mask_vol):
        if self.axis == 'axial':
            return np.where(np.sum(mask_vol, axis=(0, 1)) > 0)[0]
        elif self.axis == 'coronal':
            return np.where(np.sum(mask_vol, axis=(0, 2)) > 0)[0]
        elif self.axis == 'sagittal':
            return np.where(np.sum(mask_vol, axis=(1, 2)) > 0)[0]
        else:
            raise ValueError(f"Unknown axis: {self.axis}")

    def __getitem__(self, index):
        item = self.data_list[index]

        img_rel  = item.get('image', item.get('img', '')).replace('./', '')
        lbl_rel  = item.get('label', item.get('seg', '')).replace('./', '')

        img_path  = os.path.join(self.data_path, img_rel)
        mask_path = os.path.join(self.data_path, lbl_rel)

        img_obj  = nib.load(img_path)
        mask_obj = nib.load(mask_path)

        img_vol  = img_obj.get_fdata()
        mask_vol = mask_obj.get_fdata()
        mask_vol = self._merge_label_mask(mask_vol)

        # ★ 取出原始 spacing（用於 mm² 換算）
        original_spacing = self._get_spacing(img_obj)  # (sp_h, sp_w, sp_d)

        # Resampling：axial 以外的方向才做，讓三個方向的解析度一致
        if self.axis != 'axial':
            print(f"[RESAMPLE] {img_rel.split('/')[-1]} original spacing: {original_spacing}")
            img_vol  = resample_volume(img_vol,  original_spacing, target_spacing=(1.0, 1.0, 1.0), order=1)
            mask_vol = resample_volume(mask_vol.astype(np.float32), original_spacing,
                                       target_spacing=(1.0, 1.0, 1.0), order=0)
            mask_vol = (mask_vol > 0.5).astype(np.uint8)
            print(f"[RESAMPLE] resampled shape: {img_vol.shape}")
            # resampling 後 spacing 變成 1mm isotropic
            pixel_spacing = (1.0, 1.0)
        else:
            pixel_spacing = (original_spacing[0], original_spacing[1])

        n_slices = self._get_num_slices(mask_vol)
        foreground_slices = self._get_foreground_slices(mask_vol)

        if self.mode == 'Training':
            if len(foreground_slices) > 0:
                if np.random.rand() < 0.8:
                    slice_idx = np.random.choice(foreground_slices)
                else:
                    slice_idx = np.random.randint(0, n_slices)
            else:
                slice_idx = np.random.randint(0, n_slices)
        else:
            # ★ Val 模式：找 GT 面積最大的切片
            if len(foreground_slices) > 0:
                max_area = -1
                slice_idx = foreground_slices[0]
                for s in foreground_slices:
                    area = self._get_slice(mask_vol, s).sum()
                    if area > max_area:
                        max_area = area
                        slice_idx = s
            else:
                slice_idx = n_slices // 2

        img  = self._get_slice(img_vol,  slice_idx)
        mask = self._get_slice(mask_vol, slice_idx)

        # ★ 原始切片尺寸（用於 mm² 換算）
        orig_slice_h, orig_slice_w = img.shape[:2]

        img  = self._preprocess_ct(img)
        img  = cv2.resize(img,  (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.out_size, self.out_size), interpolation=cv2.INTER_NEAREST)

        img  = torch.tensor(img).float().unsqueeze(0)
        img  = img.repeat(3, 1, 1)
        mask = torch.tensor(mask).unsqueeze(0).float()

        pt = None
        point_label = None
        if self.prompt == 'click':
            binary_mask = (mask[0] > 0).int().numpy()
            point_label, pt = random_click(binary_mask, 1)

        image_meta_dict = {
            'filename_or_obj': f"{img_rel}_{self.axis}_slice{slice_idx}"
        }

        # ★ mm²/pixel 換算：
        # out_size 的每個像素對應原始多少 mm
        # pixel_spacing 是原始影像的 mm/pixel
        # 縮放比例 = orig_slice_h / out_size
        scale_h = orig_slice_h / self.out_size
        scale_w = orig_slice_w / self.out_size
        mm2_per_pixel = pixel_spacing[0] * scale_h * pixel_spacing[1] * scale_w

        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': image_meta_dict,
            'pixel_spacing': pixel_spacing,
            'mm2_per_pixel': mm2_per_pixel,
        }