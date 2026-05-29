#!/bin/bash

# 1. 激活 Conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate med

# 2. 設定 GPU
export CUDA_VISIBLE_DEVICES=0

# 3. 強制修正 OverflowError 的關鍵環境變數
export MONAI_VAR_MAX_SEED=2147483647

# 4. 執行訓練指令
python train.py \
    -net sam \
    -mod sam_adpt \
    -exp_name msa-btcv-rkid-sagittal \
    -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
    -image_size 1024 \
    -out_size 256 \
    -b 2 \
    -vis 5 \
    -dataset btcv \
    -data_path /home/user412771213/folder/data \
    -weights /home/user412771213/folder/114-Medical-Image-Segmentation/logs/msa-btcv-rkid-axial_2026_05_19_15_16_49/Model/best_dice_checkpoint.pth\
    -num_sample 4 \
    -axis sagittal \

echo "訓練任務已結束"