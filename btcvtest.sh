#!/bin/bash

# 1. 激活 Conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate med

# 2. 先強制清理掉之前卡住或崩潰的進程，釋放顯存
pkill -9 python

# 3. 設定記憶體碎片優化環境變數
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# 4. 執行推論指令
python val.py \
    -net sam \
    -mod sam_adpt \
    -dataset btcv \
    -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
    -weights /home/user412771213/folder/114-Medical-Image-Segmentation/logs/msa-btcv-rkid-sagittal_2026_05_19_21_38_57/Model/best_dice_checkpoint.pth \
    -exp_name test_btcv_rkid_sagittal \
    -vis 1 \
    -image_size 1024 \
    -out_size 256 \
    -data_path /home/user412771213/folder/data \
    -axis sagittal
echo "推論結束"