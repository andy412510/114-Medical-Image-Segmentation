#!/bin/bash

# 1. 激活 Conda 環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate med

# 2. 設定 GPU
export CUDA_VISIBLE_DEVICES=0

# 3. 強制修正 OverflowError 的關鍵環境變數
export MONAI_VAR_MAX_SEED=2147483647

# 4. 執行訓練指令
# 注意：我移除了 -seed 2026 參數，改由環境變數控制，看看是否能避開 Compose 初始化的衝突
python train.py \
    -net sam \
    -mod sam_adpt \
    -exp_name msa-3d-sam-btcv \
    -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
    -image_size 512 \
    -b 1 \
    -dataset decathlon \
    -thd True \
    -chunk 12 \
    -data_path ../data \
    -num_sample 4 \
    -lr 0.0001

echo "訓練任務已結束"