#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate med

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

python infer.py \
    -net sam \
    -mod sam_adpt \
    -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
    -weights /home/user412771213/folder/114-Medical-Image-Segmentation/logs/msa-btcv_2026_04_13_16_15_57/Model/best_dice_checkpoint.pth \
    -exp_name infer_btcv \
    -image_size 1024 \
    -out_size 256 \
    -gpu_device 0 \
    -nii_path /home/user412771213/folder/data_nii/74221040027761_16096879/1Abdomen_Routine_Abdomen_201.nii.gz

echo "推論結束"