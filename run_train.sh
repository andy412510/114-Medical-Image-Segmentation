#!/bin/bash
# run_train.sh — 啟動 training 的 script (author: 412770132)

CONDA_HOME="$HOME/miniconda3"
ENV_NAME="sam_adapt_win"
PROJECT_DIR="$HOME/114-Medical-Image-Segmentation"
SAM_CKPT="./checkpoint/sam/efficient_sam_vits.pt"
DATASET="BTCV"
DATA_PATH="/srv/Datasets/BTCV"
IMAGE_SIZE=512
BATCH_SIZE=1
EXP_NAME="msa_test_efficient"

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${PROJECT_DIR}" || exit 1
mkdir -p logs
python train.py -net efficient_sam -mod sam_adpt -exp_name "${EXP_NAME}" \
  -sam_ckpt "${SAM_CKPT}" -dataset "${DATASET}" -data_path "${DATA_PATH}" \
  -image_size ${IMAGE_SIZE} -b ${BATCH_SIZE} -val_freq 3 -vis 5 \
  > "logs/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log" 2>&1
