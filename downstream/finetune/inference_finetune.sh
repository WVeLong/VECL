#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
export VECL_PROJECT_ROOT=${VECL_PROJECT_ROOT:-$REPO_ROOT}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export WANDB_MODE=${WANDB_MODE:-disabled}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python "$SCRIPT_DIR/inference_finetune.py" \
  --ckpt_dir "${CKPT_DIR:-$REPO_ROOT/data/ckpt/VECL_finetune_classification}" \
  --evaluation_method "${EVALUATION_METHOD:-POS}" \
  --dataset_name "${DATASET_NAME:-ChestXray14}" \
  --train_pct "${TRAIN_PCT:-0.1}" \
  --cfg_path "${CFG_PATH:-$REPO_ROOT/configs/ChestXray14_classification_config.yaml}" \
  --experiment_name "${EXPERIMENT_NAME:-VECL}" \
  --output_path "${OUTPUT_PATH:-$REPO_ROOT/data/output/VECL_finetune_classification}"
