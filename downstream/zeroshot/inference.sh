#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
export VECL_PROJECT_ROOT=${VECL_PROJECT_ROOT:-$REPO_ROOT}
export VECL_FINAL_CKPT=${VECL_FINAL_CKPT:-$REPO_ROOT/pretrain_model/vecl_miccai_final.ckpt}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export WANDB_MODE=${WANDB_MODE:-disabled}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python "$SCRIPT_DIR/inference.py" \
  --experiment_name pretrain \
  --ckpt_path "$VECL_FINAL_CKPT" \
  --evaluation_method "${EVALUATION_METHOD:-PNC}" \
  --test_Dateset "${TEST_DATASETS:-0,1,2,3,4}" \
  --output_path "${OUTPUT_PATH:-$REPO_ROOT/data/output/zero_shot_${EVALUATION_METHOD:-PNC}.txt}"
