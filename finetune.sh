#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export VECL_PROJECT_ROOT=${VECL_PROJECT_ROOT:-$REPO_ROOT}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export WANDB_MODE=${WANDB_MODE:-disabled}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python "$REPO_ROOT/run.py" \
  --config "${CONFIG:-$REPO_ROOT/configs/pretrain.yaml}" \
  --train \
  --train_pct "${TRAIN_PCT:-1.0}"
