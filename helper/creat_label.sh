#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export VECL_PROJECT_ROOT=${VECL_PROJECT_ROOT:-$REPO_ROOT}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export WANDB_MODE=${WANDB_MODE:-disabled}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python "$REPO_ROOT/helper/LLM_prompt.py" \
  --model_name "${MODEL_NAME:-Meta-Llama-3.1-70B-Instruct-AWQ-INT4}" \
  --input_path "${INPUT_PATH:-$REPO_ROOT/Dataset/MIMIC/report_sentences.csv}" \
  --output_dir "${OUTPUT_DIR:-$REPO_ROOT/Dataset/MIMIC/llm_labels}"
