#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
export VECL_PROJECT_ROOT=${VECL_PROJECT_ROOT:-$REPO_ROOT}
export VECL_FINAL_CKPT=${VECL_FINAL_CKPT:-$REPO_ROOT/pretrain_model/vecl_miccai_final.ckpt}
export VECL_BIOCLINICAL_BERT=${VECL_BIOCLINICAL_BERT:-Laihaoran/BioClinicalMPBERT}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export WANDB_MODE=${WANDB_MODE:-disabled}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python "$SCRIPT_DIR/retrieval_inference.py" \
  --tokenizer_path "$VECL_BIOCLINICAL_BERT" \
  --ckpt_path "$VECL_FINAL_CKPT" \
  --test_data_path "${TEST_DATA_PATH:-$REPO_ROOT/Dataset/MIMIC/mimic-cxr-test.csv}" \
  --MIMIC_data_path "${MIMIC_DATA_PATH:-$REPO_ROOT/Dataset/MIMIC/mimic-cxr-label-LLM_report-xinhuo-chexpertformat.csv}" \
  --report_corpus_path "${REPORT_CORPUS_PATH:-$REPO_ROOT/Dataset/MIMIC/mimic-cxr-label-LLM_report-xinhuo-chexpertformat.csv}" \
  --split_id "${SPLIT_ID:-1}" \
  --num_chunks "${NUM_CHUNKS:-4}" \
  --batch_size "${BATCH_SIZE:-256}"
