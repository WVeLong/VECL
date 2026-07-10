#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
export VECL_PROJECT_ROOT=${VECL_PROJECT_ROOT:-$REPO_ROOT}

python "$SCRIPT_DIR/retrieval_post_process.py" \
  --test_data_path "${TEST_DATA_PATH:-$REPO_ROOT/Dataset/MIMIC/mimic-cxr-test.csv}" \
  --MIMIC_data_path "${MIMIC_DATA_PATH:-$REPO_ROOT/Dataset/MIMIC/mimic-cxr-label-LLM_report-xinhuo-chexpertformat.csv}" \
  --report_corpus_path "${REPORT_CORPUS_PATH:-$REPO_ROOT/Dataset/MIMIC/retrieval_report_corpus.pickle}" \
  --num_chunks "${NUM_CHUNKS:-4}"
