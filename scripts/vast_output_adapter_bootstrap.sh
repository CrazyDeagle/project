#!/usr/bin/env bash
set -euo pipefail

cd "${SILEX_DIR:-/workspace/silexcode}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_NAME="${1:-bootstrap_output_adapter_b0}"
OUT_DIR="runs/${RUN_NAME}"
LOG_PATH="${OUT_DIR}.log"

mkdir -p runs
rm -rf "${OUT_DIR}"

nohup setsid /venv/main/bin/python -u run_bootstrap.py \
  --output-dir "${OUT_DIR}" \
  --enable-output-adapter \
  --output-adapter-only \
  --output-adapter-rank "${OUTPUT_ADAPTER_RANK:-64}" \
  --output-adapter-lr "${OUTPUT_ADAPTER_LR:-0.003}" \
  --levels "${BOOTSTRAP_LEVELS:-0}" \
  --updates-per-level "${UPDATES_PER_LEVEL:-500}" \
  --eval-every "${EVAL_EVERY:-25}" \
  --val-size "${VAL_SIZE:-8}" \
  --max-records-per-chunk "${MAX_RECORDS_PER_CHUNK:-16}" \
  --candidate-multiplier "${CANDIDATE_MULTIPLIER:-4}" \
  --checkpoint-every-evals "${CHECKPOINT_EVERY_EVALS:-1}" \
  > "${LOG_PATH}" 2>&1 < /dev/null &

echo "started_pid=$!"
echo "metrics=${OUT_DIR}/bootstrap_metrics.jsonl"
echo "checkpoint=${OUT_DIR}/bootstrap_latest.plastic.silex"
echo "log=${LOG_PATH}"
