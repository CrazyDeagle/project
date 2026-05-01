#!/usr/bin/env bash
set -euo pipefail

cd "${SILEX_DIR:-/workspace/silexcode}"

RUN_NAME="${1:-bootstrap_b0_b4}"
OUT_DIR="runs/${RUN_NAME}"
LOG_PATH="${OUT_DIR}.log"

mkdir -p runs
rm -rf "${OUT_DIR}"

nohup setsid /venv/main/bin/python -u run_bootstrap.py \
  --output-dir "${OUT_DIR}" \
  --levels "${BOOTSTRAP_LEVELS:-0,1,2,3,4}" \
  --updates-per-level "${UPDATES_PER_LEVEL:-1000}" \
  --eval-every "${EVAL_EVERY:-100}" \
  --val-size "${VAL_SIZE:-16}" \
  --max-records-per-chunk "${MAX_RECORDS_PER_CHUNK:-16}" \
  --candidate-multiplier "${CANDIDATE_MULTIPLIER:-4}" \
  --eta "${ETA:-0.01}" \
  --damping "${DAMPING:-0.01}" \
  --trust-region-delta "${TRUST_REGION_DELTA:-0.00003}" \
  --kfac-warmup-updates "${KFAC_WARMUP_UPDATES:-100}" \
  --checkpoint-every-evals "${CHECKPOINT_EVERY_EVALS:-0}" \
  > "${LOG_PATH}" 2>&1 < /dev/null &

echo "started_pid=$!"
echo "metrics=${OUT_DIR}/bootstrap_metrics.jsonl"
echo "checkpoint=${OUT_DIR}/bootstrap_latest.plastic.silex"
echo "log=${LOG_PATH}"
