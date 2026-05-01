#!/usr/bin/env bash
set -euo pipefail

cd "${SILEX_DIR:-/workspace/silexcode}"

RUN_NAME="${1:-stage1_probe_next}"
OUT_DIR="runs/${RUN_NAME}"
LOG_PATH="${OUT_DIR}.log"

mkdir -p runs
rm -rf "${OUT_DIR}"

nohup setsid /venv/main/bin/python -u run_accelerated_curriculum.py \
  --output-dir "${OUT_DIR}" \
  --stages 1 \
  --max-updates "${MAX_UPDATES:-1000}" \
  --eval-every "${EVAL_EVERY:-100}" \
  --val-size "${VAL_SIZE:-16}" \
  --max-records-per-chunk "${MAX_RECORDS_PER_CHUNK:-8}" \
  --candidate-multiplier "${CANDIDATE_MULTIPLIER:-4}" \
  --packing "${PACKING:-shortest}" \
  --kfac-warmup-updates "${KFAC_WARMUP_UPDATES:-25}" \
  --eta-scale "${ETA_SCALE:-0.5}" \
  --damping-scale "${DAMPING_SCALE:-3.0}" \
  --trust-scale "${TRUST_SCALE:-0.3}" \
  --checkpoint-every-evals "${CHECKPOINT_EVERY_EVALS:-0}" \
  > "${LOG_PATH}" 2>&1 < /dev/null &

echo "started_pid=$!"
echo "metrics=${OUT_DIR}/accelerated_metrics.jsonl"
echo "log=${LOG_PATH}"
