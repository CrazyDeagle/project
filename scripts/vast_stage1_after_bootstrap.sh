#!/usr/bin/env bash
set -euo pipefail

cd "${SILEX_DIR:-/workspace/silexcode}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BOOTSTRAP_RUN="${1:-bootstrap_b0_b4}"
RUN_NAME="${2:-stage1_after_bootstrap}"
BOOTSTRAP_CKPT="runs/${BOOTSTRAP_RUN}/bootstrap_latest.plastic.silex"
OUT_DIR="runs/${RUN_NAME}"
LOG_PATH="${OUT_DIR}.log"

if [ ! -f "${BOOTSTRAP_CKPT}" ]; then
  echo "missing bootstrap checkpoint: ${BOOTSTRAP_CKPT}" >&2
  exit 1
fi

mkdir -p runs
rm -rf "${OUT_DIR}"

nohup setsid /venv/main/bin/python -u run_accelerated_curriculum.py \
  --resume "${BOOTSTRAP_CKPT}" \
  --output-dir "${OUT_DIR}" \
  ${ENABLE_OUTPUT_ADAPTER:+--enable-output-adapter} \
  --output-adapter-rank "${OUTPUT_ADAPTER_RANK:-64}" \
  ${OUTPUT_ADAPTER_ONLY:+--output-adapter-only} \
  --output-adapter-lr "${OUTPUT_ADAPTER_LR:-0.001}" \
  --stages 1 \
  --max-updates "${MAX_UPDATES:-1000}" \
  --eval-every "${EVAL_EVERY:-100}" \
  --val-size "${VAL_SIZE:-16}" \
  --max-records-per-chunk "${MAX_RECORDS_PER_CHUNK:-8}" \
  --candidate-multiplier "${CANDIDATE_MULTIPLIER:-4}" \
  --packing "${PACKING:-shortest}" \
  --kfac-warmup-updates "${KFAC_WARMUP_UPDATES:-100}" \
  --eta-scale "${ETA_SCALE:-0.1}" \
  --damping-scale "${DAMPING_SCALE:-10}" \
  --trust-scale "${TRUST_SCALE:-0.03}" \
  --native-optimizer "${NATIVE_OPTIMIZER:-kfac}" \
  > "${LOG_PATH}" 2>&1 < /dev/null &

echo "started_pid=$!"
echo "resume=${BOOTSTRAP_CKPT}"
echo "metrics=${OUT_DIR}/accelerated_metrics.jsonl"
echo "log=${LOG_PATH}"
