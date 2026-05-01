#!/usr/bin/env bash
set -euo pipefail

cd "${SILEX_DIR:-/workspace/silexcode}"

RUN_NAME="${1:-stage1_probe_next}"
METRICS="runs/${RUN_NAME}/accelerated_metrics.jsonl"
BOOTSTRAP_METRICS="runs/${RUN_NAME}/bootstrap_metrics.jsonl"
LOG_PATH="runs/${RUN_NAME}.log"

echo "--- gpu ---"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,power.draw --format=csv,noheader

echo "--- process ---"
ps -ef | grep -E 'run_accelerated_curriculum.py|run_bootstrap.py' | grep -v grep || true

echo "--- metrics tail ---"
if [ -f "${METRICS}" ]; then
  tail -n "${TAIL_LINES:-20}" "${METRICS}" 2>/dev/null || true
elif [ -f "${BOOTSTRAP_METRICS}" ]; then
  tail -n "${TAIL_LINES:-20}" "${BOOTSTRAP_METRICS}" 2>/dev/null || true
fi

echo "--- analysis ---"
if [ -f "${METRICS}" ]; then
  /venv/main/bin/python analyze_curriculum_metrics.py "${METRICS}" || true
elif [ -f "${BOOTSTRAP_METRICS}" ]; then
  /venv/main/bin/python analyze_bootstrap_metrics.py "${BOOTSTRAP_METRICS}" || true
fi

echo "--- log tail ---"
tail -n "${TAIL_LINES:-20}" "${LOG_PATH}" 2>/dev/null || true
