#!/usr/bin/env bash
set -euo pipefail

pkill -f run_accelerated_curriculum.py || true
pkill -f run_bootstrap.py || true
echo "vast_training_stopped"
