#!/usr/bin/env bash
set -euo pipefail

cd "${SILEX_DIR:-/workspace/silexcode}"
git pull
/venv/main/bin/python -m pip install -e . --no-build-isolation
/venv/main/bin/python -m pytest tests/test_accelerated_curriculum.py -q
echo "vast_setup=PASS"
