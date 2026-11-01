#!/usr/bin/env bash
# One-time setup for STRIDE, locally or on the cluster.
#   ./scripts/setup_stride.sh
# Needs Python 3.10+ (3.11 recommended). On the cluster, AURA's setup must have
# run once already (uv Pythons, Ollama and qwen2.5-coder:14b under $AURA_SCRATCH).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3.11}"
command -v "$PY" >/dev/null || PY=python3
"$PY" -m venv .devenv
./.devenv/bin/pip install -q --upgrade pip
./.devenv/bin/pip install -q -r requirements-stride.txt
./.devenv/bin/python -m pytest tests/test_stride_repair.py -q -p no:cacheprovider
./.devenv/bin/python -m stride.cli barriers concept-transformer
echo "STRIDE is ready."
