#!/usr/bin/env bash
#
# One-time setup on the cluster's LOGIN node (it needs internet access).
#
#   export AURA_SCRATCH=/path/on/scratch/or/turbo/aura
#   ./scripts/slurm/setup_cluster.sh
#
# Installs, without root: uv and the Pythons the subjects need, the harness
# virtualenv, Ollama, and the two models. Safe to re-run; finished steps are
# skipped.

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/slurm/cluster_env.sh

step() { printf '\n==> %s\n' "$*"; }

step "uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

step "Pythons (3.7-3.13 for subjects, 3.11 for the harness)"
# 3.8 as well: the pins this work is about predate 3.9, and a subject that
# cannot install on the version it was written for is rejected for the
# interpreter rather than for anything about its dependencies.
# The full range a subject may declare. A project that needs 3.12 is as valid
# a subject as one pinned to 3.9, and screening now tries the newest first.
uv python install 3.8 3.9 3.10 3.11 3.12 3.13
# 3.7 for projects from before 3.8 existed (torch 1.4 has Linux wheels only up
# to 3.7). Current uv no longer offers a 3.7 download; uv 0.2.37 still knows
# the python-build-standalone 3.7.9 build, so a copy of that uv installs it
# into the same directory, under the name the harness looks for.
if ! ls "$UV_PYTHON_INSTALL_DIR" | grep -q '^cpython-3\.7\.'; then
  OLD_UV="$(mktemp -d)/olduv"
  python3 -m venv "$OLD_UV" && "$OLD_UV/bin/pip" install -q "uv==0.2.37"
  UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" "$OLD_UV/bin/uv" python install 3.7.9
  rm -rf "$(dirname "$OLD_UV")"
fi

step "harness virtualenv (.devenv)"
if [ ! -x .devenv/bin/python ]; then
  "$(uv python find 3.11)" -m venv .devenv
fi
.devenv/bin/pip install -q --upgrade pip
.devenv/bin/pip install -q -r requirements-harness.txt

step "Ollama"
if [ ! -x "$OLLAMA_BIN" ]; then
  case "$(uname -m)" in
    x86_64)        arch=amd64 ;;
    aarch64|arm64) arch=arm64 ;;
    *) echo "unsupported architecture $(uname -m)"; exit 1 ;;
  esac
  mkdir -p "$AURA_SCRATCH/ollama"
  # Ollama ships Linux builds as .tar.zst (the old .tgz link now returns 404).
  url="https://ollama.com/download/ollama-linux-${arch}.tar.zst"
  if command -v zstd >/dev/null 2>&1; then
    curl -fL "$url" | zstd -dc | tar -x -C "$AURA_SCRATCH/ollama"
  else
    # No zstd tool on this machine: decompress with Python instead.
    .devenv/bin/pip install -q zstandard
    curl -fL "$url" | .devenv/bin/python -c \
      "import sys, zstandard; zstandard.ZstdDecompressor().copy_stream(sys.stdin.buffer, sys.stdout.buffer)" \
      | tar -x -C "$AURA_SCRATCH/ollama"
  fi
fi
"$OLLAMA_BIN" --version || true

step "models (downloaded once into $OLLAMA_MODELS)"
MODELS=$(.devenv/bin/python -c "from evaluation.arms import MODELS; print(' '.join(s.split(':',1)[1] for s in MODELS.values()))")
export OLLAMA_HOST="127.0.0.1:$((20000 + RANDOM % 10000))"
"$OLLAMA_BIN" serve > "$AURA_SCRATCH/ollama-setup.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null && break; sleep 1; done
for model in $MODELS; do "$OLLAMA_BIN" pull "$model"; done

step "check: the subjects' Pythons are found"
.devenv/bin/python - <<'PYEOF'
import json
from aura.environment import find_interpreter
for s in json.load(open("subjects.json")):
    v = s.get("python_version", "3.11")
    print(f"  {s['name']:<28} Python {v}: {find_interpreter(v) or 'NOT FOUND'}")
PYEOF

step "check: fast test suite"
.devenv/bin/python -m pytest tests/ -q -m "not slow" -p no:cacheprovider | tail -1

step "free the installers' download caches (home is capped at 50 GB)"
uv cache clean >/dev/null 2>&1 || true
.devenv/bin/pip cache purge >/dev/null 2>&1 || true
rm -rf "$HOME/.cache/pip"

step "disk use (home is capped at 50 GB)"
du -sh "$OLLAMA_MODELS" "$AURA_SCRATCH/ollama" "$UV_PYTHON_INSTALL_DIR" .devenv 2>/dev/null
du -sh "$HOME" 2>/dev/null | sed 's/^/total home: /'

cat <<'MSG'

Setup done. Before submitting, check that COMPUTE nodes can reach PyPI:
  srun --time=5 --pty curl -sI https://pypi.org/simple/ | head -1
Then:
  ./scripts/slurm/submit.sh
MSG
