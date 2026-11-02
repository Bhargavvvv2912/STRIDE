#!/bin/bash
# Install the command-line coding agents STRIDE compares, at fixed versions,
# under $AURA_SCRATCH (no root needed). Run once on the login node:
#
#   source scripts/slurm/cluster_env.sh && ./scripts/install_agents.sh
#
# mini-SWE-agent is a Python package (requirements-stride.txt); these two are
# Node programs, so a private Node.js is installed first.
set -euo pipefail
: "${AURA_SCRATCH:?source scripts/slurm/cluster_env.sh first}"

NODE_VERSION=22.12.0
CODEX_VERSION=0.166.0
OPENCODE_VERSION=1.19.5

NODE_DIR="$AURA_SCRATCH/node"
AGENTS_DIR="$AURA_SCRATCH/agents"

if [ ! -x "$NODE_DIR/bin/node" ]; then
  echo "installing node $NODE_VERSION ..."
  mkdir -p "$NODE_DIR"
  curl -fsSL "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz" \
    | tar -xJ -C "$NODE_DIR" --strip-components=1
fi
export PATH="$NODE_DIR/bin:$PATH"

echo "installing codex $CODEX_VERSION and opencode $OPENCODE_VERSION ..."
npm install --silent --global --prefix "$AGENTS_DIR" \
  "@openai/codex@$CODEX_VERSION" "opencode-ai@$OPENCODE_VERSION"

"$AGENTS_DIR/bin/codex" --version
"$AGENTS_DIR/bin/opencode" --version
echo "done: $AGENTS_DIR/bin (stride.sbatch puts it on PATH)"
