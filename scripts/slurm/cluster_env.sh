# Settings shared by setup, submission and every job. Source it; don't run it.
#
# Everything large lives under AURA_SCRATCH (scratch or Turbo storage), because
# cluster home directories are small: Python builds, Ollama and its models, the
# wheel cache, and the per-arm checkouts and virtualenvs.
#
#   export AURA_SCRATCH=/path/on/scratch/or/turbo/aura
#
# Defaults to ~/aura-scratch, which is where setup_cluster.sh puts everything
# when nothing else is set: a job submitted straight with sbatch inherits the
# submitting shell's environment, and that shell has often not exported this.
AURA_SCRATCH="${AURA_SCRATCH:-$HOME/aura-scratch}"
export AURA_SCRATCH
[ -d "$AURA_SCRATCH" ] || mkdir -p "$AURA_SCRATCH"

export UV_PYTHON_INSTALL_DIR="$AURA_SCRATCH/uv-python"   # read by find_interpreter
export OLLAMA_MODELS="$AURA_SCRATCH/ollama-models"
export OLLAMA_BIN="$AURA_SCRATCH/ollama/bin/ollama"
export PIP_CACHE="$AURA_SCRATCH/wheels"
export WORKSPACES="$AURA_SCRATCH/workspaces"
export REPO_CACHE="$AURA_SCRATCH/repo_cache"
export PATH="$AURA_SCRATCH/agents/bin:$AURA_SCRATCH/node/bin:$AURA_SCRATCH/ollama/bin:$HOME/.local/bin:$PATH"

mkdir -p "$UV_PYTHON_INSTALL_DIR" "$OLLAMA_MODELS" "$PIP_CACHE" "$WORKSPACES" "$REPO_CACHE"
