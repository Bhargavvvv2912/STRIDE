#!/bin/bash
#
# One STRIDE arm on one subject.
#
#   sbatch scripts/slurm/stride.sbatch concept-transformer stride --first torch
#   sbatch scripts/slurm/stride.sbatch concept-transformer pin_latest
#
# Everything after the arm is passed to `python -m stride.cli run`.
# The code agent's model comes from STRIDE_MODEL (default qwen2.5-coder:14b):
#
#   STRIDE_MODEL=qwen2.5-coder:32b sbatch scripts/slurm/stride.sbatch concept-transformer stride
#   STRIDE_MODEL=qwen3-coder:30b sbatch scripts/slurm/stride.sbatch concept-transformer stride --agent codex
#
# Agents: mini-swe-agent (default), codex, opencode (install the last two with
# scripts/install_agents.sh).
#
# AURA's own version-search LLM stays at qwen14b (AURA_MODELS) in every run,
# so only the code agent differs between them.
# Reuses AURA's cluster layout (scripts/slurm/cluster_env.sh): uv Pythons,
# Ollama and its models under $AURA_SCRATCH.
#
#SBATCH --job-name=stride
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/stride_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the STRIDE directory}"
mkdir -p slurm_logs
source scripts/slurm/cluster_env.sh

SUBJECT=${1:?subject}
ARM=${2:-stride}
shift 2 || true

for var in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
           NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS; do
  export "$var=1"
done

echo "subject : $SUBJECT   arm: $ARM   extra: $*"
echo "node    : $(hostname)   cpus: ${SLURM_CPUS_PER_TASK:-?}"
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)
echo "gpu     : $GPU"
# Without a GPU, Ollama falls back to the CPU and every job wastes ~25 minutes
# failing to load the model. Stop at once instead; the queue moves on.
if ! grep -q "MiB" <<<"$GPU"; then
  echo "NO GPU VISIBLE on $(hostname): stopping. Check with: srun --gres=gpu:1 nvidia-smi"
  exit 3
fi

# Environments and checkouts on the node's local disk (NFS stalls on pip).
LOCAL_DISK="${TMPDIR:-/tmp}"
export AURA_VENV_ROOT="$LOCAL_DISK/stride-venvs-$SLURM_JOB_ID"
export STRIDE_WORK="$LOCAL_DISK/stride-work-$SLURM_JOB_ID"
export PIP_CACHE="$LOCAL_DISK/aura-pip-cache-$USER"
mkdir -p "$AURA_VENV_ROOT" "$STRIDE_WORK" "$PIP_CACHE"

# Ollama on a port of this job's own. The code agent sends long transcripts,
# so the context window is 64k: at 32k the transcript of one concept-transformer
# attempt reached ~31k tokens, and past the window Ollama drops the front (the task).
export OLLAMA_HOST="127.0.0.1:$((11500 + SLURM_JOB_ID % 5000))"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-65536}"
export OLLAMA_LOAD_TIMEOUT="${OLLAMA_LOAD_TIMEOUT:-20m}"   # server settings: before `serve`
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
"$OLLAMA_BIN" serve > "slurm_logs/ollama_stride_$SLURM_JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true; rm -rf "${AURA_VENV_ROOT:?}" "${STRIDE_WORK:?}"' EXIT
for _ in $(seq 1 60); do
  curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null && break
  sleep 2
done
curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null || { echo "ollama did not start"; exit 1; }

# Load the model before STRIDE starts. On a GPU shared with other jobs the
# first load can take minutes; Ollama's default 5-minute load timeout then
# fails the agent's first call, which STRIDE would count as a failed repair.
STRIDE_MODEL="${STRIDE_MODEL:-qwen3-coder:30b}"
if ! "$OLLAMA_BIN" list 2>/dev/null | awk '{print $1}' | grep -qx "$STRIDE_MODEL"; then
  echo "pulling $STRIDE_MODEL into $OLLAMA_MODELS ..."
  "$OLLAMA_BIN" pull "$STRIDE_MODEL" || { echo "could not pull $STRIDE_MODEL"; exit 1; }
fi
echo "loading $STRIDE_MODEL (context $OLLAMA_CONTEXT_LENGTH) ..."
curl -s --max-time 1500 "http://$OLLAMA_HOST/api/generate" \
  -d "{\"model\": \"$STRIDE_MODEL\", \"prompt\": \"ok\", \"stream\": false, \"keep_alive\": \"24h\"}" \
  | grep -q '"done":true' || { echo "model did not load; see slurm_logs/ollama_stride_$SLURM_JOB_ID.log"; exit 1; }
echo "model loaded"

export AURA_MODELS="${AURA_MODELS:-qwen14b}"
export MSWEA_COST_TRACKING=ignore_errors
./.devenv/bin/python -m stride.cli run "$SUBJECT" --arm "$ARM" \
  --model "ollama_chat/$STRIDE_MODEL" --work "$STRIDE_WORK" --repo-cache "$REPO_CACHE" --pip-cache "$PIP_CACHE" "$@"
