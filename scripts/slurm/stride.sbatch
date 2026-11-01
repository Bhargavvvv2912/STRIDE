#!/bin/bash
#
# One STRIDE arm on one subject.
#
#   sbatch scripts/slurm/stride.sbatch concept-transformer stride --first torch
#   sbatch scripts/slurm/stride.sbatch concept-transformer pin_latest
#
# Everything after the arm is passed to `python -m stride.cli run`.
# Reuses AURA's cluster layout (scripts/slurm/cluster_env.sh): uv Pythons,
# Ollama and its models under $AURA_SCRATCH.
#
#SBATCH --job-name=stride
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=16:00:00
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/gpu     : /'

# Environments and checkouts on the node's local disk (NFS stalls on pip).
LOCAL_DISK="${TMPDIR:-/tmp}"
export AURA_VENV_ROOT="$LOCAL_DISK/stride-venvs-$SLURM_JOB_ID"
export STRIDE_WORK="$LOCAL_DISK/stride-work-$SLURM_JOB_ID"
export PIP_CACHE="$LOCAL_DISK/aura-pip-cache-$USER"
mkdir -p "$AURA_VENV_ROOT" "$STRIDE_WORK" "$PIP_CACHE"

# Ollama on a port of this job's own. The code agent sends long transcripts,
# so the context window is 32k (Ollama's default would cut the front off).
export OLLAMA_HOST="127.0.0.1:$((11500 + SLURM_JOB_ID % 5000))"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}"
"$OLLAMA_BIN" serve > "slurm_logs/ollama_stride_$SLURM_JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true; rm -rf "${AURA_VENV_ROOT:?}" "${STRIDE_WORK:?}"' EXIT
for _ in $(seq 1 60); do
  curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null && break
  sleep 2
done
curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null || { echo "ollama did not start"; exit 1; }

export AURA_MODELS="${AURA_MODELS:-qwen14b}"
export MSWEA_COST_TRACKING=ignore_errors
./.devenv/bin/python -m stride.cli run "$SUBJECT" --arm "$ARM" \
  --work "$STRIDE_WORK" --repo-cache "$REPO_CACHE" --pip-cache "$PIP_CACHE" "$@"
