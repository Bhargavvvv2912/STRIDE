#!/bin/bash
#
# Behaviour checks on a subject's accepted patches (no model, no GPU):
# diff coverage, and the subject's differential probe if it has one.
#
#   sbatch scripts/slurm/stride_check.sbatch concept-transformer
#   sbatch scripts/slurm/stride_check.sbatch concept-transformer --results results_v1
#
# Writes results/<subject>/<run>/behaviour.json and
# results/<subject>/behaviour_summary.json.
#
#SBATCH --job-name=stride-check
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/stride_check_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the STRIDE directory}"
mkdir -p slurm_logs
source scripts/slurm/cluster_env.sh

SUBJECT=${1:?subject}
shift || true
LOCAL_DISK="${TMPDIR:-/tmp}"
export AURA_VENV_ROOT="$LOCAL_DISK/stride-check-venvs-$SLURM_JOB_ID"
WORK="$LOCAL_DISK/stride-check-work-$SLURM_JOB_ID"
PIP_CACHE="$LOCAL_DISK/aura-pip-cache-$USER"
mkdir -p "$AURA_VENV_ROOT" "$WORK" "$PIP_CACHE"
trap 'rm -rf "${AURA_VENV_ROOT:?}" "${WORK:?}"' EXIT

./.devenv/bin/python -m stride.cli check "$SUBJECT" --work "$WORK" \
  --repo-cache "$REPO_CACHE" --pip-cache "$PIP_CACHE" "$@"
