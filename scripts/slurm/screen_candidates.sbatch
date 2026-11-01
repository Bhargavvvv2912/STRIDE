#!/bin/bash
#
# Screen heavy candidates on the cluster, where the laptop cannot: torch and
# TensorFlow wheels are too large to install under emulation. Each array task
# takes every Nth candidate of the registry and runs the two arms that are a
# screen -- baseline (does the project work at all?) and pip_upgrade (does the
# naive upgrade break it?) -- one candidate after another. No LLM is involved.
#
#   sbatch --array=0-3 scripts/slurm/screen_candidates.sbatch
#
# Reads REGISTRY (default screening/cluster_candidates.json). Then, on the
# laptop after pulling:  python scripts/screening_report.py
#
#SBATCH --job-name=aura-screen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/screen_%A_%a.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the AURA directory}"
source scripts/slurm/cluster_env.sh
for var in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
           NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS; do
  export "$var=1"
done

export REGISTRY="${REGISTRY:-screening/cluster_candidates.json}"
export SCREENING=1   # a candidate list, not the sample: skip the sample-mix audit
TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
INDEX=${SLURM_ARRAY_TASK_ID:-0}

# Node-local disk for everything thrown away, as in aura_array.sbatch.
LOCAL_DISK="${TMPDIR:-/tmp}"
export AURA_VENV_ROOT="$LOCAL_DISK/aura-venvs-${SLURM_ARRAY_JOB_ID:-$$}-${INDEX}"
export WORKSPACES="$LOCAL_DISK/aura-work-${SLURM_ARRAY_JOB_ID:-$$}-${INDEX}"
export PIP_CACHE="$LOCAL_DISK/aura-pip-cache-$USER"
mkdir -p "$AURA_VENV_ROOT" "$WORKSPACES" "$PIP_CACHE"
trap 'rm -rf "${AURA_VENV_ROOT:?}" "${WORKSPACES:?}"' EXIT

NAMES=$(.devenv/bin/python -c "
import json, os
names = [s['name'] for s in json.load(open(os.environ['REGISTRY']))]
print(' '.join(names[$INDEX::$TASKS]))")
echo "task $INDEX of $TASKS on $(hostname): $NAMES"

for SUBJECT in $NAMES; do
  echo "===== $SUBJECT  $(date +%H:%M)"
  ./scripts/run_aura.sh "$SUBJECT" "baseline,pip_upgrade" || echo "  ($SUBJECT ended with an error; see its log)"
  rm -rf "${WORKSPACES:?}/$SUBJECT"
done
echo "done $(date +%H:%M)"
