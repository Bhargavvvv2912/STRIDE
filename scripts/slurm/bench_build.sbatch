#!/bin/bash
#
# Build the coupled-upgrade benchmark (CPU only; pip's resolver, no GPU).
# Four tasks, each a quarter of the families, into data/bench/cases_<task>.jsonl.
# Resumable: resubmit and each task continues where it stopped.
#
#   sbatch --array=0-3 scripts/slurm/bench_build.sbatch
#   FAMILIES="transformers fastapi" OUTNAME=extra sbatch scripts/slurm/bench_build.sbatch
#
#SBATCH --job-name=aura-bench
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/bench_build_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the AURA directory}"
source scripts/slurm/cluster_env.sh
export PIP_CACHE_DIR="$PIP_CACHE"
FAMS=(flask torch django aws transformers tensorflow celery fastapi
      sqlalchemy aiohttp httpx langchain numpy aiobotocore marshmallow graphql)
T=${SLURM_ARRAY_TASK_ID:-0}; N=${SLURM_ARRAY_TASK_COUNT:-1}
MINE=(); for i in "${!FAMS[@]}"; do [ $((i % N)) -eq "$T" ] && MINE+=("${FAMS[$i]}"); done
if [ -n "${FAMILIES:-}" ]; then MINE=($FAMILIES); T="${OUTNAME:-extra}"; fi   # named families, own file
echo "task $T: ${MINE[*]}"
mkdir -p data/bench
.devenv/bin/python scripts/bench/build_cases.py --out "data/bench/cases_${T}.jsonl" \
  --families "${MINE[@]}" --per-family "${PER_FAMILY:-30}" --easy-cap "${EASY_CAP:-6}"
