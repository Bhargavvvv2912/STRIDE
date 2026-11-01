#!/bin/bash
#
# Evaluate base vs fine-tuned Qwen 14B on the coupled-upgrade benchmark.
# Resumable: finished cases are kept in results/bench_eval.json.
#
#   sbatch scripts/slurm/bench_eval.sbatch
#   OUT=results/bench_eval_s1.json EXTRA="--temperature 0.3 --seed 1" sbatch scripts/slurm/bench_eval.sbatch
#   MODELS=qwen14b-aura:latest sbatch scripts/slurm/bench_eval.sbatch           # add the SFT model
#
#SBATCH --job-name=aura-bench-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/bench_eval_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the AURA directory}"
source scripts/slurm/cluster_env.sh
export PIP_CACHE_DIR="$PIP_CACHE"
export OLLAMA_HOST="127.0.0.1:$((11500 + SLURM_JOB_ID % 5000))"
export OLLAMA_CONTEXT_LENGTH=8192
"$OLLAMA_BIN" serve > "slurm_logs/ollama_bench_${SLURM_JOB_ID}.log" 2>&1 &
trap 'kill $! 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null && break; sleep 2; done
.devenv/bin/python scripts/bench/eval_cases.py --cases data/bench/cases_*.jsonl \
  --models ${MODELS:-qwen2.5-coder:14b qwen14b-aura-kto:latest} --out "${OUT:-results/bench_eval.json}" ${EXTRA:-}
