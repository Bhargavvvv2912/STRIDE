#!/bin/bash
#
# Offline replay: held-out LLM-tier prompts, base 14B vs each fine-tuned 14B.
#
#   sbatch scripts/slurm/replay_eval.sbatch
#   N=500 sbatch scripts/slurm/replay_eval.sbatch
#   MODE=three sbatch scripts/slurm/replay_eval.sbatch    # base vs matched SFT vs KTO
#
#SBATCH --job-name=aura-replay-ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/replay_eval_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the AURA directory}"
source scripts/slurm/cluster_env.sh
[ -s data/ft/all.jsonl ] || .devenv/bin/python scripts/build_ft_dataset.py --out data/ft
export OLLAMA_HOST="127.0.0.1:$((11500 + SLURM_JOB_ID % 5000))"
export OLLAMA_CONTEXT_LENGTH=8192
"$OLLAMA_BIN" serve > "slurm_logs/ollama_replay_${SLURM_JOB_ID}.log" 2>&1 &
trap 'kill $! 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null && break; sleep 2; done
if [ "${MODE:-}" = "three" ]; then
  # Base vs matched SFT vs KTO: both fine-tuned models held out these 3 subjects.
  .devenv/bin/python scripts/ft/replay_eval.py \
    --models qwen2.5-coder:14b qwen14b-aura-sft:latest qwen14b-aura-kto:latest \
    --subjects rhizonet authors-of-leisure age-of-empires-api \
    --n "${N:-300}" --out results/ft_replay_3way.json
  exit 0
fi
# Each fine-tuned model is compared with the base on subjects IT never trained on:
# the two runs held out different subjects, so no prompt set is unseen by both.
.devenv/bin/python scripts/ft/replay_eval.py \
  --models qwen2.5-coder:14b qwen14b-aura-kto:latest \
  --subjects rhizonet authors-of-leisure age-of-empires-api \
  --n "${N:-300}" --out results/ft_replay_kto.json
.devenv/bin/python scripts/ft/replay_eval.py \
  --models qwen2.5-coder:14b qwen14b-aura:latest \
  --subjects flask-basic-registration flask-dj sayhello wenet wgadmin \
  --n "${N:-300}" --out results/ft_replay_sft.json
