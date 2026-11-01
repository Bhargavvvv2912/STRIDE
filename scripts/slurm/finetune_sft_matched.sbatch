#!/bin/bash
#
# Matched SFT ablation for the KTO model: the same base model, QLoRA settings,
# held-out subjects and accepted examples (KTO's own frozen kto.jsonl, good
# rows only), one epoch -- differing from KTO only in leaving out the failed
# examples.
#
#   sbatch scripts/slurm/finetune_sft_matched.sbatch
#   then: NAME=qwen14b-aura-sft sbatch scripts/slurm/export_ollama.sbatch
#
#SBATCH --job-name=aura-sft-m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/finetune_sft_matched_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the AURA directory}"
source scripts/slurm/cluster_env.sh
export HF_HOME="$AURA_SCRATCH/hf"
KTO="$AURA_SCRATCH/ft/qwen14b-aura-kto/kto.jsonl"
OUT="$AURA_SCRATCH/ft/qwen14b-aura-sft"
[ -s "$KTO" ] || { echo "missing $KTO (the KTO run's frozen data)"; exit 1; }
mkdir -p "$OUT"
cp "$KTO" "$OUT/kto.jsonl"
echo "== train $(date +%H:%M)"
"$AURA_SCRATCH/ft-venv/bin/python" scripts/ft/train_qlora.py \
    --data "$OUT/kto.jsonl" --out "$OUT" --flask-cap 0 --epochs 1 \
    --val-subjects rhizonet authors-of-leisure age-of-empires-api
echo "== done $(date +%H:%M)"
