#!/bin/bash
#
# QLoRA fine-tune of Qwen2.5-Coder-14B on AURA's own LLM-tier logs.
# Builds the dataset from results/ on this cluster, then trains on one GPU.
#
#   sbatch scripts/slurm/finetune.sbatch
#   EXTRA="--epochs 3" sbatch scripts/slurm/finetune.sbatch
#
#SBATCH --job-name=aura-ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/finetune_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the AURA directory}"
source scripts/slurm/cluster_env.sh
export HF_HOME="$AURA_SCRATCH/hf"
OUT="$AURA_SCRATCH/ft/qwen14b-aura"
mkdir -p "$OUT" data/ft

echo "== dataset $(date +%H:%M)"
.devenv/bin/python scripts/build_ft_dataset.py --out data/ft
cp data/ft/sft.jsonl "$OUT/sft.jsonl"     # the exact training data, kept with the model

echo "== train $(date +%H:%M)"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$AURA_SCRATCH/ft-venv/bin/python" scripts/ft/train_qlora.py \
    --data data/ft/sft.jsonl --out "$OUT" ${EXTRA:-}
echo "== done $(date +%H:%M)"
