#!/bin/bash
#
# QLoRA + KTO fine-tune of Qwen2.5-Coder-14B on AURA's LLM-tier logs, good and
# bad cases, from the base model. Trains on every subject but three non-LLM-win
# validation subjects; the test set is new subjects run end to end. Builds the dataset from results/ here, then
# trains on one GPU. Checkpoints every 100 steps: if the 12-hour limit stops
# it, submit the same command again and it resumes.
#
#   sbatch scripts/slurm/finetune_kto.sbatch
#   EXTRA="--bad-ratio 3" sbatch scripts/slurm/finetune_kto.sbatch    # subsample bad cases
#
#SBATCH --job-name=aura-kto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/finetune_kto_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the AURA directory}"
source scripts/slurm/cluster_env.sh
export HF_HOME="$AURA_SCRATCH/hf"
OUT="$AURA_SCRATCH/ft/qwen14b-aura-kto"
mkdir -p "$OUT" data/ft

if [ ! -s "$OUT/kto.jsonl" ]; then
  echo "== dataset $(date +%H:%M)"
  .devenv/bin/python scripts/build_ft_dataset.py --out data/ft
  cp data/ft/kto.jsonl "$OUT/kto.jsonl"   # frozen with the model; a resumed run reuses it
fi

echo "== train $(date +%H:%M)"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$AURA_SCRATCH/ft-venv/bin/python" scripts/ft/train_kto.py \
    --data "$OUT/kto.jsonl" --out "$OUT" \
    --val-subjects rhizonet authors-of-leisure age-of-empires-api ${EXTRA:-}
echo "== done $(date +%H:%M)"
