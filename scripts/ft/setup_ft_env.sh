#!/bin/bash
# One-time: the fine-tuning environment, separate from AURA's harness venv.
# Blackwell GPUs (RTX PRO 6000) need a CUDA 12.8 build of PyTorch.
#
#   bash scripts/ft/setup_ft_env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/slurm/cluster_env.sh
FT="$AURA_SCRATCH/ft-venv"
uv venv -q -p 3.11 "$FT"
uv pip install -q -p "$FT/bin/python" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch>=2.7" "transformers>=4.51,<5" "peft>=0.15" "trl>=0.19" \
  "bitsandbytes>=0.46" "accelerate>=1.6" datasets sentencepiece
"$FT/bin/python" -c "import torch, transformers, peft, trl, bitsandbytes; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| trl', trl.__version__)"
echo "ft env ready: $FT"
