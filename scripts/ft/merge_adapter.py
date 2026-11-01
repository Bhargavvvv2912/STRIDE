"""Merge the trained LoRA adapter into the bf16 base model, for import into Ollama.

Uses the final adapter if the run saved one, else its latest checkpoint.

    python scripts/ft/merge_adapter.py --run $AURA_SCRATCH/ft/qwen14b-aura
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-14B-Instruct")
args = ap.parse_args()

run = Path(args.run)
adapter = run / "adapter"
if not (adapter / "adapter_config.json").exists():
    ckpts = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        raise SystemExit(f"no adapter or checkpoint under {run}")
    adapter = ckpts[-1]
print("adapter:", adapter)

base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map={"": 0})
merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
out = run / "merged"
merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
(out / "ADAPTER_USED").write_text(str(adapter) + "\n")
print("merged model:", out)
