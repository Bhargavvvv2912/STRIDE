"""QLoRA fine-tune of Qwen2.5-Coder-14B-Instruct on AURA's LLM-tier SFT data.

Input is data/ft/sft.jsonl from scripts/build_ft_dataset.py: the exact compact
prompt AURA sent, and an accepted version plan as the reply. The loss is on the
reply only, so the model learns to answer AURA's prompt with a plan -- versions,
never code.

Flask is ~64% of the raw data; --flask-cap keeps at most that many Flask
examples (joint moves first) so the other families are not swamped.

    python scripts/ft/train_qlora.py --data data/ft/sft.jsonl --out $AURA_SCRATCH/ft/qwen14b-aura
"""
from __future__ import annotations

import argparse
import json
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen2.5-Coder-14B-Instruct"


def load(path: str, flask_cap: int, eval_frac: float, seed: int, val_subjects=None):
    """SFT examples, from sft.jsonl, or from KTO's kto.jsonl keeping only the good ones.

    With kto.jsonl and --val-subjects, the model trains on exactly the accepted
    examples the KTO model saw, with the same held-out subjects: the matched
    ablation, differing from KTO only in leaving the failed examples out.
    """
    rows = [json.loads(line) for line in open(path)]
    if rows and "label" in rows[0]:
        rows = [{"messages": r["prompt"] + r["completion"], "subject": r["subject"],
                 "family": r["family"]} for r in rows if r["label"]]
    rng = random.Random(seed)
    joint = lambda r: len(json.loads(r["messages"][1]["content"])["proposed_plan"]) >= 2
    flask = [r for r in rows if r["family"] == "flask"]
    other = [r for r in rows if r["family"] != "flask"]
    if flask_cap and len(flask) > flask_cap:
        rng.shuffle(flask)
        flask.sort(key=lambda r: not joint(r))     # stable: joint moves first
        flask = flask[:flask_cap]
    rows = flask + other
    rng.shuffle(rows)
    if val_subjects:
        eval_subj = set(val_subjects)
    else:
        # Hold out whole subjects for eval loss, so it measures generalisation
        subjects = sorted({r["subject"] for r in rows})
        rng.shuffle(subjects)
        eval_subj = set(subjects[:max(1, int(len(subjects) * eval_frac))])
    fmt = lambda r: {"prompt": [r["messages"][0]], "completion": [r["messages"][1]]}
    train = [fmt(r) for r in rows if r["subject"] not in eval_subj]
    evals = [fmt(r) for r in rows if r["subject"] in eval_subj]
    print(f"train {len(train)}  eval {len(evals)} ({len(eval_subj)} subjects held out: {sorted(eval_subj)})")
    return Dataset.from_list(train), Dataset.from_list(evals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ft/sft.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--flask-cap", type=int, default=600)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-subjects", nargs="*", default=None)
    args = ap.parse_args()

    train, evals = load(args.data, args.flask_cap, args.eval_frac, args.seed, args.val_subjects)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")

    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, gradient_checkpointing=True,
        lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
        max_length=args.max_len, completion_only_loss=True,
        # PEFT hides `labels` from the forward signature; without this the
        # Trainer computes no eval_loss at all.
        label_names=["labels"],
        logging_steps=10, eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_steps=50, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        report_to="none", seed=args.seed)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train,
                         eval_dataset=evals, processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(f"{args.out}/adapter")
    tok.save_pretrained(f"{args.out}/adapter")
    print("saved adapter to", f"{args.out}/adapter")


if __name__ == "__main__":
    main()
