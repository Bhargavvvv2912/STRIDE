"""QLoRA + KTO fine-tune of Qwen2.5-Coder-14B-Instruct on AURA's LLM-tier logs.

One stage, from the base model, on good AND bad cases: data/ft/kto.jsonl from
scripts/build_ft_dataset.py holds every distinct (prompt, reply) AURA saw,
labelled good when AURA accepted the plan and bad otherwise (wrong Python,
resolution conflict, missing version, tests broke, or a reply the planner had to
reject). KTO needs no pairs: it raises the likelihood of good replies and lowers
that of bad ones relative to the base model, so the model learns both what to
propose and what not to. Replies are version plans only -- never code.

Good examples are ~12% of the data; desirable_weight rebalances them as the KTO
paper recommends, so that desirable_weight * n_good ~= n_bad.

The run checkpoints every --save-steps and resumes from its latest checkpoint,
so a job stopped by the 12-hour limit continues when resubmitted.

    python scripts/ft/train_kto.py --data data/ft/kto.jsonl --out $AURA_SCRATCH/ft/qwen14b-aura-kto
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import KTOConfig, KTOTrainer

BASE = "Qwen/Qwen2.5-Coder-14B-Instruct"


def split_subjects(rows, val_frac: float, test_frac: float, seed: int):
    """Assign whole subjects to train / val / test.

    Splitting by example would put near-copies of test prompts (one subject's
    retries of the same conflict) into training. Subjects are shuffled, then
    each goes to the split furthest below its share of the *good* examples, so
    validation and test each hold ~their share of good cases as well as bad.
    """
    rng = random.Random(seed)
    good = {}
    for r in rows:
        good[r["subject"]] = good.get(r["subject"], 0) + r["label"]
    subjects = sorted(good)
    rng.shuffle(subjects)
    total_good = sum(good.values()) or 1
    want = {"val": val_frac, "test": test_frac}
    have = {"val": 0, "test": 0}
    n_sub = {k: max(1, round(len(subjects) * f)) for k, f in want.items()}
    split = {}
    for s in subjects:
        open_ = [k for k in want if sum(1 for v in split.values() if v == k) < n_sub[k]]
        if open_ and (good[s] or rng.random() < 0.5):
            k = min(open_, key=lambda k: have[k] / total_good - want[k])
            if have[k] / total_good < want[k] * 1.5:
                split[s] = k
                have[k] += good[s]
                continue
        split[s] = "train"
    return split


def load(path: str, val_frac: float, test_frac: float, seed: int, bad_ratio: float, out: str,
         val_subjects: list[str] | None = None):
    rows = [json.loads(line) for line in open(path)]
    rng = random.Random(seed)
    if val_subjects:
        # Named validation subjects, no test split: the test set is new
        # subjects run end to end (and their prompts replayed offline).
        missing = set(val_subjects) - {r["subject"] for r in rows}
        if missing:
            raise SystemExit(f"validation subjects not in the data: {sorted(missing)}")
        split = {r["subject"]: ("val" if r["subject"] in val_subjects else "train") for r in rows}
    else:
        split = split_subjects(rows, val_frac, test_frac, seed)
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "split.json").write_text(json.dumps(
        {k: sorted(s for s, v in split.items() if v == k) for k in ("train", "val", "test")}, indent=1))
    with open(Path(out, "test.jsonl"), "w") as f:
        for r in rows:
            if split[r["subject"]] == "test":
                f.write(json.dumps(r) + "\n")
    parts = {k: [r for r in rows if split[r["subject"]] == k] for k in ("train", "val", "test")}
    train = parts["train"]
    if bad_ratio:
        g = [r for r in train if r["label"]]
        b = [r for r in train if not r["label"]]
        rng.shuffle(b)
        train = g + b[: int(bad_ratio * len(g))]
    rng.shuffle(train)
    keep = lambda r: {"prompt": r["prompt"], "completion": r["completion"], "label": r["label"]}
    for k, p in (("train", train), ("val", parts["val"]), ("test", parts["test"])):
        n_sub = len({r["subject"] for r in p})
        print(f"{k:<5} {len(p):>6} examples  good {sum(r['label'] for r in p):>5}  "
              f"bad {sum(not r['label'] for r in p):>6}  subjects {n_sub}")
    print("test subjects (never trained on):", json.loads(Path(out, "split.json").read_text())["test"])
    n_good = sum(r["label"] for r in train)
    return (Dataset.from_list([keep(r) for r in train]),
            Dataset.from_list([keep(r) for r in parts["val"]]), n_good, len(train) - n_good)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ft/kto.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--bad-ratio", type=float, default=0,
                    help="bad examples per good one; 0 = use all")
    ap.add_argument("--val-frac", type=float, default=0.1, help="share of subjects for validation")
    ap.add_argument("--test-frac", type=float, default=0.1, help="share of subjects held out for testing")
    ap.add_argument("--val-subjects", nargs="*", default=None,
                    help="name the validation subjects instead; then there is no test split")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, evals, n_good, n_bad = load(args.data, args.val_frac, args.test_frac, args.seed,
                                       args.bad_ratio, args.out, args.val_subjects)
    # KTO's recommended balance: desirable_weight * n_good ~= n_bad.
    desirable_weight = round(n_bad / max(n_good, 1), 3)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")

    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    wanted = dict(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        beta=args.beta, desirable_weight=desirable_weight, undesirable_weight=1.0,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, gradient_checkpointing=True,
        lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
        max_length=args.max_len, max_completion_length=512,
        logging_steps=10, eval_strategy="steps", eval_steps=args.save_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=3,
        report_to="none", seed=args.seed, remove_unused_columns=False)
    # trl renames and drops KTO options between releases; pass what this one has.
    fields = set(KTOConfig.__dataclass_fields__)
    dropped = sorted(k for k in wanted if k not in fields)
    if dropped:
        print("KTOConfig in this trl lacks (skipped):", dropped)
    cfg = KTOConfig(**{k: v for k, v in wanted.items() if k in fields})
    print(f"desirable_weight {desirable_weight}  (good {n_good}, bad {n_bad})")
    trainer = KTOTrainer(model=model, args=cfg, train_dataset=train, eval_dataset=evals,
                         processing_class=tok, peft_config=lora)

    resume = any(Path(args.out).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(f"{args.out}/adapter")
    tok.save_pretrained(f"{args.out}/adapter")
    print("saved adapter to", f"{args.out}/adapter")


if __name__ == "__main__":
    main()
