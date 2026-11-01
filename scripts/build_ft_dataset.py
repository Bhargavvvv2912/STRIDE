"""Build fine-tuning data for the LLM tier from AURA's own run logs.

Every ``llm_call`` event records the planner's inputs and the model's raw reply;
the ``attempt`` events after it record what happened when AURA tried the plan.
This rebuilds the exact prompt the model saw (``build_compact_prompt``) and
labels each reply by that outcome, giving:

  sft.jsonl  -- prompt -> reply, for replies whose plan AURA accepted
  dpo.jsonl  -- prompt, chosen (accepted) reply, rejected (failed) reply
  all.jsonl  -- every call with its label, for analysis and replay evaluation

Only version plans are ever targets: the model proposes versions, never code.
Each example carries its subject and family so evaluation can hold out whole
families (a test subject's family must never appear in training).

    python scripts/build_ft_dataset.py --out data/ft [--holdout-family flask1to2 ...]
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aura.planner import _parse_json_object, build_compact_prompt  # noqa: E402

RESULTS = Path("results")

# Library family by the packages a subject pins; first match wins.
FAMILIES = [
    ("transformers", {"transformers", "huggingface-hub", "tokenizers"}),
    ("tensorflow", {"tensorflow", "keras", "tensorflow-estimator"}),
    ("torch", {"torch", "torchvision", "torchaudio", "pytorch-lightning", "lightning"}),
    ("flask", {"flask", "werkzeug"}),
    ("django", {"django", "djangorestframework", "channels"}),
    ("aws", {"boto3", "botocore"}),
    ("graphql", {"graphene", "graphql-core"}),
    ("fastapi", {"fastapi", "starlette", "pydantic"}),
]


def family_of(pins: dict) -> str:
    names = set(pins)
    for fam, keys in FAMILIES:
        if names & keys:
            return fam
    return "other"


def _value(v):
    """Log fields were written as Python reprs inside JSON; decode either form."""
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def _plan_key(plan) -> frozenset:
    if isinstance(plan, dict):
        return frozenset((k.lower(), str(v)) for k, v in plan.items())
    out = set()
    for item in plan or []:
        if "==" in str(item):
            n, v = str(item).split("==", 1)
            out.add((n.strip().lower(), v.strip()))
    return frozenset(out)


def label_calls(path: Path):
    """Yield (call, outcome) for each llm_call in one run log.

    The outcome is the attempt(s) on the same package that follow the call and
    precede the next call: ``accepted`` if any was accepted, else the first
    attempt's error class; ``invalid`` if no plan reached an attempt (it was
    unparseable, rejected by the planner's checks, or marked unresolvable).
    """
    events = [json.loads(line) for line in open(path)]
    calls = [i for i, e in enumerate(events) if e.get("event") == "llm_call"]
    for n, i in enumerate(calls):
        call = events[i]
        end = calls[n + 1] if n + 1 < len(calls) else len(events)
        attempts = [e for e in events[i + 1:end]
                    if e.get("event") == "attempt" and e.get("tier") == "llm"]
        proposed = _plan_key(_value(call.get("proposed")))
        if proposed:
            matching = [a for a in attempts if _plan_key(_value(a.get("plan"))) >= proposed]
            attempts = matching or attempts
        if not attempts:
            outcome = "invalid"
        elif any(_value(a.get("accepted")) is True for a in attempts):
            outcome = "accepted"
        else:
            outcome = _value(attempts[0].get("error_class")) or "rejected"
        yield call, outcome


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ft")
    ap.add_argument("--holdout-family", action="append", default=[],
                    help="exclude these families from sft/dpo (kept in all.jsonl)")
    ap.add_argument("--holdout-subject", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, families = [], {}
    for path in sorted(glob.glob(str(RESULTS / "*" / "*__aura_*.jsonl"))):
        subject = Path(path).parent.name
        arm = Path(path).name.split("__")[1].rsplit(".", 1)[0]
        if arm.endswith("ft"):
            continue    # the fine-tuned model's own runs never train it
        for call, outcome in label_calls(Path(path)):
            inp = _value(call.get("inputs")) or {}
            if not isinstance(inp, dict) or "target" not in inp:
                continue
            current = _value(inp.get("current")) or {}
            history = [tuple(h) for h in (_value(inp.get("history")) or [])]
            prompt = build_compact_prompt(
                target=inp["target"], error_log=inp.get("error_log") or "",
                current=current, available=_value(inp.get("available")) or {},
                history=history)
            parsed = _parse_json_object(str(call.get("raw_response") or ""))
            fam = families.setdefault(subject, family_of(current))
            rows.append({
                "subject": subject, "family": fam, "arm": arm,
                "model": call.get("model"), "target": inp["target"],
                "prompt": prompt, "response": str(call.get("raw_response") or ""),
                "plan": (parsed or {}).get("proposed_plan"),
                "outcome": outcome,
                "pid": hashlib.sha1(prompt.encode()).hexdigest()[:16],
            })

    with open(out / "all.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    held = lambda r: r["family"] in args.holdout_family or r["subject"] in args.holdout_subject
    train = [r for r in rows if not held(r)]

    # SFT: one accepted reply per distinct prompt, in the canonical JSON form the
    # planner parses -- the model learns the format and the plan, nothing else.
    sft, seen = [], set()
    for r in train:
        if r["outcome"] != "accepted" or not r["plan"] or r["pid"] in seen:
            continue
        seen.add(r["pid"])
        reply = json.dumps({"resolvable": True, "plausible": True, "proposed_plan": r["plan"]})
        sft.append({"messages": [{"role": "user", "content": r["prompt"]},
                                 {"role": "assistant", "content": reply}],
                    "subject": r["subject"], "family": r["family"]})

    # DPO: same prompt, an accepted reply against a failed one.
    by_prompt = collections.defaultdict(lambda: {"good": [], "bad": []})
    for r in train:
        if not r["plan"] and r["outcome"] != "invalid":
            continue
        by_prompt[r["pid"]]["good" if r["outcome"] == "accepted" else "bad"].append(r)
    rng = random.Random(args.seed)
    dpo = []
    for pid, g in by_prompt.items():
        goods = {json.dumps(sorted(x["plan"])) for x in g["good"] if x["plan"]}
        bads = {x["response"] for x in g["bad"] if x["response"]}
        if not goods or not bads:
            continue
        r0 = g["good"][0]
        for bad in sorted(bads)[:3]:
            plan = json.loads(rng.choice(sorted(goods)))
            dpo.append({"prompt": r0["prompt"],
                        "chosen": json.dumps({"resolvable": True, "plausible": True, "proposed_plan": plan}),
                        "rejected": bad, "subject": r0["subject"], "family": r0["family"]})

    # KTO: every distinct (prompt, reply) labelled good or bad -- no pairing
    # needed. Good replies in the canonical JSON form; bad replies exactly as the
    # model wrote them, so training pushes down what it actually produced.
    kto, seen_k = [], set()
    for r in train:
        if r["outcome"] == "accepted":
            if not r["plan"]:
                continue
            reply = json.dumps({"resolvable": True, "plausible": True, "proposed_plan": r["plan"]})
        else:
            reply = r["response"].strip()
            if not reply:
                continue
        key = (r["pid"], reply)
        if key in seen_k:
            continue
        seen_k.add(key)
        kto.append({"prompt": [{"role": "user", "content": r["prompt"]}],
                    "completion": [{"role": "assistant", "content": reply}],
                    "label": r["outcome"] == "accepted", "outcome": r["outcome"],
                    "subject": r["subject"], "family": r["family"]})

    for name, data in (("sft", sft), ("dpo", dpo), ("kto", kto)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")

    # Summary
    print(f"calls: {len(rows)}   distinct prompts: {len({r['pid'] for r in rows})}")
    print("outcomes:", dict(collections.Counter(r["outcome"] for r in rows).most_common()))
    print(f"sft examples: {len(sft)}   dpo pairs: {len(dpo)}   held out: {args.holdout_family + args.holdout_subject}")
    print(f"kto examples: {len(kto)}  good {sum(k['label'] for k in kto)}  bad {sum(not k['label'] for k in kto)}  "
          f"{dict(collections.Counter(k['outcome'] for k in kto).most_common())}")
    fam_count = collections.Counter(r["family"] for r in rows)
    fam_acc = collections.Counter(r["family"] for r in rows if r["outcome"] == "accepted")
    fam_sft = collections.Counter(r["family"] for r in sft)
    fam_subj = collections.defaultdict(set)
    for r in rows:
        fam_subj[r["family"]].add(r["subject"])
    print(f"\n{'family':<13}{'subjects':>9}{'calls':>8}{'accepted':>10}{'sft':>6}")
    for fam, n in fam_count.most_common():
        print(f"{fam:<13}{len(fam_subj[fam]):>9}{n:>8}{fam_acc[fam]:>10}{fam_sft[fam]:>6}")


if __name__ == "__main__":
    main()
