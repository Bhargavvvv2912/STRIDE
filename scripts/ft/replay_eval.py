"""Replay held-out LLM-tier prompts through several Ollama models and score them.

Takes prompts from subjects no model was trained on (default: the KTO run's
validation subjects), asks each model the exact prompt AURA sent, and scores
every reply without running AURA:

  parsed      a JSON object the planner accepts
  unresolv    it says "resolvable": false
  has_target  the plan includes the package the prompt asks about
  in_range    every version is above "installed" and not above "newest"
  exists      every version is a real, non-yanked release on PyPI
  valid       parsed, has_target, in_range and exists together
  same_as_ok  the plan equals one AURA accepted for this prompt (when known)

    python scripts/ft/replay_eval.py --models qwen2.5-coder:14b qwen14b-aura:latest qwen14b-aura-kto:latest \
        --subjects rhizonet authors-of-leisure age-of-empires-api --n 300 --out results/ft_replay.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packaging.version import InvalidVersion, Version  # noqa: E402

from aura.index import PackageIndex  # noqa: E402
from aura.planner import _parse_json_object  # noqa: E402

canon = lambda n: re.sub(r"[-_.]+", "-", n).lower()
ROW = re.compile(r"^\s{2}(\S+)\s+installed\s+(\S+)\s+newest\s+(\S+)", re.M)


def V(v):
    try:
        return Version(v)
    except (InvalidVersion, TypeError):
        return None


def ask(model: str, prompt: str) -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": 512}}).encode()
    req = urllib.request.Request(f"http://{host}/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r).get("response", "")


def score(reply: str, prompt: str, target: str, idx: PackageIndex, accepted: set) -> dict:
    table = {canon(n): (i, nw) for n, i, nw in ROW.findall(prompt)}
    d = _parse_json_object(reply) or {}
    plan = d.get("proposed_plan") if isinstance(d.get("proposed_plan"), list) else None
    s = {"parsed": plan is not None or d.get("resolvable") is False,
         "unresolv": d.get("resolvable") is False}
    pins = {}
    for item in plan or []:
        if "==" in str(item):
            n, v = str(item).split("==", 1)
            pins[canon(n.strip())] = v.strip()
    s["has_target"] = canon(target) in pins
    in_range = bool(pins)
    exists = bool(pins)
    for n, v in pins.items():
        vv = V(v)
        lo, hi = table.get(n, (None, None))
        if vv is None or (lo and V(lo) and vv <= V(lo)) or (hi and V(hi) and vv > V(hi)):
            in_range = False
        h = idx.history(n)
        if not h or v not in h.versions or h.is_yanked(v):
            exists = False
    s["in_range"], s["exists"] = in_range, exists
    s["valid"] = bool(plan) and s["has_target"] and in_range and exists
    s["same_as_ok"] = bool(pins) and frozenset(pins.items()) in accepted
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ft/all.jsonl")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ft_replay.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    rows = [r for r in rows if r["subject"] in args.subjects]
    accepted = defaultdict(set)
    for r in rows:
        if r["outcome"] == "accepted" and r["plan"]:
            accepted[r["pid"]].add(frozenset((canon(p.split("==")[0]), p.split("==")[1])
                                             for p in r["plan"] if "==" in p))
    by_pid = {}
    for r in rows:
        by_pid.setdefault(r["pid"], r)
    prompts = sorted(by_pid.values(), key=lambda r: r["pid"])
    random.Random(args.seed).shuffle(prompts)
    prompts = prompts[: args.n]
    print(f"{len(prompts)} distinct held-out prompts from {sorted({p['subject'] for p in prompts})}")

    idx = PackageIndex()
    results = {m: [] for m in args.models}
    for i, p in enumerate(prompts):
        for m in args.models:
            reply = ask(m, p["prompt"])
            s = score(reply, p["prompt"], p["target"], idx, accepted.get(p["pid"], set()))
            results[m].append({"pid": p["pid"], "subject": p["subject"], "reply": reply[:600], **s})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(prompts)}", flush=True)

    keys = ["parsed", "valid", "has_target", "in_range", "exists", "unresolv", "same_as_ok"]
    summary = {m: {k: round(sum(r[k] for r in rs) / max(1, len(rs)), 3) for k in keys}
               for m, rs in results.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "n": len(prompts),
                                          "subjects": args.subjects, "results": results}, indent=1))
    print(f"\n{'model':<28}" + "".join(f"{k:>11}" for k in keys))
    for m, s in summary.items():
        print(f"{m:<28}" + "".join(f"{s[k]:>11.1%}" for k in keys))
    print("written to", args.out)


if __name__ == "__main__":
    main()
