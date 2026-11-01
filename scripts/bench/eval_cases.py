#!/usr/bin/env python3
"""Evaluate models on the coupled-upgrade benchmark (scripts/bench/build_cases.py).

For each case and model, the loop AURA's LLM tier runs: ask with AURA's compact
prompt; check the plan; on failure, re-ask with the failed plan and pip's error
in the prompt's "already tried" list; at most --attempts tries. A plan is
checked by pip's real resolver, with the case's project-side cap in force:

  valid     parses, names the target, every version above its start and not
            above the case's "newest" (which honours the cap), and real on PyPI
  resolves  pip resolves the start state with the plan applied (and the cap)
  solved    valid and resolves, and the target is above its start

Per case it records whether and on which attempt the model solved it, and how
far the target got: the share of the release gap between start and the
reference answer that it covered (1.0 = as far as the reference or further).

    python scripts/bench/eval_cases.py --cases data/bench/cases.jsonl \
        --models qwen2.5-coder:14b qwen14b-aura-kto:latest --out results/bench_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cases import canon, pip_dry, resolver_python  # noqa: E402
from packaging.version import InvalidVersion, Version  # noqa: E402

from aura.index import PackageIndex  # noqa: E402
from aura.planner import _parse_json_object, build_compact_prompt, build_prompt  # noqa: E402

STYLE = {"name": "compact"}


def conflict_log(case) -> str:
    """The pip conflict log the case's compact prompt was built from."""
    p = case["prompt"]
    if "ERROR\n" in p and "\n\nPACKAGES YOU MAY CHANGE" in p:
        return p.split("ERROR\n", 1)[1].split("\n\nPACKAGES YOU MAY CHANGE", 1)[0]
    return ""


def make_prompt(case, log, history):
    """AURA's compact prompt, or the full prompt with or without explanation-first."""
    if STYLE["name"] == "compact":
        return build_compact_prompt(target=case["target"], error_log=log, current=case["start"],
                                    available=case["newest"], history=history)
    return build_prompt(target=case["target"], error_log=log, current=case["start"],
                        available=case["newest"], history=history, style="full",
                        explanation_first=STYLE["name"] == "full-expl")


def culprits(log) -> set:
    """Packages pip's 'conflict is caused by' lines name."""
    out, on = set(), False
    for line in log.splitlines():
        if "conflict is caused by" in line:
            on = True
            continue
        if on:
            m = re.match(r"\s+(?:The user requested )?([A-Za-z0-9_.\-]+)", line)
            if not m or not line.strip():
                if out:
                    break
                continue
            out.add(canon(m.group(1)))
    return out


def V(v):
    try:
        return Version(str(v))
    except (InvalidVersion, TypeError):
        return None


OPTIONS = {"temperature": 0, "num_predict": 512}


def ask(model, prompt):
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": OPTIONS}).encode()
    req = urllib.request.Request(f"http://{host}/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r).get("response", "")


def plan_of(reply):
    d = _parse_json_object(reply) or {}
    pins = {}
    for item in d.get("proposed_plan") or []:
        if "==" in str(item):
            n, v = str(item).split("==", 1)
            pins[canon(n.strip())] = v.strip()
    return pins, d.get("resolvable") is False


def check(case, pins, idx):
    start, allowed, t = case["start"], case["newest"], case["target"]
    why = None
    if not pins:
        why = "no plan"
    elif t not in pins:
        why = f"plan does not include {t}"
    else:
        for n, v in pins.items():
            if n not in start:
                why = f"{n} is not one of the packages it may change"
            elif V(v) is None or V(v) <= V(start[n]):
                why = f"{n}=={v} is not above the installed {start[n]}"
            elif V(v) > V(allowed[n]):
                why = f"{n}=={v} is above the newest allowed {allowed[n]}"
            else:
                h = idx.history(n)
                if not h or v not in h.versions:
                    why = f"{n}=={v} does not exist"
            if why:
                break
    return why


def progress(case, final_target, idx):
    h = idx.history(case["target"])
    vs = h.versions if h else []
    pos = lambda v: vs.index(v) if v in vs else None
    s, f, r = pos(case["start"][case["target"]]), pos(final_target), pos(case["reference"][case["target"]])
    if None in (s, f, r) or r <= s:
        return None
    return round(min(1.0, (f - s) / (r - s)), 3)


def run_case(case, model, py, idx, attempts):
    history, log = [], conflict_log(case)
    for k in range(1, attempts + 1):
        prompt = (case["prompt"] if k == 1 and STYLE["name"] == "compact"
                  else make_prompt(case, log, history))
        reply = ask(model, prompt)
        pins, unresolv = plan_of(reply)
        why = "said not resolvable" if unresolv and not pins else check(case, pins, idx)
        record = {"attempt": k, "plan": pins, "reply": reply[:400]}
        if k == 1 and STYLE["name"] == "full-expl":
            expl = str((_parse_json_object(reply) or {}).get("explanation") or "")
            named = {canon(t) for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", expl)}
            cul = culprits(log)
            record["explanation"] = expl[:300]
            record["expl_names_culprit"] = bool(cul & named) if cul else None
        if why is None:
            state = {**case["start"], **pins}
            ok, out, _ = pip_dry(py, [f"{p}=={v}" for p, v in state.items()] + case.get("cap", []))
            if ok:
                return {**record, "solved": True, "valid_first": k == 1,
                        "progress": progress(case, pins[case["target"]], idx), "attempts_used": k}
            why = "resolution conflict"
            log = out
        history.append((" ".join(f"{p}=={v}" for p, v in pins.items()) or "(no plan)", why))
        if k == 1:
            first_why = why
            first_record = record
    return {"solved": False, "attempts_used": attempts, "first_failure": first_why,
            "last": record, "progress": 0.0,
            **{k: v for k, v in first_record.items() if k in ("explanation", "expl_names_culprit")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["data/bench/cases.jsonl"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--out", default="results/bench_eval.json")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="repeat runs: >0 with a --seed, so runs differ only in sampling")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--prompt-style", choices=["compact", "full-expl", "full-plan"], default="compact",
                    help="AURA's compact prompt, or the full prompt with/without explanation-first")
    ap.add_argument("--only", choices=["all", "hard"], default="all")
    args = ap.parse_args()
    OPTIONS["temperature"] = args.temperature
    if args.seed is not None:
        OPTIONS["seed"] = args.seed
    STYLE["name"] = args.prompt_style
    cases, seen_ids = [], set()
    for f in args.cases:
        for l in open(f):
            c = json.loads(l)
            if c["id"] not in seen_ids:     # a build task run twice repeats cases
                seen_ids.add(c["id"])
                cases.append(c)
    if args.only == "hard":
        cases = [c for c in cases if c["kind"] == "hard"]
    print(f"{len(cases)} distinct cases, prompt style {args.prompt_style}")
    idx = PackageIndex()
    cache = Path(args.cases[0]).parent / ".resolvers"
    pys = {p: resolver_python(p, cache) for p in sorted({c["python"] for c in cases})}
    out = Path(args.out)
    res = json.loads(out.read_text())["results"] if out.exists() else {}
    for m in args.models:
        res.setdefault(m, {})
        for i, c in enumerate(cases):
            if c["id"] in res[m]:
                continue
            res[m][c["id"]] = run_case(c, m, pys[c["python"]], idx, args.attempts)
            if (i + 1) % 20 == 0:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"results": res}, indent=1))
                print(f"  {m}: {i + 1}/{len(cases)}", flush=True)
    out.write_text(json.dumps({"results": res}, indent=1))

    # Summary: by kind and by seen/unseen family
    groups = {"all": lambda c: True, "hard": lambda c: c["kind"] == "hard",
              "easy": lambda c: c["kind"] == "easy",
              "unseen families": lambda c: not c["seen_in_training"],
              "seen families": lambda c: c["seen_in_training"]}
    print(f"\n{'subset':<17}{'n':>5}" + "".join(f"{m[:26]:>30}" for m in args.models))
    for g, f in groups.items():
        sub = [c for c in cases if f(c)]
        if not sub:
            continue
        cells = []
        for m in args.models:
            r = [res[m][c["id"]] for c in sub]
            solved = sum(x["solved"] for x in r) / len(r)
            first = sum(x.get("valid_first", False) for x in r) / len(r)
            prog = [x["progress"] for x in r if x.get("progress") is not None]
            cells.append(f"solved {solved:5.1%} 1st {first:5.1%} prog {sum(prog)/max(1,len(prog)):.2f}")
        print(f"{g:<17}{len(sub):>5}" + "".join(f"{c:>30}" for c in cells))
    print("written to", out)


if __name__ == "__main__":
    main()
