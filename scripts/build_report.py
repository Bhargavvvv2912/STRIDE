#!/usr/bin/env python
"""Collect every included subject's arm results into one JSON for the report.

    python scripts/build_report.py [out.json]

Per (subject, arm): packages advanced / measured, % at latest after the run,
tokens and calls, outcome. Per subject where a full arm's LLM tier had a move
accepted: which models, what they moved, greedy's final versions against the
best full arm's, and whether that arm ends strictly ahead of greedy.
"""
import json
import re
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODELS = ["qwen3b", "llama8b", "qwen14b", "qwen32b"]
ARMS = (["pip_upgrade", "dependabot", "aura_linear_greedy"] + [f"aura_full_{m}" for m in MODELS]
        + [f"aura_llm_{m}" for m in MODELS] + [f"llm_oneshot_{m}" for m in MODELS])
FAMILY = [("flask", "Flask 1 → 2 stack"), ("werkzeug", "Flask 1 → 2 stack"), ("jinja2", "Flask 1 → 2 stack"), ("torch", "torch / torchvision lockstep"),
          ("tensorflow", "TensorFlow / estimator lockstep"), ("databases", "databases / SQLAlchemy lockstep"),
          ("celery", "celery family"), ("boto3", "boto3 / botocore"), ("pytest", "pytest + plugins")]
CAVEATS = {"nbdt": "Greedy's target, torch 2.5.1, declares Python >= 3.8 but has no 3.8 wheel; "
                   "pip reported no matching distribution, so greedy found no blocker to follow."}


def canon(n):
    return re.sub(r"[-_.]+", "-", n).lower()


def V(v):
    try:
        return Version(v)
    except (InvalidVersion, TypeError):
        return Version("0")


def pins(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==\s*([^\s;#]+)", line)
            if m:
                out[canon(m.group(1))] = m.group(3)
    return out


def cell(path):
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    before = r.get("version_age_before") or {}
    after = r.get("version_age_after") or before
    c = r.get("counters") or {}
    return {"adv": len(r.get("updated") or {}), "n": before.get("n_packages") or 0,
            "pct": after.get("at_latest_pct"), "tok": (c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0),
            "calls": c.get("llm_calls") or 0, "out": r.get("outcome"), "why": (r.get("reason") or "")[:120]}


def llm_moves(subject, arm):
    log = RES / subject / f"{subject}__{arm}.jsonl"
    moves = []
    if log.exists():
        for line in log.open():
            e = json.loads(line)
            if e.get("event") == "attempt" and e.get("accepted") and e.get("tier") == "llm":
                moves.append(e.get("plan") or {})
    return moves


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "report_data.json"
    subjects, llm_rows = [], []
    for s in json.loads((ROOT / "subjects.json").read_text()):
        if s.get("domain") in ("ft-test", "control-noncoupled"):
            continue    # the fine-tuning test and the controls are reported separately
        if s.get("rejection_reason"):
            continue
        name = s["name"]
        d = RES / name
        cells = {a: cell(d / f"{name}__{a}.result.json") for a in ARMS}
        if not any(cells.values()):
            continue
        done = sum(1 for a in ARMS if a != "dependabot" and cells[a])
        # Which tier made each package's final move, in the full arm that advanced most.
        fulls = [a for a in ARMS if a.startswith("aura_full_") and cells[a]]
        tiers = {}
        if fulls:
            best_arm = max(fulls, key=lambda a: (cells[a]["adv"], -(cells[a].get("pct") or 0) * 0))
            log = d / f"{name}__{best_arm}.jsonl"
            last = {}
            if log.exists():
                for line in log.open():
                    e = json.loads(line)
                    if e.get("event") == "attempt" and e.get("accepted") and e.get("tier") in ("linear", "greedy", "llm"):
                        for k in (e.get("plan") or {}):
                            last[canon(k)] = e["tier"]
            for t in last.values():
                tiers[t] = tiers.get(t, 0) + 1
            tiers["arm"] = best_arm
        pip = cells.get("pip_upgrade") or {}
        if not done:
            continue    # Dependabot-only (e.g. graphene-mongodb): no AURA result to report
        subjects.append({"name": name, "py": s.get("python_version"), "date": s.get("commit_date"),
                         "keep": pip.get("out") == "failed", "arms_done": done, "cells": cells, "tiers": tiers})
        greedy = pins(d / f"{name}__aura_linear_greedy.requirements.txt")
        per_model, best = {}, None
        for m in MODELS:
            arm = f"aura_full_{m}"
            mv = llm_moves(name, arm)
            if not mv:
                continue
            full = pins(d / f"{name}__{arm}.requirements.txt")
            higher = [k for k in full if k in greedy and V(full[k]) > V(greedy[k])]
            lower = [k for k in full if k in greedy and V(full[k]) < V(greedy[k])]
            ahead = bool(higher) and not lower
            per_model[m] = {"moves": len(mv), "ahead": ahead, "higher": higher}
            reach = sum(V(full[k]).major * 1000 + V(full[k]).minor for k in higher)
            if ahead and (best is None or reach > best[3]):
                best = (m, full, higher, reach)
        if not per_model:
            continue
        moved = sorted({canon(k) for m in MODELS for p in llm_moves(name, f"aura_full_{m}") for k in p})
        fam = next((label for key, label in FAMILY if any(key in p for p in moved)), "other")
        row = {"name": name, "family": fam, "decisive": best is not None, "models": per_model,
               "moved": moved, "caveat": CAVEATS.get(name, ""), "complete": done >= 14}
        if best:
            m, full, higher, _ = best
            row["best_model"] = m
            row["gain"] = [{"pkg": k, "greedy": greedy[k], "llm": full[k],
                            "level": ("major" if V(full[k]).major > V(greedy[k]).major else
                                      "minor" if V(full[k]).minor > V(greedy[k]).minor else "patch")}
                           for k in sorted(higher)]
        # AURA without the LLM tier (greedy) against AURA with it (the best full arm).
        arm = f"aura_full_{row.get('best_model') or next(iter(per_model))}"
        row["llm_arm"] = arm
        row["without_llm"] = cells.get("aura_linear_greedy")
        row["with_llm"] = cells.get(arm)
        llm_rows.append(row)
    out_path.write_text(json.dumps({"subjects": subjects, "llm": llm_rows}, indent=1))
    dec = sum(r["decisive"] for r in llm_rows)
    print(f"{len(subjects)} subjects; LLM contributed on {len(llm_rows)}, decisive on {dec} -> {out_path}")


if __name__ == "__main__":
    main()
