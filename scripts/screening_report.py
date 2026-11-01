#!/usr/bin/env python
"""Verdicts for candidates screened on the cluster by screen_candidates.sbatch.

    python scripts/screening_report.py [screening/cluster_candidates.json]

A candidate is a KEEP when it has a working baseline and the naive upgrade
breaks it, WEAK when the naive upgrade also passes, and REJECT when no baseline
could be established (or nothing ran). Reads results/<name>/ only.
"""
import json
import re
import sys
from pathlib import Path


def tests_passed(log: Path) -> str:
    if not log.exists():
        return "-"
    found = re.findall(r"(\d+) passed", log.read_text(errors="replace"))
    return found[0] if found else "0"


def verdict(name: str):
    root = Path("results") / name
    base = root / f"{name}__baseline.result.json"
    pip = root / f"{name}__pip_upgrade.result.json"
    if not base.exists():
        return "not run", "-", "-", "no baseline result yet"
    b = json.loads(base.read_text())
    events = [json.loads(line) for line in (root / f"{name}__baseline.jsonl").open()] \
        if (root / f"{name}__baseline.jsonl").exists() else []
    recovered = next((e.get("stage") for e in events
                      if e.get("note") == "baseline_recoverable" and e.get("stage")), "")
    usable = b["outcome"] == "kept_baseline" or bool(recovered)
    pins = (b.get("version_age_before") or {}).get("n_packages", "-")
    passed = tests_passed(root / f"{name}__baseline.log")
    if not usable:
        return "REJECT", passed, pins, (b.get("reason") or "no working baseline")[:110]
    if not pip.exists():
        return "baseline ok", passed, pins, "pip_upgrade not run yet"
    p = json.loads(pip.read_text())
    if p["outcome"] == "failed":
        return "KEEP", passed, pins, ("naive upgrade breaks: " + (p.get("reason") or ""))[:110] \
            + (f" (baseline {recovered})" if recovered else "")
    return "weak", passed, pins, "naive upgrade also passes"


def main() -> int:
    registry = sys.argv[1] if len(sys.argv) > 1 else "screening/cluster_candidates.json"
    rows = [(s["name"], s["url"].split("github.com/")[-1], *verdict(s["name"]))
            for s in json.loads(Path(registry).read_text())]
    order = {"KEEP": 0, "weak": 1, "baseline ok": 2, "REJECT": 3, "not run": 4}
    print(f"{'verdict':<12} {'candidate':<42} {'tests':>5} {'pins':>4}  note")
    for name, repo, v, passed, pins, note in sorted(rows, key=lambda r: order.get(r[2], 9)):
        print(f"{v:<12} {repo:<42} {passed!s:>5} {pins!s:>4}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
