#!/usr/bin/env python
"""Verdicts from walk_candidates.sbatch: which candidates are LOCKED.

    python scripts/walk_report.py [screening/walk_registry.json]
"""
import json
import sys
from pathlib import Path

reg = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "screening/walk_registry.json").read_text())
rows = []
for r in reg:
    f = Path("results/walk") / f"{r['name']}.json"
    if not f.exists():
        rows.append(("pending", r, {}))
        continue
    w = json.loads(f.read_text())
    v = ("LOCKED" if w.get("locked") else "error" if w.get("error") else "walkable")
    rows.append((v, r, w))
order = {"LOCKED": 0, "walkable": 1, "error": 2, "pending": 3}
counts = {}
for v, _, _ in rows:
    counts[v] = counts.get(v, 0) + 1
print(counts)
for v, r, w in sorted(rows, key=lambda x: order[x[0]]):
    if v == "LOCKED":
        print(f"LOCKED  {r['url'].split('github.com/')[1]:45} {r.get('family','')}  tests={w['baseline'][0]}  "
              f"walk={w['walk']}  joint={w['target']}  extra={w.get('extra')}")
for v, r, w in rows:
    if v == "error":
        print(f"error   {r['name']}: {str(w.get('error'))[:80]}")
