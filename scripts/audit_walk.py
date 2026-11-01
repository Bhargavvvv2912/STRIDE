#!/usr/bin/env python3
"""Read-only audit of walk results; classifications are log-evidence hints.

python3 scripts/audit_walk.py --registry screening/walk_registry_1.json
Use --json for complete per-candidate evidence. Never infers retry safety.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def classify(w):
    error = w.get("error")
    evidence = "\n".join(str(w.get(k) or "") for k in ("error", "why", "tail"))
    low = evidence.lower()
    if error:
        if "clone failed" in low or "no interpreter" in low:
            return "infrastructure", evidence
        if "timeoutexpired" in low or "timed out" in low or "timeout" in low:
            return "timeout evidence", evidence
        if any(s in low for s in ("no space left", "disk quota", "cannot allocate memory")):
            return "resource failure", evidence
        if any(s in low for s in ("resolutionimpossible", "no matching distribution", "could not find a version", "failed building wheel", "metadata-generation-failed", "legacy-install-failure", "invalid requirement:")):
            return "installation failure evidence", evidence
        if "could not open requirements file" in low:
            return "requirements path failure", evidence
        if "git clone" in low and "exit code: 128" in low:
            return "dependency checkout failure", evidence
        if "keyerror: 'display'" in low or "selenium.common" in low:
            return "GUI/browser requirement evidence", evidence
        if "bucket does not exist" in low or "429 client error" in low:
            return "external service evidence", evidence
        if "improperlyconfigured" in low or "isn't in an application in installed_apps" in low:
            return "Django configuration failure", evidence
        if "syntaxerror:" in low:
            return "source syntax failure", evidence
        if "import file mismatch" in low:
            return "test collection path collision", evidence
        if "no module named" in low:
            return "missing module (test/runtime role needs review)", evidence
        if any(s in low for s in ("connectionrefusederror", "connection refused", "name resolution", "network is unreachable")):
            return "network/service failure evidence", evidence
        if "filenotfounderror" in low or "no such file or directory" in low:
            return "missing file evidence", evidence
        if "assertionerror" in low:
            return "test assertion failure evidence", evidence
        if any(s in low for s in ("importerror:", "attributeerror:", "typeerror:", "numpy.dtype size changed", "keyerror: 'break_loop'")):
            return "runtime/API incompatibility evidence", evidence
        if "walk crashed" in low:
            return "walk crashed (cause unknown)", evidence
        return "baseline/error cause unknown", evidence
    base = w.get("baseline")
    if not isinstance(base, (list, tuple)) or len(base) != 2 or not all(isinstance(n, (int, float)) for n in base):
        return "unrecognized result", evidence
    if base[0] <= 0 or base[1] != 0:
        return "unclean baseline accepted by checker", evidence
    if w.get("locked") is True:
        return "LOCKED reported (confirm required)", evidence
    if w.get("locked") is not False:
        return "unrecognized result", evidence
    if w.get("greedy_reaches"):
        return "negative: passing greedy probe", evidence
    if w.get("target"):
        return "negative: inspect target/verdict inconsistency", evidence
    return "negative: no advantage found within search budget", evidence


def audit(registry, results):
    rows = []
    names = {r["name"] for r in registry}
    for r in registry:
        p = results / (r["name"] + ".json")
        row = {"name": r["name"], "url": r.get("url"), "family": r.get("family", "unknown")}
        if not p.exists():
            row.update(category="pending/no local result", evidence="")
        else:
            try:
                w = json.loads(p.read_text())
                if not isinstance(w, dict):
                    raise ValueError("result is not a JSON object")
                category, evidence = classify(w)
                row.update(category=category, evidence=evidence, result=w)
            except (OSError, ValueError) as exc:
                row.update(category="unreadable result (may be mid-write)", evidence=str(exc))
        rows.append(row)
    return {
        "counts": dict(Counter(r["category"] for r in rows)),
        "families": {family: dict(Counter(r["category"] for r in rows if r["family"] == family))
                     for family in sorted({r["family"] for r in rows})},
        "outside_registry": sorted(p.name for p in results.glob("*.json") if p.stem not in names),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("screening/walk_registry_1.json"))
    parser.add_argument("--results", type=Path, default=Path("results/walk"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.results.is_dir():
        parser.error("results directory does not exist: {}".format(args.results))
    report = audit(json.loads(args.registry.read_text()), args.results)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print("Categories are evidence hints, not automatic retry decisions.")
    print("Counts:", json.dumps(report["counts"], sort_keys=True))
    print("By family:", json.dumps(report["families"], sort_keys=True))
    for r in report["rows"]:
        if r["category"] == "pending/no local result":
            continue
        print("\n{} | {} | {}".format(r["name"], r["family"], r["category"]))
        if r["evidence"]:
            print(r["evidence"])
    if report["outside_registry"]:
        print("\nOutside registry (not counted):", ", ".join(report["outside_registry"]))


if __name__ == "__main__":
    main()
