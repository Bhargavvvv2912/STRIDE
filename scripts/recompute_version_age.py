#!/usr/bin/env python
"""Put every arm's version age on the same denominator, from what it recorded.

Version age answers "how modern is this environment", and the answer depends on
which packages are counted. The AURA arms measure the subject's own requirement
file; the one-shot arms measured whatever the environment froze, which includes
transitive packages the file never named. Where a subject's file *is* its full
closure the two agree, which is why this stayed hidden until term-image, whose
file names 11 packages and whose environment holds 29 -- there the untouched
baseline scored 41% "at latest" while the arm that advanced all 11 scored 27%.

Nothing is re-run: each arm recorded the pins it ended with, so the comparable
figure is recomputed from the logs. The runner has been fixed, so this is for
results produced before that.

    python scripts/recompute_version_age.py                 every subject
    python scripts/recompute_version_age.py term-image      one subject
    python scripts/recompute_version_age.py --check         report, change nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aura.index import PackageIndex, version_age_report  # noqa: E402

PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?\s*==\s*([^\s;#]+)", re.M)


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def measured_packages(subject: str) -> set[str]:
    """The packages a subject declares: the denominator every arm shares.

    Taken from an AURA arm's own baseline, which is the established starting
    point -- the declared file, or its reconstruction when the declared one no
    longer installs.
    """
    for path in sorted((ROOT / "results" / subject).glob(f"{subject}__aura_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "run_end":
                before = (event.get("version_age") or {}).get("before") or {}
                per_package = before.get("per_package") or {}
                if per_package:
                    return {canonical(n) for n in per_package}
    return set()


def recompute(subject: str, index: PackageIndex, check: bool) -> list[str]:
    notes = []
    measured = measured_packages(subject)
    if not measured:
        return [f"{subject}: no AURA run to take the measured set from"]

    for path in sorted((ROOT / "results" / subject).glob(f"{subject}__*.jsonl")):
        arm = path.stem.split("__", 1)[1]
        if arm.startswith("aura_"):
            continue                       # already on the right basis
        events = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if not events or events[-1].get("event") != "run_end":
            continue
        frozen: dict[str, str] = {}
        for event in events:
            if event.get("event") == "attempt" and event.get("package") in ("<arm>", "<baseline>"):
                frozen = {canonical(k): v for k, v in (event.get("plan") or {}).items()}
        if not frozen:
            continue
        restricted = {k: v for k, v in frozen.items() if k in measured}
        if not restricted:
            notes.append(f"{subject}/{arm}: nothing in common with the measured set")
            continue
        was = (events[-1].get("version_age") or {}).get("after", {}).get("at_latest_pct")
        now = version_age_report(index, restricted)
        notes.append(f"{subject}/{arm}: {len(frozen)} pkgs -> {len(restricted)} measured, "
                     f"at latest {was}% -> {now['at_latest_pct']}%")
        if check:
            continue
        events[-1].setdefault("version_age", {})["after"] = now
        events[-1]["version_age_basis"] = "declared packages"
        path.with_suffix(".jsonl.prebasis").write_text(path.read_text())
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        result = path.with_suffix(".result.json")
        if result.exists():
            payload = json.loads(result.read_text())
            payload["version_age_after"] = now
            result.write_text(json.dumps(payload, indent=2) + "\n")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="*")
    ap.add_argument("--check", action="store_true", help="report, change nothing")
    args = ap.parse_args()

    subjects = args.subjects or sorted(
        p.name for p in (ROOT / "results").iterdir() if p.is_dir())
    index = PackageIndex()
    for subject in subjects:
        for note in recompute(subject, index, args.check):
            print(" ", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
