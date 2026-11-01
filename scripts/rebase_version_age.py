#!/usr/bin/env python
"""Put every arm's version age on one date, so arms months apart compare.

"At latest" used to mean latest at the moment it was asked, and PyPI moves
while a campaign runs. Arms of a subject run together and stay consistent with
each other; the dependabot arm is collected separately, days later, and was
marked down for releases published in between, having done nothing differently.
Recollecting mean-review-collector a day later -- nothing merged, no package
changed -- moved it from 26.32% to 21.05%.

aura.index now measures against MEASURED_AS_OF. This puts results already on
disk onto that same date. Nothing is re-run: each arm recorded the pins it
ended with, and the number is recomputed from them.

    python scripts/rebase_version_age.py --check      report, change nothing
    python scripts/rebase_version_age.py              rewrite the results
    python scripts/rebase_version_age.py term-image   one subject
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aura.index import MEASURED_AS_OF, PackageIndex, version_age_report  # noqa: E402

PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?\s*==\s*([^\s;#]+)", re.M)


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pins_of(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {canonical(m.group(1)): m.group(2)
            for m in PIN.finditer(path.read_text())}


def final_pins(result_path: Path, subject: str, arm: str) -> dict[str, str]:
    """The pins an arm ended with.

    The AURA arms write a requirements file. The one-shot arms do not: they
    record the environment they installed as the plan of a single ``<arm>`` or
    ``<baseline>`` attempt, which is where this reads them from.
    """
    written = pins_of(result_path.with_name(
        result_path.name.replace(".result.json", ".requirements.txt")))
    if written:
        return written
    jsonl = result_path.with_name(f"{subject}__{arm}.jsonl")
    if not jsonl.exists():
        return {}
    plan: dict[str, str] = {}
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "attempt" and \
                event.get("package") in ("<arm>", "<baseline>"):
            if event.get("plan"):
                plan = {canonical(k): v for k, v in event["plan"].items()}
    return plan


def measured_set(subject_dir: Path, subject: str) -> set[str]:
    """The packages every arm of this subject is scored on.

    The subject's own declared set, taken from an AURA arm, which measures the
    requirement file rather than the closure it installs. The one-shot arms
    froze whatever pip pulled in -- 14 and 18 packages against the AURA arms'
    13 on test-flask -- and scoring them on that made the *baseline* arm, which
    changes nothing, report 57.14% at latest against its own 0%.
    """
    for path in sorted(subject_dir.glob(f"{subject}__aura_*.result.json")):
        payload = json.loads(path.read_text())
        before = (payload.get("version_age_before") or {}).get("per_package") or {}
        if before:
            return {canonical(n) for n in before}
    return set()


def baseline_pins(subject_dir: Path, subject: str,
                  measured: set[str]) -> dict[str, str]:
    """The environment every arm started from, at a version per package.

    The file the dependabot repository was created with is the established
    baseline and covers most of it. It does not always cover all of it: that
    file is reconstructed independently at setup time, and on flask-dj it
    holds 38 packages against the arms' 41.

    The rest is read back from the arms themselves. A package an arm did not
    advance still carries its baseline version in that arm's final pins, so
    any arm that left it alone reports what it started as. Without this the
    missing packages drop out of the denominator, which is a second silent
    change of basis while correcting the first.
    """
    pins = {k: v for k, v in
            pins_of(subject_dir / f"{subject}__dependabot.baseline.txt").items()
            if k in measured}
    for result_path in sorted(subject_dir.glob(f"{subject}__aura_*.result.json")):
        if set(pins) >= measured:
            break
        payload = json.loads(result_path.read_text())
        advanced = {canonical(n) for n in (payload.get("updated") or {})}
        final = pins_of(result_path.with_name(
            result_path.name.replace(".result.json", ".requirements.txt")))
        for name, version in final.items():
            if name in measured and name not in pins and name not in advanced:
                pins[name] = version
        # A package every arm advanced is in none of their final pins at its
        # baseline version -- but each arm recorded the move as [from, to],
        # and "from" is exactly the version wanted here.
        for name, move in (payload.get("updated") or {}).items():
            name = canonical(name)
            if name in measured and name not in pins \
                    and isinstance(move, (list, tuple)) and move and move[0]:
                pins[name] = move[0]
    return pins


def subject_as_of(subject_dir: Path, subject: str) -> str:
    """The PyPI a subject is measured against: the day its first arm started.

    Not a date after every run. Measured against a later PyPI, an arm is marked
    down for releases published after it ran, which it could not have chosen:
    test-flask's AURA arms ran on 2026-09-21 and installed wcwidth 0.8.4, then
    the newest; 0.9.1 appeared on 2026-09-23, and scored against that date they
    read as behind an arm run two days later that picked it. Measured against
    the first run, an arm that later picks a newer release is simply at latest,
    as the earlier one was. Dependabot is left out: it is collected afterwards,
    and its date says when collect ran, not when the subject was measured.
    """
    starts = []
    for path in subject_dir.glob(f"{subject}__*.jsonl"):
        if path.name.endswith("__dependabot.jsonl"):
            continue
        with path.open() as handle:
            first = handle.readline()
        try:
            started = (json.loads(first).get("env") or {}).get("started_utc") or ""
        except json.JSONDecodeError:
            continue
        if started:
            starts.append(started[:10])
    return min(starts) if starts else MEASURED_AS_OF


def rebase(subject_dir: Path, index: PackageIndex, check: bool,
           per_subject_date: bool = False) -> list[str]:
    """Put every arm of this subject on one denominator and one date."""
    notes = []
    name = subject_dir.name
    measured = measured_set(subject_dir, name)
    if not measured:
        return [f"{name}: no AURA arm to take the measured set from, skipped"]
    baseline = baseline_pins(subject_dir, name, measured)
    as_of = subject_as_of(subject_dir, name) if per_subject_date else MEASURED_AS_OF
    before = (version_age_report(index,
                                 {k: v for k, v in baseline.items() if k in measured},
                                 as_of=as_of) if baseline else None)

    for result_path in sorted(subject_dir.glob("*.result.json")):
        payload = json.loads(result_path.read_text())
        arm = result_path.name.split("__", 1)[1][: -len(".result.json")]
        final = final_pins(result_path, name, arm)
        if not final:
            # An arm that stopped before installing anything -- a model call
            # that failed, a proposal that could not be parsed -- recorded no
            # pins, but it also advanced nothing, so it ends at its baseline.
            if before is not None and not (payload.get("updated") or {}):
                if not check:
                    payload["version_age_before"] = before
                    payload["version_age_after"] = before
                    payload["version_age_as_of"] = as_of
                    result_path.write_text(json.dumps(payload, indent=2) + "\n")
                notes.append(f"{name}/{arm}: no pins recorded; advanced nothing, "
                             f"so set to its baseline")
            else:
                notes.append(f"{name}/{arm}: no pins recorded, left alone")
            continue

        stored = payload.get("version_age_after") or {}
        was, was_n = stored.get("at_latest_pct"), stored.get("n_packages")
        # Every arm scored over the same slots. An arm records only what its
        # environment froze, and the sets differ -- 9 of the subject's 13
        # declared packages for pip_upgrade, 12 for dependabot, 13 for the
        # AURA arms -- so a percentage over "what this arm happened to record"
        # is a different question per arm. A declared package an arm did not
        # move is still at its baseline version, which is what it is given.
        ends_with = {k: v for k, v in baseline.items() if k in measured}
        ends_with.update({k: v for k, v in final.items() if k in measured})
        after = version_age_report(index, ends_with, as_of=as_of)

        why = []
        # An arm that advanced nothing ends where it started. The pins it
        # recorded are the environment it installed and then rejected, which
        # is installed nowhere: reporting them made test-flask's baseline,
        # which failed validation and kept the original file, read as 57.14%
        # at latest against its own 0%.
        if before is not None and not (payload.get("updated") or {}):
            # Reported against what is stored, not against the rejected pins:
            # once a result has been rebased there is nothing left to move, and
            # saying otherwise made the repair look as though it drifted.
            if abs((was or 0) - (before["at_latest_pct"] or 0)) > 0.005:
                why.append(f"advanced nothing, so it ends at its baseline "
                           f"({was}% -> {before['at_latest_pct']}%)")
            after = before
        if was_n is not None and was_n != after["n_packages"]:
            why.append(f"denominator {was_n} -> {after['n_packages']}")
        if abs((after["at_latest_pct"] or 0) - (was or 0)) > 0.005:
            why.append(f"at latest {was}% -> {after['at_latest_pct']}%")
        if why:
            notes.append(f"{name}/{arm}: " + ", ".join(why))
        if check:
            continue
        payload["version_age_after"] = after
        if before is not None:
            payload["version_age_before"] = before
        payload["version_age_as_of"] = as_of
        result_path.write_text(json.dumps(payload, indent=2) + "\n")

    # The baseline arm advances nothing, so it has to finish where it started.
    # Anything else means the two ends are still being measured differently.
    base_result = subject_dir / f"{name}__baseline.result.json"
    if before is not None and base_result.exists() and not check:
        payload = json.loads(base_result.read_text())
        got = (payload.get("version_age_after") or {}).get("at_latest_pct")
        if abs((got or 0) - (before["at_latest_pct"] or 0)) > 0.005:
            notes.append(f"{name}/baseline: *** still {got}% after vs "
                         f"{before['at_latest_pct']}% before, and it changed "
                         f"nothing -- the two ends disagree")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subjects", nargs="*")
    ap.add_argument("--check", action="store_true", help="report, change nothing")
    ap.add_argument("--per-subject-date", action="store_true",
                    help="measure each subject against PyPI as of the day its first "
                         "arm ran, instead of the one campaign date")
    args = ap.parse_args()

    wanted = set(args.subjects)
    index = PackageIndex()
    print("measuring each subject against PyPI as of the day its first arm ran\n"
          if args.per_subject_date else f"measuring every arm against PyPI as of {MEASURED_AS_OF}\n")
    changed = 0
    for subject_dir in sorted((ROOT / "results").iterdir()):
        if not subject_dir.is_dir() or (wanted and subject_dir.name not in wanted):
            continue
        for note in rebase(subject_dir, index, args.check, args.per_subject_date):
            print(" ", note)
            changed += 1
    print(f"\n{changed} arm-result(s) {'would move' if args.check else 'moved'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
