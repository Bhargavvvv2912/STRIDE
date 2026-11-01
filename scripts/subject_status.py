#!/usr/bin/env python
"""Write SUBJECTS.md: the sample, why each repository is in it, and what ran.

The sample is decided in three places -- subjects.json says what is included,
screening/screening.jsonl says what was measured to decide that, and results/
says what has been run -- and keeping the answer in someone's head is how a
campaign ends up with two subjects nobody can account for. This regenerates the
one file that states all three.

    python scripts/subject_status.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.subjects import ERAS, era_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FULL_ARM_RESULT = "*__aura_full_qwen14b.result.json"


def slug_of(subject: dict) -> str:
    return subject["url"].rstrip("/").removesuffix(".git").split("github.com/")[-1]


def tests_passing(record: dict) -> str:
    """How many tests pass at the baseline the subject is actually judged on."""
    baseline = record.get("baseline") or {}
    usable = record.get("usable_test_targets")
    per_file = baseline.get("per_file") or {}
    if usable and per_file:
        # A suite can be part sound and part unrunnable; only the files the
        # subject is judged on count.
        return str(sum(per_file[t]["passed"] for t in usable if t in per_file))
    return str(baseline.get("passed", "?"))


def main() -> int:
    subjects = {s["name"]: s for s in json.loads((ROOT / "subjects.json").read_text())}
    records: dict[str, dict] = {}
    log = ROOT / "screening" / "screening.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                records[row["repo"]] = row          # last verdict wins
    done = {p.parent.name for p in (ROOT / "results").glob(f"*/{FULL_ARM_RESULT}")}

    out = [
        "# Subjects",
        "",
        "The sample, and why each repository is in it. Regenerate with",
        "`python scripts/subject_status.py`.",
        "",
        "A subject needs a working environment to measure against -- its own",
        "requirements, or the reconstruction of them at its commit date -- and its",
        "own tests. It does *not* need the naive upgrade to break: requiring that",
        "would select exactly the cases where the baselines fail, so subjects where",
        "upgrading everything already works are kept, and every arm is expected to",
        "tie on them. An arm that breaks such a subject is worth knowing about.",
        "",
        "| subject | era | repository | tests | python | baseline | naive upgrade | runs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, subject in subjects.items():
        record = records.get(slug_of(subject), {})
        stage = record.get("baseline_stage", "declared")
        repaired = len(record.get("baseline_repaired") or {})
        baseline = ("as written" if stage == "declared"
                    else f"reconstructed ({repaired} pkgs)")
        upgrade = ("breaks" if record.get("verdict") == "keep"
                   else "passes (tie expected)" if record.get("verdict") == "weak"
                   else "?")
        state = "done (11 arms)" if name in done else "not yet run"
        out.append(f"| {name} | {era_of(subject.get('commit_date', ''))} "
                   f"| {slug_of(subject)} | {tests_passing(record)} "
                   f"| {subject.get('python_version', '?')} | {baseline} | {upgrade} "
                   f"| {state} |")

    # The shape of the sample, stated rather than implied.
    counts: dict[str, int] = {name: 0 for name, _, _, _ in ERAS}
    for subject in subjects.values():
        era = era_of(subject.get("commit_date", ""))
        counts[era] = counts.get(era, 0) + 1
    out += ["", "## Strata", "",
            "Dependency rot is a function of time, so the sample is stratified by",
            "the date of each subject's pinned commit. A stratum with no subjects",
            "is a gap in the sample, not an absence of projects.", "",
            "| era | years | what it means | subjects |",
            "| --- | --- | --- | --- |"]
    for name, start, end, why in ERAS:
        years = f"{start[:4]}-{end[:4]}" if end[:4] != "2100" else f"{start[:4]}+"
        out.append(f"| {name} | {years} | {why} | {counts.get(name, 0)} |")

    rejected = [r for r in records.values() if r["verdict"] == "reject"]
    out += [
        "",
        f"## Screened and not used ({len(rejected)} of {len(records)})",
        "",
        "Kept as a record rather than deleted: how often a published Python",
        "project can no longer be installed or tested is itself a result, and it",
        "is the denominator behind every claim about the sample.",
        "",
    ]
    for record in sorted(rejected, key=lambda r: r["repo"]):
        out.append(f"- `{record['repo']}` -- {record['reason'][:110]}")

    (ROOT / "SUBJECTS.md").write_text("\n".join(out) + "\n")
    print(f"{len(subjects)} subject(s), {len(done)} run, "
          f"{len(rejected)} screened and not used -> SUBJECTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
