#!/usr/bin/env python
"""Build the anonymous replication package for the paper.

    python scripts/make_artifact.py            -> dist/aura-artifact.zip

Copies the code, the 50 subjects' results for every approach, the benchmark
and the paper's analysis inputs into dist/aura-artifact/, replaces every string
that identifies the authors (home paths, user names, the cluster's host name,
the GitHub account), then scans the copy and refuses to zip it if anything
identifying is left. Nothing is uploaded anywhere.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "aura-artifact"
SUBJECTS = (ROOT / "scripts" / "bot_subjects.txt").read_text().split()

# Order matters: longer, more specific strings first.
SCRUB = [
    ("/Users/bhargavks/Documents/AUR/AURA", "/path/to/AURA"),
    ("/Users/bhargavks", "/home/user"),
    ("/z/bsrinid", "/home/user"),
    ("/scratch/or/turbo/aura", "/scratch/aura"),
    ("bsrinid@frances.eecs.umich.edu", "user@cluster"),
    ("frances.eecs.umich.edu", "cluster"),
    ("bsrinid@umich.edu", "anonymous@example.org"),
    ("Bhargavvvv2912", "anonymous"),
    ("bhargavks", "user"),
    ("bsrinid", "user"),
    ("Bhargavs-MacBook-Pro", "laptop"),
]
# Paths cut off mid-name in error excerpts ("srinid/aura-scratch/..."), which
# no exact string above matches.
SCRUB_RE = [(re.compile(r"[\w./-]*srinid(?=/)"), "/home/user")]
# Notes in the code that quote the previous round of reviews: harmless, but
# they are not part of the package. Python sources only; API paths
# ("/v1/chat/completions") and "cgroup v1" are left alone.
HISTORY_RE = [
    (re.compile(r"Reviewer [ABC]'s"), "a reviewer's"),
    (re.compile(r"Reviewer [ABC]"), "a reviewer"),
    (re.compile(r"(?<!cgroup )(?<![/\w])v1's"), "the earlier version's"),
    (re.compile(r"(?<!cgroup )(?<![/\w])v1\b(?![/.\d])"), "the earlier version"),
]
# Anything matching these after scrubbing stops the build.
LEAKS = re.compile(r"bhargav|bsrinid|umich|frances\.|Bhargavvvv|srinid", re.I)

TEXT_SUFFIXES = {".py", ".sh", ".sbatch", ".json", ".jsonl", ".log", ".txt", ".md",
                 ".toml", ".cfg", ".ini", ".yml", ".yaml", ".csv", ".in", ""}
SKIP_RESULT = re.compile(r"\.(bak|prebasis)$|\.dryrun$|\.(setup|collect)\.out$")


def tracked_code() -> list[Path]:
    """Every file git tracks, plus the new scripts not yet committed; the
    working-tree version of each, so today's changes are included."""
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    extra = ["scripts/real_renovate.py", "scripts/renovate_all.sh",
             "scripts/renovate_collect_one.sh", "scripts/renovate_proposals.py",
             "scripts/slurm/submit_chunks.sh", "scripts/bot_subjects.txt",
]
    keep = []
    for f in files + extra:
        # dist/: the old PyPI builds carry author metadata. This script: it
        # holds the very strings it scrubs.
        # docs/: internal working notes, not part of the package.
        if f.startswith(("results/", "docs/", "slurm_logs/", ".github/", "dist/")) \
                or f in ("scripts/make_artifact.py", "scripts/artifact_README.md"):
            continue
        if (ROOT / f).is_file():
            keep.append(Path(f))
    return sorted(set(keep))


def copy(src: Path, rel: Path) -> None:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def dataset_md() -> str:
    reg = {s["name"]: s for s in json.loads((ROOT / "subjects.json").read_text())}
    rows = ["# The 50 projects", "",
            "| # | Project | Repository | Commit | Python | Domain |",
            "|---|---|---|---|---|---|"]
    for i, name in enumerate(SUBJECTS, 1):
        s = reg[name]
        rows.append(f"| {i} | {name} | {s['url']} | `{s.get('commit', '')[:12]}` "
                    f"| {s.get('python_version', '')} | {s.get('domain', '')} |")
    return "\n".join(rows) + "\n"


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)

    for f in tracked_code():
        copy(ROOT / f, f)
    # The selection protocol's records (screening) and the benchmark cases.
    # Not: the benchmark's own resolver virtualenvs, or the raw search logs
    # in screening/work (other people's repositories, not our records).
    for d in ("screening", "data/bench", "data/ft"):
        for p in (ROOT / d).rglob("*"):
            rel = p.relative_to(ROOT)
            if p.is_file() and ".resolvers" not in rel.parts and "work" not in rel.parts:
                copy(p, rel)
    # Every approach's result for the 50 subjects, and the paper's analysis inputs.
    for name in SUBJECTS:
        for p in (ROOT / "results" / name).iterdir():
            if p.is_file() and not SKIP_RESULT.search(p.name):
                copy(p, p.relative_to(ROOT))
    for p in (ROOT / "results").glob("*.json"):
        copy(p, p.relative_to(ROOT))
    for p in (ROOT / "docs" / "paper" / "data").glob("*"):
        copy(p, Path("analysis") / p.name)
    (OUT / "DATASET.md").write_text(dataset_md())
    # The package's own guide on top; the tool's usage guide kept beside it.
    shutil.move(OUT / "README.md", OUT / "TOOL_README.md")
    shutil.copy2(ROOT / "scripts" / "artifact_README.md", OUT / "README.md")

    # Scrub, then scan.
    leaks = []
    for p in OUT.rglob("*"):
        if not p.is_file() or p.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        new = text
        for old, rep in SCRUB:
            new = new.replace(old, rep)
        for pat, rep in SCRUB_RE:
            new = pat.sub(rep, new)
        if p.suffix == ".py":
            for pat, rep in HISTORY_RE:
                new = pat.sub(rep, new)
        if new != text:
            p.write_text(new)
        for m in LEAKS.finditer(new):
            line = new[max(0, m.start() - 60): m.end() + 60].replace("\n", " ")
            leaks.append(f"{p.relative_to(OUT)}: ...{line}...")
    binaries = [p for p in OUT.rglob("*") if p.is_file() and p.suffix not in TEXT_SUFFIXES]

    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"staged {sum(1 for p in OUT.rglob('*') if p.is_file())} files, "
          f"{size / 1e6:.0f} MB, in {OUT}")
    if binaries:
        print(f"{len(binaries)} non-text file(s) not scrubbed, check by hand:")
        for p in binaries[:20]:
            print("   ", p.relative_to(OUT))
    if leaks:
        print(f"\n{len(leaks)} identifying string(s) left; NOT zipping:")
        for l in leaks[:40]:
            print("   ", l)
        return 1

    # The rewritten copy must still pass its own tests.
    print("running the test suite on the staged copy ...", flush=True)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"],
                           cwd=OUT, capture_output=True, text=True)
    print("   " + (tests.stdout.strip().splitlines() or ["(no output)"])[-1])
    if tests.returncode != 0:
        print("tests fail on the staged copy; NOT zipping")
        return 1

    zpath = OUT.parent / "aura-artifact.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, Path("aura-artifact") / p.relative_to(OUT))
    print(f"\nno identifying strings found; wrote {zpath} "
          f"({zpath.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
