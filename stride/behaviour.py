"""Behaviour checks on accepted patches, beyond "the tests pass".

Run after a grid, the same way for every arm; acceptance is not changed.

diff coverage      Which of the lines a patch changed do the project's own
                   tests execute? A change the tests never run is not checked
                   by them, however green they are.
differential run   A probe calls project code (for example the method that
                   builds the learning-rate scheduler) and returns plain
                   values. It runs twice: original code in AURA's environment,
                   patched code in the upgraded one. Different values mean the
                   patch changed what the project does. Probes are written per
                   subject (stride/probes/<subject>.py, a ``probe()`` function),
                   aimed at code the agents replaced, such as a dropped
                   library's function.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

COVERAGE = "coverage==7.6.1"          # supports Python 3.8-3.13
PROBES = Path(__file__).parent / "probes"
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _is_source(path: str) -> bool:
    parts = Path(path).parts
    return (path.endswith(".py") and not Path(path).name.startswith("test_")
            and not any(p in ("tests", "test") or p.startswith((".aura", ".stride"))
                        or p.startswith("stride-work-") for p in parts))


def changed_lines(diff: str) -> dict[str, set[int]]:
    """Lines a patch added or modified, by file (new numbering), source only."""
    out: dict[str, set[int]] = {}
    current, line = None, 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            name = raw[4:].strip()
            current = name[2:] if name.startswith("b/") else None
            if current and not _is_source(current):
                current = None
            continue
        m = _HUNK.match(raw)
        if m:
            line = int(m.group(1))
            continue
        if current is None or raw.startswith(("---", "diff ", "index ")):
            continue
        if raw.startswith("+"):
            out.setdefault(current, set()).add(line)
            line += 1
        elif raw.startswith(" "):
            line += 1
    return {k: v for k, v in out.items() if v}


def diff_coverage(python: str, workspace: Path, targets: list[str],
                  changed: dict[str, set[int]]) -> dict:
    """Run the tests under coverage and score the changed lines."""
    install = subprocess.run([python, "-m", "pip", "install", "-q", COVERAGE],
                             capture_output=True, text=True)
    if install.returncode != 0:
        return {"error": "could not install coverage: " + install.stderr[-500:]}
    data = workspace / ".stride_coverage"
    subprocess.run([python, "-m", "coverage", "run", f"--data-file={data}",
                    f"--source={workspace}", "--omit=*/.aura*/*,*/tests/*,*/test/*",
                    "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider"],
                   cwd=workspace, capture_output=True, text=True, timeout=3600)
    report = workspace / ".stride_coverage.json"
    subprocess.run([python, "-m", "coverage", "json", f"--data-file={data}", "-o",
                    str(report)], cwd=workspace, capture_output=True, text=True)
    if not report.exists():
        return {"error": "coverage produced no report"}
    files = json.loads(report.read_text()).get("files", {})
    by_file = {}
    total = covered = 0
    for rel, lines in changed.items():
        entry = files.get(rel) or files.get(str(workspace / rel)) or {}
        executed = set(entry.get("executed_lines", []))
        executable = executed | set(entry.get("missing_lines", []))
        mine = lines & executable          # comments and blank lines do not count
        hit = mine & executed
        by_file[rel] = {"changed": len(mine), "covered": len(hit),
                        "uncovered_lines": sorted(mine - hit)}
        total += len(mine)
        covered += len(hit)
    return {"changed_lines": total, "covered_lines": covered,
            "coverage": round(covered / total, 3) if total else None, "files": by_file}


PROBE_RUNNER = r"""
import json, runpy, sys
sys.path.insert(0, ".")
mod = runpy.run_path(sys.argv[1])
try:
    value = mod["probe"]()
except Exception as exc:
    value = {"error": type(exc).__name__ + ": " + str(exc)[:300]}
print("STRIDE_PROBE " + json.dumps(value))
"""


def run_probe(python: str, workspace: Path, probe: Path) -> object:
    out = subprocess.run([python, "-c", PROBE_RUNNER, str(probe)], cwd=workspace,
                         capture_output=True, text=True, timeout=1800)
    for line in out.stdout.splitlines()[::-1]:
        if line.startswith("STRIDE_PROBE "):
            return json.loads(line[len("STRIDE_PROBE "):])
    return {"error": "probe printed nothing: " + (out.stderr or out.stdout)[-500:]}


def same(a, b, rel: float = 1e-6, abs_: float = 1e-12) -> bool:
    """Equal values, with a float tolerance, through lists and dicts."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], rel, abs_) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, rel, abs_) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
        except (TypeError, ValueError):
            return False
    return a == b


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def max_relative_difference(a, b) -> float | None:
    """Largest relative gap between matching numbers (None if shapes differ)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return None
        parts = [max_relative_difference(a[k], b[k]) for k in a]
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return None
        if a and all(_number(x) and _number(y) for x, y in zip(a, b)):
            # A series (say, the learning rate per epoch): the gap relative to
            # the series' own size, so a value near zero does not read as 100%.
            scale = max(max(abs(x) for x in a), max(abs(y) for y in b))
            return 0.0 if scale == 0 else max(abs(x - y) for x, y in zip(a, b)) / scale
        parts = [max_relative_difference(x, y) for x, y in zip(a, b)]
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        scale = max(abs(a), abs(b))
        return 0.0 if scale == 0 else abs(a - b) / scale
    else:
        return 0.0 if a == b else None
    if any(x is None for x in parts):
        return None
    return max(parts, default=0.0)


def first_difference(a, b, path: str = "") -> str:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return f"{path}/{k}: only in {'new' if k not in a else 'old'}"
            if not same(a[k], b[k]):
                return first_difference(a[k], b[k], f"{path}/{k}")
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            if not same(x, y):
                return first_difference(x, y, f"{path}[{i}]")
    return f"{path}: old={str(a)[:120]} new={str(b)[:120]}"


def probe_for(subject: str) -> Path | None:
    path = PROBES / f"{subject.replace('-', '_')}.py"
    return path if path.exists() else None
