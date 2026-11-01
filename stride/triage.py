"""Is a barrier really about the project's code?

Some barriers break inside installed packages only: fastapi 0.77 pulls a
starlette whose test client calls an API that anyio 4 removed, so every test
fails with "module 'anyio' has no attribute ...", while the project's own code
is never on the stack. A code agent cannot fix that; a pin on the transitive
package can. Triage reads the tracebacks, and when no project frame is
involved, tries such pins before any code change.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from aura.reqfile import safe_parse_version

FRAME = re.compile(r"^(?P<path>[^\s:][^:\n]*\.py):\d+: in ", re.MULTILINE)
MISSING_ATTR = re.compile(r"module '([A-Za-z0-9_\.]+)' has no attribute")
CANNOT_IMPORT = re.compile(r"cannot import name '\w+' from '([A-Za-z0-9_\.]+)'")
SKIP_DISTS = {"pytest", "pluggy", "_pytest", "py", "iniconfig", "setuptools", "pip"}


@dataclass
class Triage:
    kind: str                       # "code" | "dependency" | "unknown"
    project_frames: list[str] = field(default_factory=list)
    package_frames: list[str] = field(default_factory=list)
    implicated: list[str] = field(default_factory=list)


def run_with_tracebacks(python: str, targets: list[str], cwd: str, timeout_s: int = 1800) -> str:
    cmd = [python, "-m", "pytest", *targets, "-q", "--tb=short", "-p", "no:cacheprovider"]
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        return out.stdout + out.stderr
    except subprocess.TimeoutExpired as exc:
        return (exc.stdout or "") if isinstance(exc.stdout, str) else ""


def classify(output: str, test_dirs: list[str]) -> Triage:
    """Project frames are relative paths inside the checkout, outside its tests.
    Absolute paths and ``../`` paths are the interpreter or the environment."""
    project, packages, from_frames = [], [], []
    for m in FRAME.finditer(output):
        path = m.group("path").strip()
        if "site-packages" in path or "dist-packages" in path:
            packages.append(path)
            top = path.split("site-packages/")[-1].split("dist-packages/")[-1].split("/")[0]
            from_frames.append(top.replace(".py", ""))
        elif path.startswith(("/", "..")) or "<" in path:
            continue                                   # stdlib, venv, generated code
        elif not any(path.startswith(t.rstrip("/") + "/") or path == t for t in test_dirs):
            project.append(path)
    named = []
    for pattern in (MISSING_ATTR, CANNOT_IMPORT):
        named += [m.split(".")[0] for m in pattern.findall(output)]
    # The package the error names first, then the innermost frames.
    implicated = [p for p in dict.fromkeys(named + from_frames[::-1])
                  if p and p not in SKIP_DISTS]
    if project:
        kind = "code"
    elif packages:
        kind = "dependency"
    else:
        kind = "unknown"
    return Triage(kind, sorted(set(project)), sorted(set(packages)), implicated)


def _major(v: str):
    p = safe_parse_version(v)
    return p.release[0] if p is not None and p.release else None


def candidate_pins(index, name: str, current: str, python_version: str,
                   limit: int = 4) -> list[str]:
    """Releases to try for a transitive package: the newest of each older major
    series (newest first), then the newest overall if it is not installed."""
    history = index.history(name)
    if history is None or not history.versions:
        return []
    cur = safe_parse_version(current)
    stable = [v for v in history.versions
              if (p := safe_parse_version(v)) is not None and not p.is_prerelease
              and history.usable_on(v, python_version)]
    older: dict[object, str] = {}
    for v in stable:
        if cur is not None and safe_parse_version(v) < cur and _major(v) != _major(current):
            best = older.get(_major(v))
            if best is None or safe_parse_version(v) > safe_parse_version(best):
                older[_major(v)] = v
    picks = sorted(older.values(), key=safe_parse_version, reverse=True)[:limit]
    newest = max(stable, key=safe_parse_version) if stable else None
    if newest and newest != current and newest not in picks:
        picks.append(newest)
    return picks
