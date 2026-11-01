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
    raised_in: dict[str, str] = field(default_factory=dict)   # test -> where it raised


def run_with_tracebacks(python: str, targets: list[str], cwd: str, timeout_s: int = 1800) -> str:
    cmd = [python, "-m", "pytest", *targets, "-q", "--tb=short", "-p", "no:cacheprovider"]
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        return out.stdout + out.stderr
    except subprocess.TimeoutExpired as exc:
        return (exc.stdout or "") if isinstance(exc.stdout, str) else ""


SECTION = re.compile(r"^_{3,} (?:ERROR collecting )?(\S+?) _{3,}$", re.MULTILINE)


def _sections(output: str) -> list[tuple[str, str]]:
    """(test name, traceback text) for each failure pytest printed."""
    marks = list(SECTION.finditer(output))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(output)
        body = output[m.end():end].split("short test summary info")[0]
        out.append((m.group(1), body))
    return out or [("", output)]


def _where(path: str, test_dirs: list[str]) -> str:
    if "site-packages" in path or "dist-packages" in path:
        return "package"
    if path.startswith(("/", "..")) or "<" in path:
        return "other"                                 # stdlib, venv, generated code
    if any(path.startswith(t.rstrip("/") + "/") or path == t for t in test_dirs):
        return "test"
    return "project"


def _top(path: str) -> str:
    top = path.split("site-packages/")[-1].split("dist-packages/")[-1].split("/")[0]
    return top.replace(".py", "")


def classify(output: str, test_dirs: list[str], only: list[str] | None = None,
             stepped: set[str] | None = None) -> Triage:
    """Judge each failure by where its error was raised: the innermost frame
    that is project or package code. A project file merely on the stack (it
    imported the package) does not make the failure the project's: torch 2.0
    removed ``torch._six``, and lightning-bolts 0.5 raises on importing it even
    though the project's ``import pl_bolts`` is on the stack.

    ``only`` restricts the verdict to these tests (the new failures), so a
    failure that predates the step does not decide it.

    An error raised inside a package being stepped (torch, when torch moves)
    is the project calling a changed API: that is code. Only an error raised
    in some other package makes the failure a dependency problem."""
    moving = {n.lower().replace("-", "_") for n in (stepped or set())}
    project, packages, raised_in, implicated = [], [], {}, []
    wanted = {t.split("::")[-1] for t in (only or [])}
    for name, body in _sections(output):
        if wanted and name and name.split("::")[-1] not in wanted:
            continue
        frames = [m.group("path").strip() for m in FRAME.finditer(body)]
        for f in frames:
            kind = _where(f, test_dirs)
            if kind == "project":
                project.append(f)
            elif kind == "package":
                packages.append(f)
        inner = [f for f in frames if _where(f, test_dirs) in ("project", "package")]
        if inner:
            raised_in[name] = inner[-1]
            if _where(inner[-1], test_dirs) == "package":
                implicated.append(_top(inner[-1]))
        named = []
        for pattern in (MISSING_ATTR, CANNOT_IMPORT):
            named += [m.split(".")[0] for m in pattern.findall(body)]
        implicated = named + implicated
    implicated += [_top(p) for p in packages[::-1]]
    implicated = [p for p in dict.fromkeys(implicated) if p and p not in SKIP_DISTS]
    def origin(path: str) -> str:
        where = _where(path, test_dirs)
        if where == "package" and _top(path).lower() in moving:
            return "project"
        return where
    origins = {origin(p) for p in raised_in.values()}
    if "project" in origins:
        kind = "code"
    elif "package" in origins:
        kind = "dependency"
    elif project:
        kind = "code"
    else:
        kind = "unknown"
    return Triage(kind, sorted(set(project)), sorted(set(packages)), implicated, raised_in)


def _series(v: str):
    """Release series: the major, or (0, minor) for 0.x packages, where minor
    releases are the breaking ones."""
    p = safe_parse_version(v)
    if p is None or not p.release:
        return None
    r = p.release
    return (0, r[1] if len(r) > 1 else 0) if r[0] == 0 else (r[0],)


def candidate_pins(index, name: str, current: str, python_version: str,
                   limit: int = 6, as_of=None) -> list[str]:
    """Releases to try for a package a failure was raised in: the newest of each
    newer series (nearest first, since the package may need to catch up with
    the step), then the newest of each older series (newest first)."""
    history = index.history(name)
    if history is None or not history.versions:
        return []
    cur = safe_parse_version(current)
    if cur is None:
        return []
    dates = index.upload_dates(name) if as_of else {}
    stable = [v for v in history.versions
              if (p := safe_parse_version(v)) is not None and not p.is_prerelease
              and history.usable_on(v, python_version)
              and (as_of is None or dates.get(v, "")[:10] <= as_of)]
    best: dict[object, str] = {}
    for v in stable:
        key = _series(v)
        if key == _series(current) and safe_parse_version(v) <= cur:
            continue
        if key not in best or safe_parse_version(v) > safe_parse_version(best[key]):
            best[key] = v
    newer = sorted((v for v in best.values() if safe_parse_version(v) > cur),
                   key=safe_parse_version)
    older = sorted((v for v in best.values() if safe_parse_version(v) < cur),
                   key=safe_parse_version, reverse=True)
    return (newer + older)[:limit]
