"""Code barriers: releases that install cleanly but break the project's tests.

AURA logs every candidate it tries. A barrier is a package that did not reach
its latest release, together with a newer release of it that installed but
introduced new test failures. Crossing it needs a code change, not a different
version, which is the job STRIDE hands to a code agent.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from aura.reqfile import safe_parse_version


@dataclass
class Barrier:
    package: str
    current: str                 # version AURA ended on
    target: str                  # smallest newer release that broke the tests
    plan: dict[str, str]         # the full set of pins AURA tried with it
    new_failures: list[str]      # tests that newly failed at the target
    error_excerpt: str = ""
    releases_behind: int = 0
    tier: str = ""
    crosses_major: bool = False
    tried_versions: list[str] = field(default_factory=list)
    kind: str = "code"           # "code": installs, tests break; "install": blocked
    blocker: str = ""            # install barrier: the declared package in the way

    def label(self) -> str:
        base = f"{self.package} {self.current} -> {self.target}"
        return f"{base} (without {self.blocker})" if self.kind == "install" else base

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _major(version: str):
    parsed = safe_parse_version(version)
    if parsed is None:
        return None
    release = parsed.release or (0,)
    return release[:2] if release[0] == 0 else release[:1]


def _newer(a: str, b: str) -> bool:
    pa, pb = safe_parse_version(a), safe_parse_version(b)
    return pa is not None and pb is not None and pa > pb


def read_log(path: str | Path) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def baseline_from_log(events: list[dict]) -> dict:
    """The tests that already failed before anything was upgraded, and the counts."""
    for e in events:
        if e.get("event") == "attempt" and e.get("tier") == "baseline":
            return {"failing": sorted(e.get("failing") or []),
                    "counts": dict(e.get("tests") or {})}
    for e in events:
        if e.get("event") == "validation" and e.get("phase") == "baseline":
            return {"failing": sorted(e.get("failing") or []), "counts": {},
                    "metrics": dict(e.get("metrics") or {})}
    return {"failing": [], "counts": {}}


def find_barriers(events: list[dict], final_pins: dict[str, str],
                  releases_behind: dict[str, int] | None = None,
                  line_names: dict[int, str] | None = None) -> list[Barrier]:
    """Every package held back by a test failure, with its smallest breaking release.

    ``final_pins`` is the configuration AURA accepted; ``releases_behind`` comes
    from its ``version_age_after`` report and drops packages already at latest.
    """
    releases_behind = releases_behind or {}
    lowered = {k.lower(): v for k, v in final_pins.items()}
    found: dict[str, Barrier] = {}
    for e in events:
        if e.get("event") != "attempt" or not e.get("install_ok") or e.get("accepted"):
            continue
        if not e.get("new_failures"):
            continue
        package = (e.get("package") or "").lower()
        plan = {k.lower(): v for k, v in (e.get("plan") or {}).items()}
        target = plan.get(package)
        current = lowered.get(package)
        if not target or not current or not _newer(target, current):
            continue
        if releases_behind and releases_behind.get(package, releases_behind.get(
                e.get("package"), 1)) == 0:
            continue
        known = found.get(package)
        if known is not None:
            known.tried_versions.append(target)
            if not _newer(known.target, target):
                continue
        tried = (known.tried_versions if known else []) + ([] if known else [target])
        found[package] = Barrier(
            package=package, current=current, target=target, plan=plan,
            new_failures=list(e.get("new_failures") or []),
            error_excerpt=(e.get("error_excerpt") or "")[-4000:],
            releases_behind=int(releases_behind.get(package, 0) or 0),
            tier=e.get("tier") or "",
            crosses_major=_major(target) != _major(current),
            tried_versions=tried,
        )
    for b in found.values():
        b.tried_versions = sorted(set(b.tried_versions), key=safe_parse_version)
    for b in install_barriers(events, lowered, releases_behind, line_names or {}):
        found.setdefault(b.package, b)   # a code barrier for the package wins
    return rank(list(found.values()))


_LINE = re.compile(r"\(line (\d+)\)")


def install_barriers(events: list[dict], final_pins: dict[str, str],
                     releases_behind: dict[str, int],
                     line_names: dict[int, str]) -> list[Barrier]:
    """Packages held back because another declared package will not install
    with their newer release (pip: "Cannot install ... (line N) and P==T").

    The blocker is usually a library the project uses that never supported
    the new release. Version search cannot cross that; a code change that
    stops using the blocker can. The target is the newest release blocked
    only by that one package, where the payoff is largest.
    """
    found: dict[str, Barrier] = {}
    for e in events:
        if e.get("event") != "attempt" or e.get("install_ok"):
            continue
        if e.get("error_class") != "resolution_conflict":
            continue
        package = (e.get("package") or "").lower()
        plan = {k.lower(): v for k, v in (e.get("plan") or {}).items()}
        target, current = plan.get(package), final_pins.get(package)
        if not target or not current or not _newer(target, current):
            continue
        if releases_behind and releases_behind.get(package, 1) == 0:
            continue
        lines = {int(n) for n in _LINE.findall(e.get("error_excerpt") or "")}
        blockers = {line_names[n] for n in lines if n in line_names} - {package} - set(plan)
        if len(blockers) != 1:     # nothing named, or several: not one library to replace
            continue
        blocker = blockers.pop()
        known = found.get(package)
        if known is not None and not _newer(target, known.target):
            continue
        found[package] = Barrier(
            package=package, current=current, target=target, plan=plan, new_failures=[],
            error_excerpt=(e.get("error_excerpt") or "")[-2000:],
            releases_behind=int(releases_behind.get(package, 0) or 0),
            tier=e.get("tier") or "", crosses_major=_major(target) != _major(current),
            tried_versions=[target], kind="install", blocker=blocker)
    return list(found.values())


def smallest_step(barrier: Barrier, index, python_version: str) -> tuple[dict[str, str], list[str]]:
    """The smallest jump across the barrier: the first release of the next major
    series when the barrier crosses one, otherwise the smallest release AURA saw
    break. Packages AURA moved with it are left unpinned, so the installer picks
    versions that match. Returns ``(pins, unpinned)``.
    """
    target = barrier.target
    if barrier.crosses_major:
        history = index.history(barrier.package)
        if history is not None:
            later = [v for v in history.versions
                     if _newer(v, barrier.current) and not _newer(v, barrier.target)
                     and history.usable_on(v, python_version)
                     and not (safe_parse_version(v).is_prerelease)]
            nxt = [v for v in later if _major(v) != _major(barrier.current)]
            if nxt:
                first = min(_major(v) for v in nxt)
                target = min((v for v in nxt if _major(v) == first), key=safe_parse_version)
    partners = [p for p in barrier.plan if p != barrier.package]
    return {barrier.package: target}, partners


def rank(barriers: list[Barrier], policy: str = "most_behind") -> list[Barrier]:
    """Order barriers for the controller.

    most_behind: the package furthest from latest first (largest payoff).
    smallest_jump: the package whose breaking release is closest first.
    """
    if policy == "smallest_jump":
        return sorted(barriers, key=lambda b: (b.crosses_major, b.releases_behind))
    return sorted(barriers, key=lambda b: (-b.releases_behind, b.package))


def barriers_from_run(log_path: str | Path, result_path: str | Path,
                      requirements_path: str | Path | None = None) -> list[Barrier]:
    """Barriers left by one finished AURA run (its .jsonl and .result.json)."""
    from aura.reqfile import ReqFile

    events = read_log(log_path)
    result = json.loads(Path(result_path).read_text())
    pins: dict[str, str] = {}
    line_names: dict[int, str] = {}
    if requirements_path and Path(requirements_path).exists():
        req = ReqFile.load(requirements_path)
        pins = req.pins()
        # AURA's probe files keep the requirements file's line order, so pip's
        # "(line N)" in a conflict names a declared package.
        line_names = {i + 1: l.name for i, l in enumerate(req.lines) if l.name}
    for name, (_, after) in (result.get("updated") or {}).items():
        pins.setdefault(name, after)
    behind = (result.get("version_age_after") or {}).get("per_package") or {}
    return find_barriers(events, pins, {k.lower(): v for k, v in behind.items()},
                         line_names)
