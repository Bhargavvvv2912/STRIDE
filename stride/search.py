"""STRIDE's search: AURA's version search, with a coding agent at code barriers.

For each package AURA first runs its own search (reverse linear scan, then
greedy co-resolution; no LLM tier). That finds the newest version that works
with no code change. When newer versions exist above it, the agent is given a
chance to make them work:

  newest release of the newest major series        agent repair
    fails -> first release of that series          agent repair
      fails -> the whole series is out; next series down
      works -> binary search between the two for the newest the agent can fix

Each try installs the version first:
  - it resolves as is                         -> install, tests, agent if needed
  - pip names declared packages in a conflict -> free them, never below their
                                                 current version, and resolve again
                                                 (greedy, for installability)
  - one declared package still blocks it      -> drop it; the agent replaces its uses
  - anything else (only undeclared packages, Python, build errors) -> skip

``aura/`` is not modified: a subclass overrides AURA's per-package step and is
swapped into AURA's runner for the length of one STRIDE run, as the scoped
rerun does.
"""
from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from pathlib import Path

from aura.agent import AuraAgent
from aura.environment import dry_run_session
from aura.index import safe_parse_version
from aura.planner import extract_conflicting_packages
from aura.reqfile import ReqFile
from aura.telemetry import Event

from stride.barriers import _LINE
from stride.repair import canonical, git, unpinned


def series_key(version: str) -> tuple:
    """Major series; for 0.x, the minor is the series (0.4 -> 0.5 breaks)."""
    v = safe_parse_version(version)
    if v is None:
        return (version,)
    rel = v.release + (0, 0)
    return (0, rel[1]) if rel[0] == 0 else (rel[0],)


def _newer(a: str, b: str) -> bool:
    va, vb = safe_parse_version(a), safe_parse_version(b)
    return bool(va and vb and va > vb)


def plan_tries(above: list[str]) -> list[list[str]]:
    """Versions above AURA's answer grouped by series, newest series first,
    each ascending."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for v in above:
        groups[series_key(v)].append(v)
    ordered = sorted(groups.values(),
                     key=lambda vs: safe_parse_version(vs[0]), reverse=True)
    return [sorted(vs, key=safe_parse_version) for vs in ordered]


def search_series(versions: list[str], attempt) -> object | None:
    """Newest, then first, then binary search between. ``attempt(v)`` returns a
    result (truthy) when the agent made ``v`` work. Returns the newest success."""
    top = attempt(versions[-1])
    if top or len(versions) == 1:
        return top
    best = attempt(versions[0])
    if not best:
        return None
    lo, hi = 0, len(versions) - 1          # lo works, hi failed
    while hi - lo > 1:
        mid = (lo + hi) // 2
        got = attempt(versions[mid])
        if got:
            lo, best = mid, got
        else:
            hi = mid
    return best


class Context:
    """What the agent hook needs from STRIDE's run."""

    def __init__(self, run, *, max_tries_per_package: int = 6,
                 time_budget_h: float = 10.0, max_greedy_rounds: int = 6):
        self.run = run
        self.max_tries = max_tries_per_package
        self.deadline = time.time() + time_budget_h * 3600
        self.max_greedy_rounds = max_greedy_rounds
        self.tries: list[dict] = []           # every agent try, for the summary
        self.counter = 0
        # Tries that failed, keyed by what they depended on: the version, the
        # pins of the packages it moved, and the code. A later AURA pass asks
        # the same question again unless one of those changed.
        self.failed: set[tuple] = set()

    @property
    def out_of_time(self) -> bool:
        return time.time() > self.deadline


def make_stride_agent(ctx: Context):
    run = ctx.run

    class StrideAgent(AuraAgent):

        def _escalate(self, candidate, baseline):
            found = super()._escalate(candidate, baseline)
            if ctx.out_of_time or self.out_of_time:
                return found
            try:
                better = self._with_code(candidate, baseline, found)
            except Exception as exc:     # a crashed try must not end AURA's run
                self.log.emit(Event.NOTE, note="stride_try_crashed",
                              package=candidate.name, error=f"{type(exc).__name__}: {exc}")
                better = None
            return better or found

        # -------------------------------------------------------------- tries

        def _with_code(self, candidate, baseline: ReqFile, found):
            name = candidate.name
            floor = (found or {}).get(name) or candidate.current
            usable, error = self._candidate_versions(candidate)
            if error or not usable:
                return None
            above = [v for v in usable if floor and _newer(v, floor)]
            if not above:
                return None
            # The working tree must hold exactly AURA's current state, committed,
            # so a failed try rolls back to it.
            current = ReqFile.from_text(baseline.to_text())
            if found:
                current = current.with_pins(found)
            current.save(self.requirements_path)
            git(self.workdir, "add", "-A", "--", run.subject.requirements_file)
            git(self.workdir, "commit", "-qm", f"STRIDE: search state before {name}",
                "--allow-empty")
            budget = [ctx.max_tries]
            accepted: dict | None = None

            def attempt(version: str):
                nonlocal accepted
                if budget[0] <= 0 or ctx.out_of_time or self.out_of_time:
                    return None
                budget[0] -= 1
                got = self._try(name, version, candidate.current)
                if got:
                    accepted = got
                return got

            for versions in plan_tries(above):
                if search_series(versions, attempt):
                    break
                if budget[0] <= 0:
                    break
            if not accepted:
                return None
            # The code changed: earlier verdicts were about other code.
            self._probe_cache.clear()
            on_disk = ReqFile.load(self.requirements_path)
            # AURA applies the pins it is handed and counts them as progress; the
            # rest of the file (pins pip chose, packages dropped) is synced here.
            before = baseline.pins()
            baseline.lines[:] = on_disk.lines
            for pkg in accepted:
                if pkg in before:
                    baseline.repin(pkg, before[pkg])
            return accepted

        def _resolve(self, reqfile: ReqFile) -> tuple[bool, str]:
            with dry_run_session(spec=self.env_spec,
                                 env_path=self.workdir / ".aura_dryrun_stride",
                                 workdir=self.workdir) as resolve:
                return resolve(reqfile)

        def _try(self, name: str, version: str, was: str) -> dict | None:
            """Make ``name==version`` install, then let the agent repair the code.
            Returns the accepted pins, or None."""
            start = ReqFile.load(self.requirements_path)
            declared = {canonical(n) for n in start.names()}
            pins = {name: version}
            freed: list[str] = []
            ok, log = self._resolve(start.with_pins(pins))
            for _ in range(ctx.max_greedy_rounds):
                if ok:
                    break
                blockers = [b for b in extract_conflicting_packages(log, known=declared)
                            if b != canonical(name) and b not in freed]
                if not blockers:
                    break
                freed += blockers
                current = {canonical(k): v for k, v in start.pins().items()}
                plan = unpinned(start.with_pins(pins), freed,
                                floors={b: current[b] for b in freed if b in current})
                ok, log = self._resolve(plan)
            # What this try's verdict depends on: the code (not the requirements
            # file), and the pins of the packages it moves. Asked again only if
            # one of those changed since it failed.
            tree = git(self.workdir, "ls-tree", "-r", "HEAD").splitlines()
            code = hash(tuple(l for l in tree
                              if not l.endswith("\t" + run.subject.requirements_file)))
            involved = start.pins()
            key = (canonical(name), version, code,
                   tuple(sorted((b, involved.get(b, "")) for b in freed)))
            if key in ctx.failed:
                self.log.emit(Event.NOTE, note="stride_try_known", package=name,
                              version=version)
                return None
            drop = None
            if not ok:
                drop = self._single_blocker(start, log, name)
                if drop is None:
                    kind = "project_metadata" if self._project_blocks(log) else "unresolvable"
                    ctx.failed.add(key)
                    self.log.emit(Event.NOTE, note="stride_skip", package=name,
                                  version=version, reason=kind, error_excerpt=log[-1500:])
                    ctx.tries.append({"package": name, "version": version,
                                      "outcome": f"skipped ({kind})"})
                    return None
            ctx.counter += 1
            label = f"{name} {was} -> {version}" + (f" (without {drop})" if drop else "")
            result = run.repair(label, pins, [name] + freed, self._baseline_dict(),
                                tag=f"try{ctx.counter:02d}_{name}_{version}",
                                unpin=freed, drop=[drop] if drop else None)
            ctx.tries.append({"package": name, "version": version, "label": label,
                              "accepted": result.accepted, "reason": result.reason,
                              "calls": (result.agent or {}).get("calls", 0),
                              "dropped": result.dropped, "freed": freed,
                              "added_test_deps": result.added_test_deps,
                              "tests_changed": result.tests_changed,
                              "test_aware": len(result.test_aware)})
            self.log.emit(Event.NOTE, note="stride_try", package=name, version=version,
                          accepted=result.accepted, reason=result.reason, freed=freed,
                          dropped=result.dropped)
            if not result.accepted:
                if result.reason != "agent_error":     # infrastructure: may retry
                    ctx.failed.add(key)
                return None
            return {k: v for k, v in result.pins.items()}

        def _project_blocks(self, log: str) -> bool:
            """pip's conflict report names the project itself as a source (its own
            install_requires caps the package). Only for installable projects."""
            from aura.planner import parse_conflict_report
            if not self.config.get("IS_INSTALLABLE_PACKAGE"):
                return False
            report = parse_conflict_report(log) or {}
            own = canonical(self.config.get("PROJECT_NAME") or run.subject.name)
            return any(c["source"] == own for c in report.get("constraints", []))

        def _single_blocker(self, reqfile: ReqFile, log: str, name: str) -> str | None:
            """One declared package that pip's conflict points at (by its line in
            the requirements file): the lightning-bolts case. Else None."""
            lines = {i + 1: l.name for i, l in enumerate(reqfile.lines) if l.name}
            named = {lines[int(n)] for n in _LINE.findall(log) if int(n) in lines}
            named -= {name}
            return next(iter(named)) if len(named) == 1 else None

        def _baseline_dict(self) -> dict:
            b = self.validator.baseline
            if b is None:
                return {"failing": []}
            return {"failing": sorted(b.failing), "metrics": dict(b.metrics or {}),
                    "counts": dict(getattr(b, "counts", {}) or {})}

    return StrideAgent


@contextlib.contextmanager
def stride_search(ctx: Context):
    """Within this block, evaluation.runner.run_aura_arm runs STRIDE's search."""
    import evaluation.runner as runner
    original = runner.AuraAgent
    runner.AuraAgent = make_stride_agent(ctx)
    try:
        yield
    finally:
        runner.AuraAgent = original
