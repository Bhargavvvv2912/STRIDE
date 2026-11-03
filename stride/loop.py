"""The STRIDE arms.

stride        version search (AURA) -> one code change across the top barrier ->
              version search again ... until no barrier is crossed or the round
              budget is spent. Starts from a finished AURA run.
pin_latest    the DepBench/RepoRescue setting: from AURA's final state, pin the
              top barrier package (and the packages AURA moved with it) to their
              latest release and ask the agent to repair. No version search.
all_latest    every package at its latest release at once, then repair.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from aura.index import MEASURED_AS_OF, PackageIndex, version_age_report
from aura.reqfile import ReqFile

from evaluation.runner import _env_spec, prepare_workspace, run_aura_arm
from evaluation.subjects import Subject

from stride.barriers import (Barrier, barriers_from_run, baseline_from_log, read_log,
                             smallest_step)
from stride.coders import CodeAgent
from stride.scoped import carry_forward, scoped_search, seed_for
from stride.repair import RepairResult, git, repair_round


SCRATCH_PATTERNS = (".aura_*", "*.aura_dryrun*", "*.stride_env_*", "stride-work-*",
                    ".stride*", "__pycache__/")


def cap_as_of(run: "Run", names) -> dict[str, str]:
    """Pip constraints capping ``names`` at the measurement date, so a free
    resolve can only pick releases out by then. Returns the caps."""
    caps = {n: v for n in names if (v := run.latest(n))}
    path = run.out / "as_of_constraints.txt"
    spec = run.spec()
    existing = Path(spec.build_constraints).read_text() if spec.build_constraints else ""
    path.write_text(existing + "".join(f"{k}<={v}\n" for k, v in sorted(caps.items())))
    run.constraints = str(path)
    run.summary["as_of_caps"] = caps
    return caps


class Run:
    """Shared state for one subject and one arm."""

    def __init__(self, subject: Subject, arm: str, out_root: Path, work_root: Path,
                 aura_dir: Path, agent: CodeAgent, repo_cache: Path | None = None,
                 pip_cache: str | None = None, use_triage: bool = True):
        self.subject, self.arm, self.agent = subject, arm, agent
        self.use_triage = use_triage
        tag = f"{arm}__{agent.label}" + ("" if use_triage else "__notriage")
        self.out = out_root / subject.name / tag
        self.out.mkdir(parents=True, exist_ok=True)
        self.workspace = work_root / subject.name / tag
        self.aura_dir = aura_dir / subject.name
        self.repo_cache, self.pip_cache = repo_cache, pip_cache
        self.index = PackageIndex()
        self.summary: dict = {"subject": subject.name, "arm": arm, "agent": agent.name,
                              "model": getattr(agent, "model", ""), "triage": use_triage,
                              "rounds": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---------------------------------------------------------------- inputs

    def aura_files(self, stem: str = "aura_full_qwen14b") -> tuple[Path, Path, Path]:
        base = self.aura_dir / f"{self.subject.name}__{stem}"
        return (base.with_suffix(".jsonl"), base.with_suffix(".result.json"),
                Path(str(base) + ".requirements.txt"))

    def prepare(self, requirements: Path) -> None:
        prepare_workspace(self.subject, self.workspace, self.repo_cache)
        shutil.copy(requirements, self.workspace / self.subject.requirements_file)
        git(self.workspace, "config", "user.email", "stride@localhost")
        git(self.workspace, "config", "user.name", "STRIDE")
        # AURA's scratch (probe copies, dry runs, env requirement files) lands in
        # the workspace; keep it out of STRIDE's commits so patches count only edits.
        exclude = self.workspace / ".git" / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with exclude.open("a") as fh:
            fh.write("\n".join(SCRATCH_PATTERNS) + "\n")
        git(self.workspace, "add", self.subject.requirements_file)
        git(self.workspace, "commit", "-qm", "STRIDE: start from AURA's final state",
            "--allow-empty")

    constraints: str | None = None     # extra pip constraints (all_latest's date caps)

    def spec(self):
        spec = _env_spec(self.subject, SimpleNamespace(installer="pip"), self.workspace,
                         self.index, self.pip_cache)
        if self.constraints:
            spec.build_constraints = self.constraints
        return spec

    def config(self) -> dict:
        return self.subject.agent_config(self.workspace)

    def pins(self) -> dict[str, str]:
        return ReqFile.load(self.workspace / self.subject.requirements_file).pins()

    def latest(self, name: str) -> str | None:
        return self.index.newest_at(name, MEASURED_AS_OF, self.subject.python_version)

    def repair(self, label: str, pins: dict[str, str], moved: list[str], baseline: dict,
               tag: str, unpin: list[str] | None = None,
               drop: list[str] | None = None) -> RepairResult:
        return repair_round(
            label=label, project=self.subject.name, workspace=self.workspace,
            requirements_file=self.subject.requirements_file, pins=pins, moved=moved,
            config=self.config(), spec=self.spec(), baseline=baseline, agent=self.agent,
            unpin=unpin, index=self.index, python_version=self.subject.python_version,
            out_dir=self.out / tag, tolerance=self.subject.metric_tolerance,
            use_triage=self.use_triage, drop=drop)

    def finish(self, start_pins: dict[str, str]) -> dict:
        end = self.pins()
        # Measure the packages the project declared; pins STRIDE added for
        # transitive packages are reported separately, not scored.
        declared = {k.lower() for k in start_pins}
        scored = {k: v for k, v in end.items() if k.lower() in declared}
        self.summary["added_pins"] = {k: v for k, v in end.items() if k.lower() not in declared}
        # Declared packages no longer installed (an install barrier crossed by
        # replacing them). They leave the score; reported here instead.
        self.summary["dropped"] = sorted(k for k in start_pins
                                         if k.lower() not in {e.lower() for e in end})
        self.summary["age_start"] = version_age_report(self.index, start_pins)
        self.summary["age_end"] = version_age_report(self.index, scored)
        self.summary["pins_end"] = end
        self.summary["patch"] = git(self.workspace, "diff", "--stat",
                                    _first_commit(self.workspace), "HEAD")
        (self.out / "patch_total.diff").write_text(
            git(self.workspace, "diff", _first_commit(self.workspace), "HEAD"))
        self.summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (self.out / "summary.json").write_text(json.dumps(self.summary, indent=2) + "\n")
        return self.summary


def _first_commit(root: Path) -> str:
    """STRIDE's starting commit (AURA's final state), for the total patch."""
    log = git(root, "log", "--format=%H %s").splitlines()
    for line in log:
        if "start from AURA's final state" in line:
            return line.split()[0]
    return "HEAD"


# ------------------------------------------------------------------- arms

def run_stride(run: Run, *, max_rounds: int = 4, barriers_per_round: int = 3,
               aura_arm: str = "aura_full_qwen14b", timeout_s: int = 6 * 3600,
               step: str = "smallest", first: str = "", skip_aura: bool = False,
               rerun: str = "scoped") -> dict:
    """AURA -> code change -> AURA ..., one barrier at a time."""
    log, result, reqs = run.aura_files(aura_arm)
    run.prepare(reqs)
    start = run.pins()
    baseline = baseline_from_log(read_log(log))
    run.summary["baseline_failing"] = baseline["failing"]
    tried: set[str] = set()
    known = barriers_from_run(log, result, reqs)
    run.summary["rerun"] = rerun

    for r in range(1, max_rounds + 1):
        barriers = [b for b in known if f"{b.package}@{b.current}" not in tried]
        if first:   # a barrier named on the command line goes first, in round 1
            barriers.sort(key=lambda b: b.package != first.lower())
            first = ""
        round_info = {"round": r, "barriers_found": [b.label() for b in barriers]}
        crossed: Barrier | None = None
        for b in barriers[:barriers_per_round]:
            tried.add(f"{b.package}@{b.current}")
            if step == "smallest":
                pins, unpin = smallest_step(b, run.index, run.subject.python_version)
            else:                       # the exact plan AURA saw break
                pins, unpin = dict(b.plan), []
            label = f"{b.package} {b.current} -> {pins[b.package]}"
            drop = None
            if b.kind == "install":     # cross by no longer using the blocking package
                drop = [b.blocker]
                unpin = [u for u in unpin if u != b.blocker]
                label += f" (without {b.blocker})"
            outcome = run.repair(label, pins, list(b.plan), baseline,
                                 tag=f"round{r}_{b.package}", unpin=unpin, drop=drop)
            round_info.setdefault("attempts", []).append(asdict(outcome))
            if outcome.accepted:
                crossed = b
                break
        run.summary["rounds"].append(round_info)
        if crossed is None:
            round_info["stop"] = "no barrier crossed"
            break
        if skip_aura:   # smoke test: stop after the first code change
            round_info["stop"] = "skip_aura"
            break

        # Version search again, from the new code and the new pins.
        stem = f"round{r}_aura"
        log = run.out / f"{stem}.jsonl"
        result = run.out / f"{stem}.result.json"
        reqs = run.out / f"{stem}.requirements.txt"
        arm = _aura_arm(aura_arm)
        run.subject.frozen_baseline = str(run.out / f"{stem}.baseline.txt")
        accepted = round_info["attempts"][-1]
        scope = seed_for(crossed.package, crossed.plan, accepted.get("pins"))
        round_info["scope_seed"] = sorted(scope)
        search = scoped_search(scope) if rerun == "scoped" else _nullcontext()
        with search:
            aura = run_aura_arm(run.subject, arm, run.workspace, log, index=run.index,
                                timeout_s=timeout_s, pip_cache=run.pip_cache)
        result.write_text(json.dumps(asdict(aura), indent=2) + "\n")
        shutil.copy(run.workspace / run.subject.requirements_file, reqs)
        git(run.workspace, "add", run.subject.requirements_file)
        git(run.workspace, "commit", "-qm", f"STRIDE: AURA round {r}", "--allow-empty")
        round_info["aura"] = {"outcome": aura.outcome, "updated": aura.updated,
                              "wall_clock_s": aura.wall_clock_s}
        fresh = barriers_from_run(log, result, reqs)
        known = carry_forward(known, fresh, run.pins())
        (run.out / "summary.json").write_text(json.dumps(run.summary, indent=2) + "\n")
    return run.finish(start)


def run_pin_latest(run: Run, *, aura_arm: str = "aura_full_qwen14b", first: str = "") -> dict:
    """Fix the target up front (latest), repair once, never search versions."""
    log, result, reqs = run.aura_files(aura_arm)
    run.prepare(reqs)
    start = run.pins()
    baseline = baseline_from_log(read_log(log))
    barriers = [b for b in barriers_from_run(log, result, reqs) if b.kind == "code"]
    if not barriers:
        run.summary["stop"] = "no barrier"
        return run.finish(start)
    if first:
        barriers.sort(key=lambda x: x.package != first.lower())
    b = barriers[0]
    pins = {name: (run.latest(name) or version) for name, version in b.plan.items()}
    # DepBench-style: the target goes to latest and every other package may
    # follow it (pip resolves, capped at the measurement date). Holding the
    # rest at their pins makes most targets uninstallable before any repair.
    others = [n for n in start if n.lower() not in {k.lower() for k in pins}]
    cap_as_of(run, others)
    label = f"{b.package} {b.current} -> {pins.get(b.package)} (latest)"
    outcome = run.repair(label, pins, list(pins), baseline, tag=f"latest_{b.package}",
                         unpin=others)
    run.summary["rounds"].append({"round": 1, "attempts": [asdict(outcome)]})
    return run.finish(start)


def run_all_latest(run: Run, *, aura_arm: str = "aura_full_qwen14b") -> dict:
    """RepoRescue's setting: drop every pin and let pip pick the newest set that
    installs together (RepoRescue: "a fresh virtual environment with no version
    pins"), then one repair. Pip may only pick releases out by the measurement
    date, the same date every arm is scored against. The project keeps its
    Python version (RepoRescue also moved Python to 3.13)."""
    log, _, reqs = run.aura_files(aura_arm)
    run.prepare(reqs)
    start = run.pins()
    baseline = baseline_from_log(read_log(log))
    cap_as_of(run, start)
    outcome = run.repair("all packages unpinned (pip resolves)", {}, list(start), baseline,
                         tag="all_latest", unpin=list(start))
    run.summary["rounds"].append({"round": 1, "attempts": [asdict(outcome)]})
    return run.finish(start)


ARMS = {"stride": run_stride, "pin_latest": run_pin_latest, "all_latest": run_all_latest}


def _nullcontext():
    import contextlib
    return contextlib.nullcontext()


def _aura_arm(name: str):
    from evaluation.arms import by_name
    return by_name(name)
