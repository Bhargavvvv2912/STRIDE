"""The arms. Every one starts from the same place: the project's working
baseline as AURA established it (``<subject>.baseline.txt``), and is scored the
same way (stride.score). The agent, its retries and the checks on its changes
are the same in every arm (stride.repair).

stride           AURA's version search; when it stops at a code barrier, the
                 agent repairs the code (stride.search). Ours.
agent_alone      a coding agent gets the repository and is asked to upgrade the
                 dependencies as far as it can, changing code as needed. It picks
                 the versions itself (stride.alone).
dependabot_agent one package at a time to its latest release, as Dependabot
                 proposes it; the agent repairs each bump; repeated until a pass
                 changes nothing.
all_latest       RepoRescue's setting: every pin removed, pip picks the newest set
                 that installs, then the agent repairs.
AURA alone (no code changes) is the published AURA run; it is read, not rerun.
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

from stride.barriers import baseline_from_log, read_log
from stride.coders import CodeAgent
from stride.repair import RepairResult, git, repair_round
from stride.score import add_score


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
                 pip_cache: str | None = None, use_triage: bool = True,
                 max_attempts: int = 5, patience: int = 2):
        self.subject, self.arm, self.agent = subject, arm, agent
        self.use_triage = use_triage
        self.max_attempts, self.patience = max_attempts, patience
        self.started_at = time.time()
        self.repairs: list[dict] = []          # every repair, in order
        tag = f"{arm}__{agent.label}" + ("" if use_triage else "__notriage")
        self.out = out_root / subject.name / tag
        self.out.mkdir(parents=True, exist_ok=True)
        self.workspace = work_root / subject.name / tag
        self.aura_dir = aura_dir / subject.name
        self.repo_cache, self.pip_cache = repo_cache, pip_cache
        self.index = PackageIndex()
        self.summary: dict = {"subject": subject.name, "arm": arm, "agent": agent.name,
                              "model": getattr(agent, "model", ""), "triage": use_triage,
                              "rounds": [], "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                              "retry_rule": {"max_attempts": max_attempts,
                                             "patience": patience}}

    # ---------------------------------------------------------------- inputs

    def aura_files(self, stem: str = "aura_full_qwen14b") -> tuple[Path, Path, Path]:
        base = self.aura_dir / f"{self.subject.name}__{stem}"
        return (base.with_suffix(".jsonl"), base.with_suffix(".result.json"),
                Path(str(base) + ".requirements.txt"))

    def baseline_file(self) -> Path:
        """The project's working baseline, as AURA established it."""
        return self.aura_dir / f"{self.subject.name}.baseline.txt"

    def baseline_tests(self, aura_arm: str = "aura_full_qwen14b") -> dict:
        """Tests already failing at the baseline (from AURA's run)."""
        return baseline_from_log(read_log(self.aura_files(aura_arm)[0]))

    def prepare(self, requirements: Path | None = None) -> None:
        requirements = requirements or self.baseline_file()
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
        git(self.workspace, "commit", "-qm", "STRIDE: start", "--allow-empty")

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
        result = repair_round(
            label=label, project=self.subject.name, workspace=self.workspace,
            requirements_file=self.subject.requirements_file, pins=pins, moved=moved,
            config=self.config(), spec=self.spec(), baseline=baseline, agent=self.agent,
            unpin=unpin, index=self.index, python_version=self.subject.python_version,
            out_dir=self.out / tag, tolerance=self.subject.metric_tolerance,
            use_triage=self.use_triage, drop=drop, max_attempts=self.max_attempts,
            patience=self.patience)
        self.repairs.append(asdict(result))
        (self.out / "summary.json").write_text(json.dumps(
            {**self.summary, "repairs": self.repairs}, indent=2) + "\n")
        return result

    def finish(self, start_pins: dict[str, str]) -> dict:
        git(self.workspace, "add", "-A")
        git(self.workspace, "commit", "-qm", "STRIDE: final state", "--allow-empty")
        self.summary["repairs"] = self.repairs
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
        add_score(self.summary)
        self.summary["budget"] = budget(self.summary)
        self.summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (self.out / "summary.json").write_text(json.dumps(self.summary, indent=2) + "\n")
        return self.summary


def budget(summary: dict) -> dict:
    """What the agent was given, over every attempt of every round: the same
    measure for every arm."""
    out = {"agent_calls": 0, "agent_steps": 0, "agent_seconds": 0.0, "repairs": 0}
    attempts = [a for r in summary.get("rounds", []) for a in r.get("attempts", [])]
    attempts += summary.get("repairs", [])
    attempts += summary.get("alone_attempts", [])
    for a in attempts:
        ag = a.get("agent") or {}
        if not ag:
            continue                      # crossed without the agent
        out["repairs"] += 1
        out["agent_calls"] += ag.get("calls", 1 if ag.get("steps") is not None else 0)
        out["agent_steps"] += ag.get("steps") or 0
        out["agent_seconds"] += ag.get("seconds") or 0
    out["agent_seconds"] = round(out["agent_seconds"], 1)
    return out


def _first_commit(root: Path) -> str:
    """The run's starting commit, for the total patch."""
    log = git(root, "log", "--format=%H %s").splitlines()
    for line in log:
        if line.split(" ", 1)[-1] in ("STRIDE: start", "STRIDE: start from AURA's final state"):
            return line.split()[0]
    return "HEAD"


# ------------------------------------------------------------------- arms

def run_stride(run: Run, *, aura_arm: str = "aura_full_qwen14b",
               time_budget_h: float = 10.0, max_tries_per_package: int = 6,
               search_arm: str = "aura_linear_greedy", **_) -> dict:
    """AURA's search with the agent at code barriers (stride.search)."""
    from evaluation.arms import by_name
    from stride.search import Context, stride_search
    run.prepare()
    start = run.pins()
    run.subject.frozen_baseline = str(run.baseline_file())
    ctx = Context(run, max_tries_per_package=max_tries_per_package,
                  time_budget_h=time_budget_h)
    log = run.out / "search.jsonl"
    with stride_search(ctx):
        aura = run_aura_arm(run.subject, by_name(search_arm), run.workspace, log,
                            index=run.index, timeout_s=int(time_budget_h * 3600),
                            pip_cache=run.pip_cache)
    run.summary["search"] = {"outcome": aura.outcome, "updated": aura.updated,
                             "wall_clock_s": aura.wall_clock_s, "reason": aura.reason}
    run.summary["tries"] = ctx.tries
    return run.finish(start)


def run_dependabot_agent(run: Run, *, aura_arm: str = "aura_full_qwen14b",
                         passes: int = 3, **_) -> dict:
    """Dependabot's proposals, one package at a time to its latest release (as of
    the measurement date), each repaired by the agent. A bump that does not
    install with the other pins is skipped, as Dependabot's pull request would
    fail. Passes repeat until nothing more moves."""
    run.prepare()
    start = run.pins()
    baseline = run.baseline_tests(aura_arm)
    for p in range(1, passes + 1):
        moved = 0
        for name in sorted(run.pins(), key=str.lower):
            current, latest = run.pins().get(name), run.latest(name)
            if not current or not latest or not _newer(latest, current):
                continue
            result = run.repair(f"{name} {current} -> {latest}", {name: latest}, [name],
                                baseline, tag=f"pass{p}_{name}")
            moved += bool(result.accepted)
        run.summary["rounds"].append({"round": p, "moved": moved})
        if not moved:
            break
    return run.finish(start)


def run_all_latest(run: Run, *, aura_arm: str = "aura_full_qwen14b", **_) -> dict:
    """RepoRescue's setting: drop every pin and let pip install the newest set
    that works together (no package below its baseline version, nothing newer
    than the measurement date), then the agent repairs."""
    run.prepare()
    start = run.pins()
    baseline = run.baseline_tests(aura_arm)
    cap_as_of(run, start)
    run.repair("all packages unpinned (pip resolves)", {}, list(start), baseline,
               tag="all_latest", unpin=list(start))
    return run.finish(start)


def run_agent_alone(run: Run, *, aura_arm: str = "aura_full_qwen14b", **_) -> dict:
    from stride.alone import agent_alone
    run.prepare()
    start = run.pins()
    caps = cap_as_of(run, start)
    agent_alone(run, run.baseline_tests(aura_arm), caps)
    return run.finish(start)


def _newer(a: str, b: str) -> bool:
    from aura.reqfile import safe_parse_version
    va, vb = safe_parse_version(a), safe_parse_version(b)
    return bool(va and vb and va > vb)


ARMS = {"stride": run_stride, "agent_alone": run_agent_alone,
        "dependabot_agent": run_dependabot_agent, "all_latest": run_all_latest}
