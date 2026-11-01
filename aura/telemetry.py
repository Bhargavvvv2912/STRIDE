"""Structured run logging: every table in the paper is generated from this.

v1 had no telemetry at all -- no timing, no token counting, no per-attempt
record -- which is why cost, latency, LLM first-attempt success and
conflict-detection precision could not be reported. Every event here is one JSON
object on one line, appended as it happens, so a run that crashes or is killed
by a timeout still leaves a usable partial record.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import threading
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class Outcome(str, Enum):
    """How a run ended. v1 could only ever report success."""

    MODERNIZED = "modernized"                # at least one package moved forward
    KEPT_BASELINE = "kept_baseline"          # nothing moved; baseline still passes
    FAILED = "failed"                        # could not establish/keep a passing env
    TRUNCATED_PASSES = "truncated_passes"    # hit MAX_RUN_PASSES with changes pending
    TRUNCATED_TIMEOUT = "truncated_timeout"  # hit the wall-clock budget
    ERROR = "error"                          # crashed


class Tier(str, Enum):
    BASELINE = "baseline"      # the project's own pins, before any change
    LINEAR = "linear"          # Reverse Linear Scan
    GREEDY = "greedy"          # Greedy Co-Resolution
    LLM = "llm"                # LLM-guided planning


class Event(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    PASS_START = "pass_start"
    PASS_END = "pass_end"
    PRIORITIZATION = "prioritization"
    ATTEMPT = "attempt"
    LLM_CALL = "llm_call"
    CONFLICT_DETECTION = "conflict_detection"
    VALIDATION = "validation"
    BEHAVIOR_CHECK = "behavior_check"
    CONSTRAINT_GRAPH = "constraint_graph"   # declared coupling read from metadata
    COVERAGE = "coverage"                   # how much the oracle actually executed
    ROUTING = "routing"                     # failure class chose the next tier
    BATCH = "batch"                         # k candidates probed concurrently
    PROPOSAL = "proposal"                   # one per-dependency bump, judged alone
    NOTE = "note"


# USD per 1M tokens. Update alongside the model table in the paper's appendix.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Self-hosted open models: no per-token price; cost is reported as GPU-hours
    # separately, so these are recorded at zero and flagged local=True.
    "qwen2.5-coder-32b-instruct": (0.0, 0.0),
    "llama-3.3-70b-instruct": (0.0, 0.0),
    "deepseek-coder-v2-lite-instruct": (0.0, 0.0),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens / 1e6) * inp + (completion_tokens / 1e6) * out


@dataclass
class LLMUsage:
    """Token accounting for a single model call."""

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    ok: bool = True
    error: str = ""

    def finalize(self) -> "LLMUsage":
        self.cost_usd = estimate_cost(self.model, self.prompt_tokens, self.completion_tokens)
        return self


@dataclass
class RunConfig:
    """Everything that distinguishes one experimental arm from another.

    Recorded verbatim at run start so an arm can be reconstructed from its log
    without consulting any external notes.
    """

    repo: str = ""
    arm: str = "aura"
    tiers: tuple[str, ...] = (Tier.LINEAR.value, Tier.GREEDY.value, Tier.LLM.value)
    ordering: str = "spread"
    ordering_seed: int = 0
    llm_model: str = ""
    llm_temperature: float = 0.0
    llm_explanation_first: bool = True
    llm_prompt_style: str = "full"
    installer: str = "pip"
    python_version: str = ""               # the version the subject declares
    commit_date: str = ""                  # YYYY-MM-DD of the pinned commit, the
                                           # date a broken baseline is rebuilt at
    repair_baseline: bool = True           # reconstruct a baseline that no longer
                                           # installs; deterministic, so every arm
                                           # starts from the identical file
    python_executable: str = ""            # the interpreter its venvs are built from
    max_passes: int = 20                   # a pass that changes nothing ends the
                                           # run, so this is a ceiling, not a target
    timeout_s: int = 6 * 3600
    validation_mode: str = "baseline_relative"
    install_cache: bool = True
    build_isolation: bool = True
    batching: bool = True                  # probe k candidates concurrently
    batch_k: int = 0                       # >0 pins k; 1 is the determinism control
    measure_coverage: bool = False         # instrument validation to size the oracle
    constraint_graph: bool = True          # read declared Requires-Dist coupling
    routing: bool = True                   # let the failure class pick the next tier
    # Off. After the search stops, a package still far behind can sometimes be
    # advanced by stepping one of its blockers back -- but the candidates are
    # ordered by architectural spread, so the blocker is usually the more
    # central package of the two. Trading a framework back to advance a
    # transitive helper inverts the ordering the agent uses to decide what
    # matters, and no measure of staleness can see that, since it treats every
    # package as equally important. Kept behind a flag as an ablation.
    regroup: bool = False
    regroup_packages: int = 8              # how many still-behind packages to try
    regroup_max_steps_back: int = 3        # a trade is a few releases, not a
                                           # version line given up
    notes: str = ""


def _interpreter_version(executable: str) -> str:
    """The ``X.Y.Z`` of an interpreter, asked rather than assumed."""
    if not executable or executable == sys.executable:
        return sys.version.split()[0]
    try:
        out = subprocess.run(
            [executable, "-c",
             "import sys;print('.'.join(str(p) for p in sys.version_info[:3]))"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_rev(path: str = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


class RunLog:
    """Append-only JSONL event log for one AURA run (one repo, one arm)."""

    def __init__(self, path: str | Path, config: RunConfig,
                 console_path: str | Path | None = None, echo: bool = False):
        from aura.console import ConsoleWriter

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.run_id = uuid.uuid4().hex[:12]
        self.t0 = time.time()
        self._counters: dict[str, float] = {}
        # Batched probing writes from several threads at once; without this the
        # JSONL interleaves mid-line and counters lose increments.
        self._lock = threading.Lock()
        # A re-run gets a clean file. Appending would merge two runs into one
        # log, and since the reader takes the first run_start and the last
        # run_end, the merged result silently doubles every attempt and reports
        # the whole thing under one config. The previous run is archived rather
        # than deleted, and the ``.bak`` suffix keeps it out of the results glob.
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        if self.path.exists() and self.path.stat().st_size:
            self.path.replace(self.path.with_suffix(self.path.suffix + f".{stamp}.bak"))
        readable = (Path(console_path) if console_path is not None
                    else self.path.with_suffix(".log"))
        if readable.exists() and readable.stat().st_size:
            readable.replace(readable.with_suffix(readable.suffix + f".{stamp}.bak"))
        self._fh = self.path.open("w", encoding="utf-8")
        # Human-readable stream, written from the same events so the two files
        # can never disagree. Defaults to <log>.log beside the JSONL.
        if console_path is None:
            console_path = self.path.with_suffix(".log")
        self.console = ConsoleWriter(console_path, echo=echo)
        self.emit(Event.RUN_START, config=asdict_safe(config), env=self._environment())

    # ------------------------------------------------------------------ core

    def _environment(self) -> dict[str, Any]:
        """Captured so 'which Python / which machine' is never an open question."""
        target = getattr(self.config, "python_executable", "") or sys.executable
        return {
            # Two different interpreters, and conflating them has bitten before:
            # the harness may run on one Python while the subject is validated on
            # another. "python" is the one the results are about.
            "python": _interpreter_version(target),
            "python_executable": target,
            "harness_python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "aura_commit": _git_rev(str(Path(__file__).resolve().parent.parent)),
            "repo_commit": _git_rev("."),
            "cpu_count": os.cpu_count(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def emit(self, event: Event | str, **fields: Any) -> None:
        record = {
            "run_id": self.run_id,
            "repo": self.config.repo,
            "arm": self.config.arm,
            "event": event.value if isinstance(event, Event) else event,
            "t": round(time.time() - self.t0, 3),
            **fields,
        }
        line = json.dumps(record, default=str) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()  # survive a kill -9 at the timeout boundary
            self.console.handle(record["event"], record)

    def bump(self, key: str, amount: float = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + amount

    @property
    def counters(self) -> dict[str, float]:
        return dict(self._counters)

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    # --------------------------------------------------------- typed helpers

    @contextmanager
    def timed(self, event: Event | str, **fields: Any) -> Iterator[dict]:
        """Time a block and emit one event with its duration and any extras."""
        extra: dict[str, Any] = {}
        start = time.time()
        try:
            yield extra
        finally:
            self.emit(event, duration_s=round(time.time() - start, 3), **fields, **extra)

    def attempt(
        self, *, package: str, tier: Tier | str, plan: dict[str, str],
        install_ok: bool, validate_ok: bool, duration_s: float,
        error_class: str = "", error_excerpt: str = "", accepted: bool = False,
        **details: Any,
    ) -> None:
        """One install-and-validate cycle -- the unit of cost in this system.

        ``details`` carries the per-stage breakdown (test counts, newly failing
        tests, validation time) from :func:`aura.environment.attempt_details`.
        """
        self.bump("attempts")
        self.bump(f"attempts_{tier.value if isinstance(tier, Tier) else tier}")
        self.bump("install_seconds", duration_s)
        self.emit(
            Event.ATTEMPT, package=package,
            tier=tier.value if isinstance(tier, Tier) else tier,
            plan=plan, install_ok=install_ok, validate_ok=validate_ok,
            accepted=accepted, duration_s=round(duration_s, 3),
            error_class=error_class, error_excerpt=error_excerpt[:2000],
            **details,
        )

    def llm_call(
        self, usage: LLMUsage, *, purpose: str, package: str = "",
        iteration: int = 0, proposed: list[str] | None = None,
        rejected: list[dict] | None = None, resolvable: bool = True,
        explanation: str = "", inputs: dict | None = None,
        reason: str = "", raw: str = "",
    ) -> None:
        """One model call, with the plan it produced and what validation rejected."""
        self.bump("llm_calls")
        self.bump("prompt_tokens", usage.prompt_tokens)
        self.bump("completion_tokens", usage.completion_tokens)
        self.bump("llm_cost_usd", usage.cost_usd)
        self.bump("llm_seconds", usage.latency_s)
        if rejected:
            self.bump("llm_rejected_versions", len(rejected))
        self.emit(
            Event.LLM_CALL, purpose=purpose, package=package, iteration=iteration,
            proposed=proposed or [], rejected=rejected or [], resolvable=resolvable,
            explanation=explanation[:1000],
            # Why the model stopped (when it declared the conflict unresolvable)
            # and exactly what it said, so every decision can be audited.
            reason=reason[:1000], raw_response=raw[:4000],
            # Recorded so this decision can be replayed against another backend.
            inputs=inputs or {}, **asdict(usage),
        )

    def conflict_detection(
        self, *, package: str, extracted: list[str], installed: list[str],
        source: str = "regex", reported: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> None:
        """What the log parser pulled out, and how much of it was even real.

        ``in_env`` is the subset of extracted names that actually appear in the
        environment -- the precision denominator for RQ on component quality.
        """
        in_env = sorted(set(extracted) & set(installed))
        self.emit(
            Event.CONFLICT_DETECTION, package=package, source=source,
            extracted=sorted(extracted), extracted_n=len(extracted),
            in_env=in_env, in_env_n=len(in_env),
            precision=round(len(in_env) / len(extracted), 4) if extracted else None,
            # "Is the name installed?" is a weak test -- every requirement name
            # passes it. When pip printed its own conflict report, precision is
            # measured against the packages it actually named.
            reported=sorted(reported) if reported is not None else None,
            precision_vs_report=(
                round(len(set(extracted) & set(reported)) / len(extracted), 4)
                if reported is not None and extracted else None),
            constraints=constraints or [],
        )

    def validation(
        self, *, phase: str, passed: bool, failing: list[str],
        new_failures: list[str], metrics: dict[str, Any] | None = None,
        duration_s: float = 0.0, mode: str = "",
    ) -> None:
        self.emit(
            Event.VALIDATION, phase=phase, passed=passed,
            failing=sorted(failing)[:200], failing_n=len(failing),
            new_failures=sorted(new_failures)[:200], new_failures_n=len(new_failures),
            metrics=metrics or {}, duration_s=round(duration_s, 3), mode=mode,
        )

    def behavior_check(
        self, *, baseline: dict[str, Any], candidate: dict[str, Any],
        drifted: list[str], tolerance: float, verdict: str,
    ) -> None:
        """Numeric comparison of what the project *produced*, not just its exit code."""
        self.emit(
            Event.BEHAVIOR_CHECK, baseline=baseline, candidate=candidate,
            drifted=sorted(drifted), drifted_n=len(drifted),
            tolerance=tolerance, verdict=verdict,
        )

    def finish(
        self, outcome: Outcome, *, updated: dict[str, tuple[str, str]] | None = None,
        unchanged: list[str] | None = None, reason: str = "",
        version_age: dict[str, Any] | None = None,
        baseline: dict[str, Any] | None = None,
    ) -> None:
        self.emit(
            Event.RUN_END, outcome=outcome.value, reason=reason,
            updated={k: list(v) for k, v in (updated or {}).items()},
            updated_n=len(updated or {}), unchanged=sorted(unchanged or []),
            wall_clock_s=round(self.elapsed, 2), counters=self.counters,
            version_age=version_age or {},
            # How the starting point came about: which rung of the repair
            # ladder produced it, its digest (so "every arm started from the
            # same file" is checkable), and what the repair itself moved.
            baseline=baseline or {},
        )
        self._fh.close()
        self.console.close()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
        self.console.close()


def asdict_safe(obj: Any) -> dict:
    try:
        return asdict(obj)
    except TypeError:
        return dict(obj.__dict__)


def read_log(path: str | Path) -> list[dict]:
    """Read a JSONL log, tolerating a truncated final line from a killed run."""
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_logs(root: str | Path, pattern: str = "**/*.jsonl") -> list[dict]:
    """Read every run log under ``root`` into one flat event list."""
    events: list[dict] = []
    for p in sorted(Path(root).glob(pattern)):
        events.extend(read_log(p))
    return events
