"""The AURA agent: prioritised, execution-validated search over pinned versions.

Differences from v1 that change what the numbers mean:

* **A run can fail.** v1's ``attempt_update_with_healing`` had no ``return
  False`` on any path, so the failure branch was unreachable dead code and
  "AURA fails on 0 repositories" was a property of the type signature rather
  than a measurement. Outcomes here are modernized / kept_baseline / failed /
  truncated_passes / truncated_timeout / error, counted separately.
* **Truncation is not convergence.** Hitting MAX_RUN_PASSES with changes still
  landing is recorded as ``truncated_passes``, not as a converged result.
* **Tiers are independently switchable.** Each tier is gated by config, and a
  run without the LLM tier holds a client that raises if called -- so an
  ablation arm is provably LLM-free rather than merely intended to be.
* **One probe path.** Every tier proposes a full requirement set and hands it to
  the same install-and-validate function.
"""

from __future__ import annotations

import copy
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path

import sys

from aura.baseline import establish as establish_baseline, freeze_once, frozen_baseline
from aura.environment import (
    EnvSpec, Environment, ErrorClass, attempt_details, classify_error,
    dry_run_session, probe,
)
from aura.batching import BatchPolicy, Resources, batches
from aura.constraints import ConstraintGraph, build_graph, co_update_plan
from aura.coverage import CoverageReport, compare as compare_coverage, measure_validation
from aura.index import IndexError_, PackageIndex
from aura.llm import LLMUnavailable, NullLLM
from aura.planner import (
    Planner, extract_conflicting_packages, parse_conflict_report,
)
from aura.prioritize import Candidate, architectural_spread, order_candidates
from aura.reqfile import ReqFile, safe_parse_version
from aura.index import version_age_report
from aura.telemetry import Event, Outcome, RunConfig, RunLog, Tier
from aura.validation import Validator


# Above this a per-probe copy of the project costs more than it protects; the
# probe then runs in the shared checkout as before, and the run says so.
ISOLATE_MAX_BYTES = 300 * 1024 * 1024
_COPY_IGNORE = shutil.ignore_patterns(".git", ".aura_*", "__pycache__", "*.pyc")


def _tree_bytes(root: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git" and not d.startswith(".aura_")]
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def isolated_project(config: dict, validator: Validator, dest: Path):
    """A fresh copy of the project for one probe, and a config and validator
    pointed at it.

    Probes run in parallel, and they used to run the tests in one shared
    checkout. A suite that writes there -- hk-flask-restful's setUp runs
    flask_migrate.init(), creating migrations/ and example.db, and tearDown
    deletes them -- then collided with itself: one probe's init found another's
    migrations/ and exited, and a test that passes at baseline failed in 307 of
    405 greedy probes whatever package they changed. Each probe now starts from
    a clean copy, as a probe running alone always did.
    """
    vc = dict(config.get("VALIDATION_CONFIG") or {})
    source = vc.get("project_dir")
    if not source or not Path(source).is_dir():
        return config, validator
    if os.path.lexists(dest):
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(source, dest, ignore=_COPY_IGNORE, symlinks=True)
    vc["project_dir"] = str(dest)
    isolated = copy.copy(validator)
    isolated.cwd = str(dest / vc["test_dir"]) if vc.get("test_dir") else str(dest)
    return {**config, "VALIDATION_CONFIG": vc}, isolated


@dataclass
class AgentResult:
    outcome: Outcome
    updated: dict[str, tuple[str, str]] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)
    reason: str = ""
    # How the starting point came about. Packages moved to *build* the baseline
    # are not packages the search modernized, and are reported apart from them.
    baseline_stage: str = "declared"
    baseline_digest: str = ""
    baseline_repaired: dict[str, list] = field(default_factory=dict)
    passes_used: int = 0
    version_age_before: dict = field(default_factory=dict)
    version_age_after: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)

    @property
    def modernized(self) -> bool:
        return self.outcome in (Outcome.MODERNIZED, Outcome.TRUNCATED_PASSES,
                                Outcome.TRUNCATED_TIMEOUT) and bool(self.updated)


@dataclass
class TierResult:
    pins: dict[str, str] | None = None      # accepted plan, or None
    error_log: str = ""
    error_class: str = ErrorClass.NONE
    exhausted: bool = False                 # searched fully, nothing worked
    deferred: list[str] | None = None       # versions routing chose not to scan


# A batched failure in one of these classes may be contention -- a shared disk,
# memory pressure, a slower clock -- rather than a fact about the candidate.
# Believing it could discard a version that installs fine on its own, so each is
# re-probed serially before it is allowed to count.
CONTENTION_SENSITIVE = frozenset({
    ErrorClass.TIMEOUT, ErrorClass.NETWORK, ErrorClass.OTHER,
    ErrorClass.BUILD_FAILURE,
})


def _conflict_line(report: dict | None, package: str) -> str:
    """The line of pip's conflict report that names ``package``."""
    for constraint in (report or {}).get("constraints", []):
        if package in (constraint["source"], constraint["target"]):
            return constraint["text"]
    return ""


@dataclass
class _ScanState:
    """Bookkeeping for one package's scan: the last failure, and what to recheck."""

    last_log: str = ""
    last_class: str = ErrorClass.NONE
    contended: list[str] = field(default_factory=list)

    def record(self, version: str, result, contended_ok: bool = False) -> None:
        if result.accepted:
            return
        self.last_log = result.stderr or self.last_log
        self.last_class = result.error_class
        if (not contended_ok and result.error_class in CONTENTION_SENSITIVE
                and version not in self.contended):
            self.contended.append(version)

    def take_contended(self, above: str | None) -> list[str]:
        """Pending rechecks newer than ``above``, newest first, removed as taken."""
        floor = safe_parse_version(above) if above else None
        keep, out = [], []
        for version in self.contended:
            parsed = safe_parse_version(version)
            if floor is not None and (parsed is None or parsed <= floor):
                keep.append(version)
            else:
                out.append(version)
        self.contended = keep
        return sorted(out, key=lambda v: safe_parse_version(v) or safe_parse_version("0"),
                      reverse=True)

    def timed_out(self) -> "TierResult":
        return TierResult(error_log=self.last_log, error_class=self.last_class)


class AuraAgent:
    def __init__(
        self,
        config: dict,
        *,
        run_config: RunConfig | None = None,
        llm=None,
        index: PackageIndex | None = None,
        run_log: RunLog | None = None,
        env_spec: EnvSpec | None = None,
        workdir: str | Path = ".",
    ):
        self.config = config
        self.rc = run_config or RunConfig(repo=config.get("PROJECT_NAME", ""))
        self.workdir = Path(workdir).resolve()
        self.requirements_path = (self.workdir / config["REQUIREMENTS_FILE"]).resolve()
        self.index = index or PackageIndex()
        self.env_spec = env_spec or EnvSpec(
            installer=self.rc.installer,
            use_cache=self.rc.install_cache,
            build_isolation=self.rc.build_isolation,
        )
        self.log = run_log or RunLog(self.workdir / "aura_run.jsonl", self.rc)
        # (package, current, target) triples already reported as pre-filtered, so
        # the count is not double-charged when a tier re-derives the same list.
        self._prefiltered: set[tuple[str, str, str]] = set()
        # Declared coupling, rebuilt each pass because the pins move under it.
        self.graph: ConstraintGraph = ConstraintGraph()
        self._graph_pins: dict[str, str] = {}
        self.validator = Validator(
            config,
            tolerance=config.get("METRIC_TOLERANCE", 0.01),
            require_metric_parity=config.get("REQUIRE_METRIC_PARITY", True),
        )
        self.tiers = set(self.rc.tiers)
        self.llm = llm or NullLLM()
        self.planner = Planner(
            self.llm, index=self.index,
            explanation_first=self.rc.llm_explanation_first,
            prompt_style=getattr(self.rc, "llm_prompt_style", "full"),
        )
        self.max_llm_refinements = config.get("MAX_LLM_REFINEMENTS", 3)
        # Greedy learns one blocker per round (pip reports only the first
        # conflict it meets), so a coupled group needs a round per member.
        self.max_greedy_iterations = config.get("MAX_GREEDY_ITERATIONS", 20)
        self.deadline = time.time() + self.rc.timeout_s
        self._probe_counter = 0
        self._pass = 0
        self._resolved_by: str | None = None
        # How this run's baseline came about, and what the repair moved: kept
        # apart from `updated`, which is what the search itself achieved.
        self.baseline_stage = "declared"
        self.baseline_repaired: dict[str, list] = {}
        self._baseline_digest = ""
        self.resources = Resources.detect(self.workdir)
        self.batch_policy = BatchPolicy(
            resources=self.resources,
            k_max=config.get("BATCH_K_MAX", 6),
            enabled=self.rc.batching,
            override=self.rc.batch_k,
        )
        self._slot_lock = threading.Lock()
        self._free_slots: set[int] = set()
        # Configurations already judged in this run, so no tier re-learns a
        # known answer at the cost of a full install.
        self._probe_cache: dict[tuple, object] = {}
        self._next_slot = 0
        # Each probe gets its own copy of the project unless the tree is huge.
        source = (config.get("VALIDATION_CONFIG") or {}).get("project_dir")
        size = _tree_bytes(Path(source)) if source and Path(source).is_dir() else 0
        self._isolate = bool(source) and size <= ISOLATE_MAX_BYTES
        self.log.emit(Event.NOTE, note="probe_isolation", isolated=self._isolate,
                      project_bytes=size)

    # ------------------------------------------------------------------ utils

    @property
    def out_of_time(self) -> bool:
        return time.time() >= self.deadline

    def _claim_slot(self) -> int:
        """Hand out a probe slot. Each slot owns a distinct virtualenv.

        Serial probing reused three rotating environments; batched probing runs
        several at once, so two probes sharing a directory would install over
        each other. Slots are handed out under a lock and released on completion.
        """
        with self._slot_lock:
            self._probe_counter += 1
            if self._free_slots:
                return self._free_slots.pop()
            slot = self._next_slot
            self._next_slot += 1
            return slot

    def _release_slot(self, slot: int) -> None:
        with self._slot_lock:
            self._free_slots.add(slot)

    def _probe(self, reqfile: ReqFile, *, package: str, tier: Tier,
               pins: dict[str, str], record_baseline: bool = False,
               force: bool = False, context: dict | None = None):
        """Install-and-validate one candidate, logging the attempt.

        A configuration already probed in this run is not probed again. Tiers
        escalate by widening the *plan*, and when a tier finds nothing to widen
        with -- greedy extracting no blockers from a test failure, say -- it
        would otherwise re-run the identical pin set the linear scan already
        rejected, costing a full install to re-learn a known answer.

        The key is the *whole* resolved requirement set, not the pins being
        changed: ``{nltk: 3.9}`` in a later pass sits on top of a different set
        of other pins and is a genuinely different configuration.

        ``force`` bypasses the cache for deliberate re-measurement -- confirming
        a batch winner alone, or re-probing a failure contention could explain.
        Those exist precisely to ask the same question under better conditions,
        so answering them from a cache would defeat them.
        """
        signature = (tuple(sorted(reqfile.pins().items())), record_baseline)
        if force:
            self._probe_cache.pop(signature, None)
        cached = None if force else self._probe_cache.get(signature)
        if cached is not None:
            self.log.bump("probes_deduplicated", 1)
            self.log.emit(Event.NOTE, note="probe_already_seen",
                          package=package, tier=tier.value if hasattr(tier, "value")
                          else str(tier), plan=pins,
                          accepted=cached.accepted, error_class=cached.error_class,
                          install_ok=cached.install_ok,
                          validate_ok=bool(cached.validation
                                           and cached.validation.passed),
                          error_excerpt=cached.stderr[-2000:],
                          **attempt_details(cached), **(context or {}))
            return cached

        slot = self._claim_slot()
        env_path = self.workdir / f".aura_probe_{slot}"
        try:
            result = self._probe_in(env_path, reqfile, package=package, tier=tier,
                                    pins=pins, record_baseline=record_baseline,
                                    context=context)
        finally:
            self._release_slot(slot)
        with self._slot_lock:
            self._probe_cache[signature] = result
        return result

    def _probe_in(self, env_path, reqfile: ReqFile, *, package: str, tier: Tier,
                  pins: dict[str, str], record_baseline: bool = False,
                  context: dict | None = None):
        config, validator = self.config, self.validator
        if self._isolate:
            config, validator = isolated_project(self.config, self.validator,
                                                 Path(f"{env_path}_src"))
        result = probe(
            reqfile, config=config, validator=validator,
            spec=self.env_spec, env_path=env_path, record_baseline=record_baseline,
        )
        if record_baseline and validator is not self.validator:
            self.validator.baseline = validator.baseline
        self.log.attempt(
            package=package, tier=tier, plan=pins,
            install_ok=result.install_ok, validate_ok=bool(
                result.validation and result.validation.passed),
            duration_s=result.duration_s, accepted=result.accepted,
            error_class=result.error_class, error_excerpt=result.stderr[-2000:],
            **attempt_details(result),
            **(context or {}),
        )
        if result.drifted_metrics:
            self.log.behavior_check(
                baseline=dict(self.validator.baseline.metrics) if self.validator.baseline else {},
                candidate=dict(result.validation.metrics) if result.validation else {},
                drifted=result.drifted_metrics,
                tolerance=self.validator.tolerance,
                verdict="drift",
            )
        if result.validation is not None:
            self.log.validation(
                phase="baseline" if record_baseline else "candidate",
                passed=result.validation.passed,
                failing=list(result.validation.failing),
                new_failures=result.new_failures,
                metrics=result.validation.metrics,
                duration_s=result.validation.duration_s,
                mode=self.rc.validation_mode,
            )
        return result

    # -------------------------------------------------------------- baseline

    def establish_baseline(self, reqfile: ReqFile) -> tuple[bool, ReqFile, str]:
        """Install the declared requirements and record what 'working' means here.

        A project whose file pins some dependencies and not others no longer
        installs, because the unpinned ones have moved on; that is most published
        projects, not a rare case. Rather than refuse them (a tiny sample) or
        repair them arbitrarily (v1 stripped every constraint and continued, so
        a subject could be evaluated against an AURA-generated baseline), the
        repair is the deterministic ladder in :mod:`aura.baseline`: the same
        inputs give the same baseline for every arm and every model, which is
        what keeps their results comparable.

        What the repair moved is recorded on the agent, never mixed into what
        the search advances.
        """
        def probe_once(candidate: ReqFile):
            return self._probe(candidate, package="<baseline>", tier=Tier.BASELINE,
                               pins={}, record_baseline=True)

        def closure_of(candidate: ReqFile) -> dict[str, str]:
            try:
                with Environment(self.workdir / ".aura_closure",
                                 self.env_spec) as env:
                    return env.resolve_closure(candidate, self.workdir)
            except Exception as exc:
                self.log.emit(Event.NOTE, note="closure_unavailable", error=str(exc))
                return {}

        frozen_path = self.config.get("FROZEN_BASELINE")
        outcome = establish_baseline(
            reqfile, probe=probe_once, closure_of=closure_of, index=self.index,
            commit_date=self.rc.commit_date, python_version=self.rc.python_version,
            repair=self.rc.repair_baseline,
            on_stage=lambda **kw: self.log.emit(Event.NOTE, note="baseline_stage", **kw),
            frozen=frozen_baseline(frozen_path))

        if not outcome.ok:
            self.log.emit(Event.NOTE, note="baseline_failed",
                          repair=self.rc.repair_baseline, detail=outcome.reason)
            return False, reqfile, outcome.reason

        if freeze_once(frozen_path, outcome.reqfile):
            self.log.emit(Event.NOTE, note="baseline_frozen", path=frozen_path,
                          digest=outcome.digest)
        self.baseline_stage = outcome.stage
        self.baseline_repaired = outcome.repaired
        self._baseline_digest = outcome.digest
        failing = sorted(getattr(self.validator.baseline, "failing", ()) or ())
        if failing:
            # Stated plainly: these tests fail before anything is upgraded, so
            # they constrain nothing, and a reader comparing arms should know
            # how much of the suite is actually holding the result up.
            self.log.bump("baseline_failing_tests", len(failing))
            self.log.emit(Event.NOTE, note="baseline_has_failures",
                          count=len(failing), tests=failing[:10])
        if outcome.repaired:
            self.log.bump("baseline_packages_repaired", len(outcome.repaired))
            self.log.emit(Event.NOTE, note="baseline_repaired",
                          stage=outcome.stage, packages=outcome.repaired,
                          on_date=self.rc.commit_date)
        # The probe of the winning stage recorded the reference validation, so
        # the frozen set it reported is this baseline's exact environment.
        locked = outcome.reqfile
        self.log.emit(Event.NOTE, note="baseline_established",
                      stage=outcome.stage, digest=outcome.digest,
                      repaired=len(outcome.repaired))
        return True, locked, ""

    # -------------------------------------------------------- candidate build

    def build_candidates(self, reqfile: ReqFile, spread: dict[str, int],
                         statements: dict[str, int]) -> list[Candidate]:
        primary = self._primary_packages()
        candidates: list[Candidate] = []
        for name, current in reqfile.pins().items():
            try:
                latest = self.index.latest(name)
            except IndexError_ as exc:
                # Never silently read an unreachable index as 'already current'.
                self.log.emit(Event.NOTE, note="index_lookup_failed",
                              package=name, error=str(exc))
                continue
            if not latest:
                continue
            cur_v, new_v = safe_parse_version(current), safe_parse_version(latest)
            if cur_v is None or new_v is None or new_v <= cur_v:
                continue
            candidates.append(Candidate(
                name=name, current=current, target=latest,
                spread=spread.get(name, 0),
                import_statements=statements.get(name, 0),
                is_primary=name in primary,
            ))
        return candidates

    def _primary_packages(self) -> set[str]:
        path = self.workdir / self.config.get("PRIMARY_REQUIREMENTS_FILE", "")
        if not path.name or not path.exists():
            return set()
        try:
            return ReqFile.load(path).names()
        except OSError:
            return set()

    # ------------------------------------------------------------ tier: linear

    def _candidate_versions(self, candidate: Candidate) -> tuple[list[str], str]:
        """Versions worth probing, newest first, after the free metadata filter.

        Reverse Linear Scan is optimal for finding the newest test-passing
        version -- any algorithm must probe every version above the answer -- so
        the only way to reduce oracle calls without losing exactness is to
        remove candidates that provably cannot work. ``requires-python`` and
        yank status come from the index listing we already fetched, so this
        elimination is free.
        """
        key = (candidate.name, candidate.current, candidate.target)
        try:
            versions = self.index.versions_between(
                candidate.name, candidate.current, candidate.target)
        except IndexError_ as exc:
            return [], str(exc)
        if not versions:
            return [], ""

        runtime = self.rc.python_version or ".".join(sys.version.split()[0].split(".")[:2])
        try:
            keep, dropped = self.index.compatible_versions(
                candidate.name, versions, runtime)
        except IndexError_:
            keep, dropped = versions, []

        if dropped and key not in self._prefiltered:
            self._prefiltered.add(key)
            self.log.emit(
                Event.NOTE, note="metadata_prefilter", package=candidate.name,
                runtime=runtime, considered=len(versions),
                eliminated=len(dropped), remaining=len(keep),
                reasons=sorted({d["reason"] for d in dropped}),
                dropped=dropped[:40],
            )
            self.log.bump("prefiltered_versions", len(dropped))
        return list(reversed(keep)), ""

    def measure_coverage(self, reqfile: ReqFile, phase: str) -> CoverageReport | None:
        """Size the oracle: how much code does validation actually execute?

        Run in its own environment, after any decision it could otherwise
        influence. Coverage is a property of the validation script rather than
        of the candidate version, so twice per subject (baseline and accepted
        result) is enough -- instrumenting all several-hundred probes would cost
        real time to re-measure a constant.
        """
        if not self.rc.measure_coverage:
            return None
        env_path = self.workdir / ".aura_coverage_env"
        try:
            with Environment(env_path, self.env_spec) as env:
                output, code = env.install_requirements(reqfile, self.workdir)
                if code != 0:
                    report = CoverageReport(
                        ok=False, reason=f"install failed: {output[-300:]}")
                else:
                    if self.config.get("IS_INSTALLABLE_PACKAGE", False):
                        env.install_project(
                            self.config.get("VALIDATION_CONFIG", {}).get(
                                "project_dir") or ".",
                            self.config.get("PROJECT_EXTRAS", ""),
                            editable=self.config.get("EDITABLE_INSTALL", False))
                    report = measure_validation(
                        env, self.validator, repo_root=self.workdir,
                        dependencies=sorted(reqfile.pins()), workdir=self.workdir,
                        timeout_s=self.validator.timeout_s)
        except Exception as exc:                  # never fail a run over telemetry
            report = CoverageReport(ok=False, reason=f"coverage error: {exc}")

        self.log.emit(Event.COVERAGE, phase=phase, **report.to_dict())
        return report

    def refresh_graph(self, baseline: ReqFile) -> None:
        """Rebuild the declared-constraint graph for the current pin set.

        Only exactly-pinned lines are read, which is what makes the graph
        binding: a ``torchvision>=0.13`` line leaves pip free to move
        torchvision, so it constrains nothing and must not be treated as if it
        did. ``ReqFile.pins()`` already returns ``==`` lines only.
        """
        if not self.rc.constraint_graph:
            return
        pins = baseline.pins()
        if pins == self._graph_pins:
            return
        self._graph_pins = dict(pins)
        try:
            self.graph = build_graph(pins, self.index,
                                     python_version=self.rc.python_version)
        except Exception as exc:            # never let metadata break the run
            self.graph = ConstraintGraph()
            self.log.emit(Event.NOTE, note="constraint_graph_failed", error=str(exc))
            return
        self.log.emit(
            Event.CONSTRAINT_GRAPH, edges=len(self.graph.edges),
            unknown=sorted(self.graph.unknown),
            couples=sorted(sorted(pair) for pair in self.graph.couples()),
            detail=[c.describe() for c in self.graph.edges if str(c.specifier)][:80],
        )

    def _route_after_conflict(self, candidate: Candidate, versions: list[str],
                              error_class: str) -> tuple[str, dict]:
        """Pick the next tier from the failure class. Returns (tier, evidence).

        The only routing decision that can *lose* a solution is skipping the
        linear scan, so it is made on proof rather than on the error class
        alone: skip only when every remaining candidate is forbidden by a
        pinned package's declared constraint. ``torchvision==0.13.0`` pinning
        ``torch==1.12.0`` blocks the whole range, so forty installs would fail
        identically; ``torch<1.14`` does not, and the scan still finds 1.13.0.
        """
        if not (self.rc.routing and self.rc.constraint_graph):
            return Tier.LINEAR.value, {}
        # Routing only chooses between the linear scan and greedy. With neither
        # enabled there is no decision to make or to report.
        if not self.tiers & {Tier.LINEAR.value, Tier.GREEDY.value}:
            return Tier.LINEAR.value, {}
        if error_class != ErrorClass.RESOLUTION_CONFLICT or not versions:
            return Tier.LINEAR.value, {}

        permitted, blocked = self.graph.feasible(candidate.name, versions)
        if permitted:
            return Tier.LINEAR.value, {"permitted": len(permitted),
                                       "blocked": len(blocked)}
        reasons = sorted({b for entry in blocked for b in entry["blockers"]})
        return Tier.GREEDY.value, {"permitted": 0, "blocked": len(blocked),
                                   "blockers": reasons, "probes_saved": len(versions)}

    def _tier_linear(self, candidate: Candidate, baseline: ReqFile,
                     seed_log: str, versions: list[str] | None = None,
                     *, route: bool = False) -> TierResult:
        """Reverse Linear Scan: newest-first through this package's version line.

        The first version tried is the newest this interpreter can install --
        what a one-shot upgrade would install -- and it is tried alone. Only
        when it fails does the scan read the failure: a build failure gets one
        wheel-only retry, and with ``route`` set, a conflict whose every older
        version is forbidden by a declared pin hands the package to the next
        tier without installing them (``deferred``).
        """
        if versions is None:
            versions, error = self._candidate_versions(candidate)
            if error:
                return TierResult(error_log=error)
        if not versions:
            return TierResult(error_log=seed_log, exhausted=True)

        state = _ScanState(seed_log)
        top, rest = versions[0], versions[1:]
        # One resolver sandbox for the whole scan, not one per candidate.
        with dry_run_session(spec=self.env_spec,
                             env_path=self.workdir / ".aura_dryrun",
                             workdir=self.workdir) as resolve:
            if self._try_newest(candidate, baseline, top, state, resolve):
                return TierResult(pins={candidate.name: top})

            if state.last_class == ErrorClass.BUILD_FAILURE and self.rc.routing:
                # A failed source build may say more about this machine's
                # toolchain than about the version. One wheel-only retry
                # separates the two.
                retry = self._retry_wheel_only(candidate, baseline,
                                               {candidate.name: top})
                if retry is not None:
                    self._resolved_by = "linear (wheel only)"
                    return TierResult(pins=retry)

            if route and rest:
                chosen, evidence = self._route_after_conflict(
                    candidate, rest, state.last_class)
                if evidence:
                    self.log.emit(Event.ROUTING, package=candidate.name,
                                  error_class=state.last_class, chose=chosen,
                                  **evidence)
                if chosen == Tier.GREEDY.value:
                    self.log.bump("routing_skipped_linear", 1)
                    self.log.bump("probes_saved", evidence.get("probes_saved", 0))
                    return TierResult(error_log=state.last_log,
                                      error_class=state.last_class, deferred=rest)

            return self._scan(candidate, baseline, rest, state, resolve)

    def _try_newest(self, candidate: Candidate, baseline: ReqFile, version: str,
                    state: "_ScanState", resolve) -> bool:
        """The scan's first step: the newest installable version, alone.

        A resolver dry run first, so a version pip cannot even resolve costs a
        resolve rather than an install; its conflict report is what routing and
        greedy read next.
        """
        plan = baseline.with_pins({candidate.name: version})
        resolvable, dry_log = resolve(plan)
        if not resolvable:
            report = parse_conflict_report(dry_log)
            state.last_log = dry_log
            state.last_class = classify_error(dry_log)
            self.log.emit(Event.NOTE, note="linear_newest_rejected",
                          package=candidate.name, version=version,
                          error_class=state.last_class,
                          reason=(report["constraints"][0]["text"]
                                  if report and report["constraints"] else ""),
                          error_excerpt=dry_log[-2000:])
            self.log.bump("dry_run_rejections", 1)
            return False
        result = self._probe(plan, package=candidate.name, tier=Tier.LINEAR,
                             pins={candidate.name: version})
        # Probed alone, so a failure here is never contention.
        state.record(version, result, contended_ok=True)
        return bool(result.accepted)

    def _scan(self, candidate: Candidate, baseline: ReqFile, versions: list[str],
              state: "_ScanState", resolve) -> TierResult:
        """The rest of the scan: older versions, newest first, batched."""
        if not versions:
            return TierResult(error_log=state.last_log, error_class=state.last_class,
                              exhausted=True)
        plan = self.batch_policy.plan(self._wheel_size(candidate))
        k = plan["k"]
        if k > 1:
            self.log.emit(Event.BATCH, phase="plan", package=candidate.name,
                          candidates=len(versions), **plan)

        dry_rejected: list[str] = []
        for group in batches(versions, k):
            if self.out_of_time:
                return state.timed_out()

            viable = []
            for version in group:
                resolvable, dry_log = resolve(
                    baseline.with_pins({candidate.name: version}))
                if resolvable:
                    viable.append(version)
                else:
                    dry_rejected.append(version)
                    state.last_log = dry_log
                    self.log.emit(Event.NOTE, note="dry_run_rejected",
                                  package=candidate.name, version=version)
            if not viable:
                continue

            winner = self._probe_group(candidate, baseline, viable, state)
            if winner:
                if dry_rejected:
                    self._note_dry_run(candidate, dry_rejected, len(versions))
                return TierResult(pins={candidate.name: winner})

        if dry_rejected:
            self._note_dry_run(candidate, dry_rejected, len(versions))
        # Nothing passed. Anything that failed for a reason contention could
        # explain has not really been ruled out yet.
        recovered = self._recheck_contended(candidate, baseline, state, above=None)
        if recovered:
            return TierResult(pins={candidate.name: recovered})
        return TierResult(error_log=state.last_log, error_class=state.last_class,
                          exhausted=True)

    def _scan_deferred(self, candidate: Candidate, baseline: ReqFile,
                       versions: list[str], seed_log: str) -> TierResult:
        """Scan the versions routing skipped, when every other tier failed."""
        state = _ScanState(seed_log)
        with dry_run_session(spec=self.env_spec,
                             env_path=self.workdir / ".aura_dryrun",
                             workdir=self.workdir) as resolve:
            return self._scan(candidate, baseline, versions, state, resolve)

    def _note_dry_run(self, candidate: Candidate, rejected: list[str],
                      considered: int) -> None:
        self.log.emit(Event.NOTE, note="linear_dry_run_summary",
                      package=candidate.name, rejected=rejected,
                      considered=considered)
        self.log.bump("dry_run_rejections", len(rejected))

    def _wheel_size(self, candidate: Candidate) -> float | None:
        try:
            return self.index.wheel_size_mb(candidate.name, candidate.target)
        except Exception:
            return None

    def _probe_group(self, candidate: Candidate, baseline: ReqFile,
                     group: list[str], state: "_ScanState") -> str | None:
        """Probe a contiguous block of candidates and return the highest passer.

        Taking the *highest* is what keeps the batch equivalent to a serial
        scan: a serial scan stops at the newest version that passes, and every
        version above it is somewhere in this block or an earlier one, where it
        failed.
        """
        if len(group) == 1:
            version = group[0]
            result = self._probe(baseline.with_pins({candidate.name: version}),
                                 package=candidate.name, tier=Tier.LINEAR,
                                 pins={candidate.name: version})
            state.record(version, result)
            return version if result.accepted else None

        started = time.time()
        results: dict[str, object] = {}
        with ThreadPoolExecutor(max_workers=len(group)) as pool:
            futures = {
                pool.submit(self._probe,
                            baseline.with_pins({candidate.name: v}),
                            package=candidate.name, tier=Tier.LINEAR,
                            pins={candidate.name: v},
                            context={"batch_size": len(group)}): v
                for v in group
            }
            for future in as_completed(futures):
                version = futures[future]
                try:
                    results[version] = future.result()
                except Exception as exc:      # a crashed probe is not a verdict
                    self.log.emit(Event.NOTE, note="batch_probe_crashed",
                                  package=candidate.name, version=version,
                                  error=str(exc))
        for version, result in results.items():
            state.record(version, result)

        passing = [v for v, r in results.items() if r.accepted]
        self.log.emit(Event.BATCH, phase="result", package=candidate.name,
                      probed=group, passed=sorted(passing),
                      wall_clock_s=round(time.time() - started, 2))
        self.log.bump("batched_probes", len(group))
        self.log.bump("batches", 1)
        if not passing:
            return None

        winner = max(passing, key=lambda v: safe_parse_version(v) or safe_parse_version("0"))

        # Guard 1: a version above the winner may have failed only because k
        # installs were competing. Re-probe those alone before accepting.
        recovered = self._recheck_contended(candidate, baseline, state, above=winner)
        if recovered:
            return recovered

        # Guard 2: the winner itself was measured under contention. Confirm it
        # serially, so nothing is ever accepted on a concurrent measurement.
        confirm = self._probe(baseline.with_pins({candidate.name: winner}),
                              package=candidate.name, tier=Tier.LINEAR,
                              pins={candidate.name: winner}, force=True,
                              context={"confirming": True})
        self.log.bump("batch_confirmations", 1)
        if confirm.accepted:
            return winner
        self.log.emit(Event.NOTE, note="batch_winner_failed_confirmation",
                      package=candidate.name, version=winner,
                      error_class=confirm.error_class)
        self.log.bump("batch_confirmation_failures", 1)
        state.record(winner, confirm)
        others = sorted((v for v in passing if v != winner),
                        key=lambda v: safe_parse_version(v) or safe_parse_version("0"),
                        reverse=True)
        for version in others:
            again = self._probe(baseline.with_pins({candidate.name: version}),
                                package=candidate.name, tier=Tier.LINEAR,
                                pins={candidate.name: version}, force=True,
                                context={"confirming": True})
            self.log.bump("batch_confirmations", 1)
            if again.accepted:
                return version
            state.record(version, again)
        return None

    def _recheck_contended(self, candidate: Candidate, baseline: ReqFile,
                           state: "_ScanState", above: str | None) -> str | None:
        """Re-probe serially the failures that contention could explain.

        Only those newer than ``above`` matter: an older one cannot beat a
        winner we already have.
        """
        pending = state.take_contended(above)
        if not pending:
            return None
        for version in pending:
            if self.out_of_time:
                return None
            self.log.bump("contention_rechecks", 1)
            result = self._probe(baseline.with_pins({candidate.name: version}),
                                 package=candidate.name, tier=Tier.LINEAR,
                                 pins={candidate.name: version}, force=True)
            state.record(version, result, contended_ok=True)
            if result.accepted:
                self.log.emit(Event.NOTE, note="contention_recheck_rescued",
                              package=candidate.name, version=version)
                self.log.bump("contention_rescued", 1)
                return version
        return None

    # ------------------------------------------------------------ tier: greedy

    def _newest_usable(self, name: str, current: str | None) -> str | None:
        """The newest release above ``current`` this interpreter can install.

        Greedy used to take the absolute latest release, which on an older
        interpreter is often one pip will refuse outright (click 8.5 needs
        Python 3.10) -- a probe spent learning what the index metadata already
        said.
        """
        if not current:
            return None
        try:
            history = self.index.history(name)
        except IndexError_:
            return None
        if history is None or not history.latest:
            return None
        above = history.between(current, history.latest)
        runtime = self.rc.python_version or ".".join(
            sys.version.split()[0].split(".")[:2])
        try:
            above, _ = self.index.compatible_versions(name, above, runtime)
        except IndexError_:
            pass
        if not above:
            return None
        return max(above, key=lambda v: safe_parse_version(v) or safe_parse_version("0"))

    def _tier_greedy(self, candidate: Candidate, baseline: ReqFile,
                     seed_log: str) -> TierResult:
        """Greedy Co-Resolution, as in the original paper.

        Start from the target alone. Each round, the resolver's conflict report
        names a blocker; add it to the group at its newest installable version
        and try again. pip reports only the first conflict it meets, so the
        group grows by one blocker per round until it installs and validates,
        or a round adds nothing new.

        Blockers are taken only from the resolver's reports -- never predicted
        from metadata -- so the group is exactly what the paper's procedure
        builds. Two refinements change the cost, not the procedure: each plan is
        checked with a resolver dry run before any install, so a round that only
        reveals the next blocker costs a resolve rather than an install; and
        "newest" means newest this interpreter can install. Declared metadata is
        used for one thing only: when pip names a blocker, the version chosen is
        the newest that still allows the target, if metadata says the newest
        release would forbid it (torchvision 0.23 requires a newer torch than
        the one being installed; 0.15.1 declares the one we want).
        """
        installed = baseline.pins()
        group: dict[str, str] = {candidate.name: candidate.target}
        attempted: set[tuple] = set()
        last_log, last_class = seed_log, ErrorClass.NONE

        preferred: dict[str, str] = {}
        proof = ""
        if self.rc.constraint_graph and self.graph.edges:
            solved, reason = co_update_plan(
                candidate.name, candidate.target, self.graph, installed,
                self.index, python_version=self.rc.python_version)
            self.log.emit(Event.CONSTRAINT_GRAPH, phase="co_update_plan",
                          package=candidate.name, target=candidate.target,
                          plan=solved, reason=reason)
            if solved:
                # Versions only. Whether a package joins the group is still
                # decided by the resolver naming it.
                preferred = {k: v for k, v in solved.items() if k != candidate.name}
            elif reason.startswith("no_unlocking_version"):
                # A proof, not a guess: nothing unlocks this target while that
                # blocker is pinned. Worth carrying into the LLM prompt.
                proof = (f"\n{reason}: no released version of "
                         f"{reason.split(':')[-1]} permits "
                         f"{candidate.name}=={candidate.target}")

        with dry_run_session(spec=self.env_spec,
                             env_path=self.workdir / ".aura_dryrun_greedy",
                             workdir=self.workdir) as resolve:
            # Start from the target alone. Its own conflict report names the
            # first blocker -- not a report left over from an older version the
            # linear scan tried last. If it resolves, it is installed and tested
            # (free when the linear scan already did so).
            alone = {candidate.name: candidate.target}
            resolvable, dry_log = resolve(baseline.with_pins(alone))
            report = None if resolvable else parse_conflict_report(dry_log)
            self.log.emit(Event.NOTE, note="greedy_start", package=candidate.name,
                          plan=dict(alone),
                          dry_run="ok" if resolvable else "rejected",
                          reason=("" if resolvable else
                                  (report["constraints"][0]["text"]
                                   if report and report["constraints"] else "")),
                          error_excerpt="" if resolvable else dry_log[-2000:])
            if resolvable:
                result = self._probe(baseline.with_pins(alone),
                                     package=candidate.name, tier=Tier.GREEDY,
                                     pins=dict(alone), context={"round": 0})
                if result.accepted:
                    return TierResult(pins=dict(alone))
                last_log = (result.stderr or dry_log) + proof
                last_class = result.error_class
            else:
                self.log.bump("greedy_dry_run_rejections", 1)
                last_log, last_class = dry_log + proof, classify_error(dry_log)
            attempted.add(tuple(sorted(alone.items())))

            for iteration in range(self.max_greedy_iterations):
                if self.out_of_time:
                    break
                before = dict(group)
                blockers = extract_conflicting_packages(last_log, known=set(installed))
                report = parse_conflict_report(last_log)
                self.log.conflict_detection(
                    package=candidate.name,
                    extracted=extract_conflicting_packages(last_log),
                    installed=sorted(installed),
                    source="pip_conflict_report" if report else "regex",
                    reported=report["packages"] if report else None,
                    constraints=([c["text"] for c in report["constraints"]]
                                 if report else None),
                )
                added_now: list[dict] = []
                for blocker in blockers:
                    if blocker in group or blocker == candidate.name:
                        continue
                    newest = self._newest_usable(blocker, installed.get(blocker))
                    target = preferred.get(blocker) or newest
                    if not target:
                        continue
                    group[blocker] = target
                    hinted = preferred.get(blocker) is not None and target != newest
                    added_now.append({
                        "package": blocker, "from": installed.get(blocker, ""),
                        "version": target,
                        "because": _conflict_line(report, blocker),
                        "chosen_as": (f"newest version that allows "
                                      f"{candidate.name}=={candidate.target}"
                                      if hinted else "newest installable version"),
                    })

                added = group != before
                signature = tuple(sorted(group.items()))
                if signature in attempted or not added:
                    if added:
                        why = "this plan was already tried"
                    elif last_class in (ErrorClass.VALIDATION_FAILED,
                                        ErrorClass.VALIDATION_DRIFT):
                        why = ((f"{candidate.name}=={candidate.target} installs on "
                                "its own, but the project's tests fail with it"
                                if iteration == 0 else
                                "the group installs, but the project's tests fail "
                                "with it") + "; there is no conflict to follow")
                    else:
                        why = "the resolver named no new blocker"
                    if iteration == 0:
                        self.log.bump("greedy_skipped_no_group", 1)
                    self.log.emit(Event.NOTE, note="greedy_stopped",
                                  package=candidate.name, rounds=iteration,
                                  reason=why)
                    break
                attempted.add(signature)

                # A plan the resolver cannot satisfy costs a dry run, not an
                # install, and its conflict report feeds the next iteration.
                proposed = baseline.with_pins(group)
                resolvable, dry_log = resolve(proposed)
                dry_report = None if resolvable else parse_conflict_report(dry_log)
                self.log.emit(Event.NOTE, note="greedy_round",
                              package=candidate.name, round=iteration + 1,
                              added=added_now, plan=dict(group),
                              dry_run="ok" if resolvable else "rejected",
                              dry_run_reason=(
                                  "" if resolvable else
                                  (dry_report["constraints"][0]["text"]
                                   if dry_report and dry_report["constraints"]
                                   else "the resolver could not satisfy this plan")))
                if not resolvable:
                    self.log.bump("greedy_dry_run_rejections", 1)
                    last_log, last_class = dry_log, classify_error(dry_log)
                    continue

                result = self._probe(proposed, package=candidate.name,
                                     tier=Tier.GREEDY, pins=dict(group),
                                     context={"round": iteration + 1})
                if result.accepted:
                    return TierResult(pins=dict(group))
                last_log, last_class = result.stderr or last_log, result.error_class
            else:
                self.log.emit(Event.NOTE, note="greedy_stopped",
                              package=candidate.name,
                              rounds=self.max_greedy_iterations,
                              reason=f"round limit ({self.max_greedy_iterations}) reached")

        return TierResult(error_log=last_log, error_class=last_class, exhausted=True)

    # --------------------------------------------------------------- tier: llm

    def _tier_llm(self, candidate: Candidate, baseline: ReqFile,
                  seed_log: str) -> TierResult:
        """LLM-guided planning with a bounded self-refine loop."""
        installed = baseline.pins()
        ceilings: dict[str, str] = {}
        for name, current in installed.items():
            try:
                latest = self.index.latest(name)
            except IndexError_:
                continue
            if latest and (lv := safe_parse_version(latest)) and \
               (cv := safe_parse_version(current)) and lv > cv:
                ceilings[name] = latest
        ceilings.setdefault(candidate.name, candidate.target)

        history: list[tuple[str, str]] = []
        last_log, last_class = seed_log, ErrorClass.NONE

        for iteration in range(1, self.max_llm_refinements + 1):
            if self.out_of_time:
                break
            try:
                plan = self.planner.propose(
                    target=candidate.name, error_log=last_log,
                    current=installed, available=ceilings, history=history,
                )
            except LLMUnavailable:
                # A deterministic arm reached the LLM tier: that is a config bug.
                raise

            self.log.llm_call(
                plan.usage, purpose="co_resolution", package=candidate.name,
                iteration=iteration, proposed=plan.requirements,
                rejected=plan.rejected, resolvable=plan.resolvable,
                explanation=plan.explanation,
                inputs=plan.decision.as_dict() if plan.decision else {},
                reason="" if plan.resolvable else plan.reason,
                raw=plan.raw or "",
            )
            if not plan.resolvable:
                self.log.emit(Event.NOTE, note="llm_early_exit",
                              package=candidate.name, reason=plan.reason)
                break
            if plan.is_empty:
                history.append(("<no valid plan>", plan.reason))
                continue

            pins = plan.as_pins()
            result = self._probe(baseline.with_pins(pins), package=candidate.name,
                                 tier=Tier.LLM, pins=pins)
            if result.accepted:
                self.log.emit(Event.NOTE, note="llm_plan_accepted",
                              package=candidate.name, iteration=iteration)
                return TierResult(pins=pins)
            last_log, last_class = result.stderr or last_log, result.error_class
            history.append((", ".join(plan.requirements), result.error_class))

        return TierResult(error_log=last_log, error_class=last_class, exhausted=True)

    # ------------------------------------------------------- per-package loop

    def update_package(self, candidate: Candidate, baseline: ReqFile) -> dict[str, str] | None:
        """Escalate through the enabled tiers. Returns accepted pins, or None.

        Brackets the search with a start and a result event, so the log can say
        per package what was tried and what came of it -- including when every
        answer came from earlier work and nothing new was installed.
        """
        self._resolved_by = None
        probes_before = self._probe_counter
        self.log.emit(Event.NOTE, note="package_start", package=candidate.name,
                      current=candidate.current, target=candidate.target,
                      pass_num=self._pass)
        pins = self._escalate(candidate, baseline)
        self.log.emit(Event.NOTE, note="package_result", package=candidate.name,
                      current=candidate.current, pins=pins or {},
                      tier=self._resolved_by, pass_num=self._pass,
                      new_installs=self._probe_counter - probes_before)
        return pins

    def _escalate(self, candidate: Candidate, baseline: ReqFile) -> dict[str, str] | None:
        # Packages accepted earlier in this pass may have brought new declared
        # constraints (flask 1.1.4 caps itsdangerous below 2.0; 1.1.1 did not).
        # A no-op when nothing changed; otherwise only the new pins are read.
        self.refresh_graph(baseline)
        # Metadata pre-filter: the versions this interpreter could install at
        # all. It is the linear scan's version line and greedy's target. The LLM
        # is not handed it -- choosing where to start is the model's job.
        usable, index_error = self._candidate_versions(candidate)
        searching = bool(self.tiers & {Tier.LINEAR.value, Tier.GREEDY.value})
        if not index_error:
            if not usable:
                self.log.emit(Event.NOTE, note="no_usable_versions",
                              package=candidate.name, current=candidate.current,
                              latest=candidate.target)
                return None
            if usable[0] != candidate.target:
                if searching:
                    self.log.emit(Event.NOTE, note="target_demoted",
                                  package=candidate.name,
                                  latest=candidate.target, target=usable[0])
                candidate = replace(candidate, target=usable[0])

        # There is no separate "try the newest" step: the linear scan starts
        # there, greedy starts from it alone, and the LLM is told nothing.
        seed_log = ""
        deferred: list[str] | None = None
        if Tier.LINEAR.value in self.tiers and not self.out_of_time:
            # Routing can only hand the package on if a later tier exists.
            later = bool(self.tiers & {Tier.GREEDY.value, Tier.LLM.value})
            outcome = self._tier_linear(
                candidate, baseline, "", None if index_error else usable,
                route=later and self.rc.routing)
            if outcome.pins:
                self._resolved_by = self._resolved_by or "linear"
                return outcome.pins
            seed_log = outcome.error_log
            deferred = outcome.deferred

        for tier_name, runner in (
            (Tier.GREEDY.value, self._tier_greedy),
            (Tier.LLM.value, self._tier_llm),
        ):
            if tier_name not in self.tiers or self.out_of_time:
                continue
            outcome = runner(candidate, baseline, seed_log)
            if outcome.pins:
                self._resolved_by = tier_name
                return outcome.pins
            seed_log = outcome.error_log or seed_log

        # Routing skipped the rest of the scan on the graph's word. If every
        # later tier also failed, scan it rather than trust metadata absolutely.
        if deferred and not self.out_of_time:
            self.log.emit(Event.NOTE, note="routing_fallback_to_linear",
                          package=candidate.name, versions=len(deferred))
            outcome = self._scan_deferred(candidate, baseline, deferred, seed_log)
            if outcome.pins:
                self.log.bump("routing_fallback_rescued", 1)
                self._resolved_by = "linear"
                return outcome.pins
        return None

    def _retry_wheel_only(self, candidate: Candidate, baseline: ReqFile,
                          pins: dict[str, str]) -> dict[str, str] | None:
        """Re-probe a build failure with source builds disabled.

        pip falls back to compiling from source when no wheel matches, so a
        build failure can mean "no compiler here" rather than "this version is
        incompatible". Forcing wheels separates the machine from the candidate;
        v1 recorded the failure and moved on, losing versions that install fine
        from a wheel.
        """
        spec = self.env_spec
        if "--only-binary=:all:" in spec.extra_pip_args:
            return None
        original = self.env_spec
        self.env_spec = replace(
            spec, extra_pip_args=spec.extra_pip_args + ("--only-binary=:all:",))
        try:
            self.log.bump("wheel_only_retries", 1)
            result = self._probe(baseline.with_pins(pins), package=candidate.name,
                                 tier=Tier.LINEAR, pins=pins,
                                 context={"wheel_only": True})
        finally:
            self.env_spec = original
        self.log.emit(Event.ROUTING, package=candidate.name,
                      error_class=ErrorClass.BUILD_FAILURE, chose="wheel_only_retry",
                      rescued=bool(result.accepted))
        if result.accepted:
            self.log.bump("wheel_only_rescued", 1)
            return pins
        return None

    # ------------------------------------------------------- tier: regrouping

    def _staleness(self, pins: dict[str, str]) -> tuple[int, int]:
        """``(days behind latest in total, packages already at latest)``.

        Days rather than release counts: cadences differ by an order of
        magnitude between projects, so trading fifteen releases of one package
        for sixteen of another says nothing about whether the environment ended
        up less stale. Time is the same unit everywhere.
        """
        days, at_latest, releases = 0, 0, 0
        for name, version in pins.items():
            try:
                days += self.index.days_behind(name, version)
                behind = self.index.releases_behind(name, version)
            except IndexError_:
                continue
            releases += behind
            if behind == 0:
                at_latest += 1
        # An index that cannot date a release still counts them, and a measure
        # of zero for everything would make every trade look free.
        return (days or releases), at_latest

    def _steps_back(self, name: str, was: str, now: str) -> int | None:
        """How many releases a trade gives up, or None if it is out of bounds.

        A trade is meant to be a small concession. Crossing a major version
        backwards is not: it gives up a line of development rather than a few
        fixes, whatever the arithmetic says.
        """
        old_v, new_v = safe_parse_version(was), safe_parse_version(now)
        if old_v is None or new_v is None:
            return None
        if old_v.release and new_v.release and old_v.release[0] != new_v.release[0]:
            return None
        try:
            return len(self.index.versions_between(name, now, was))
        except IndexError_:
            return None

    def regroup(self, reqfile: ReqFile, baseline: dict[str, str]) -> ReqFile:
        """Trade a local maximum for a better group, once the search has stopped.

        Disabled by default; see ``RunConfig.regroup``. The trade it finds is
        real -- on mean-review-collector, click gains 1699 days of releases for
        406 given up by flask, and the tests still pass -- but the package it
        steps back is typically the one the codebase depends on most, because
        candidates are ordered by architectural spread. Advancing a transitive
        helper at the cost of the framework is not a better environment, and
        staleness cannot express that.

        Escalation only asks a later tier when an earlier one *fails*, so a
        package the scan advanced is never reconsidered -- even when that
        advance is what blocks everything else. Flask 1.1.4 is the newest Flask
        installable on its own and declares ``click<8``; Flask 1.1.2 does not.
        Taking 1.1.4 therefore costs click, jinja2, werkzeug and itsdangerous,
        and no tier ever revisits it because nothing failed.

        This pass asks the question the escalation cannot: for each package
        still far behind, is there a group that reaches it by stepping a
        blocker back -- never below the version the project itself started
        from? A trade is kept only if the tests still pass *and* the whole
        environment ends up nearer the current releases, so it can never churn
        pins for their own sake.
        """
        if not (self.rc.constraint_graph and self.graph.edges):
            return reqfile
        pins = dict(reqfile.pins())
        score, at_latest = self._staleness(pins)
        stuck = sorted(
            ((name, self.index.releases_behind(name, version))
             for name, version in pins.items()),
            key=lambda item: -item[1])
        candidates = [name for name, behind in stuck if behind > 0]
        if not candidates:
            return reqfile
        self.log.emit(Event.NOTE, note="regroup_start",
                      behind=score, at_latest=at_latest, packages=candidates[:10])

        for name in candidates[: self.rc.regroup_packages]:
            if self.out_of_time:
                break
            target = self._newest_usable(name, pins.get(name))
            if not target or target == pins.get(name):
                continue
            self.refresh_graph(reqfile)
            # Blockers may only step *back* here: moving them forward is what
            # greedy already did, and if that had worked this package would not
            # still be behind.
            plan, reason = co_update_plan(
                name, target, self.graph, pins, self.index,
                python_version=self.rc.python_version, floors=baseline,
                ceilings=pins)
            if not plan:
                self.log.emit(Event.NOTE, note="regroup_rejected", package=name,
                              plan={}, reason=reason)
                continue
            # Only worth an install if it is a *trade*: something steps back.
            traded = {k: [pins[k], v] for k, v in plan.items()
                      if k in pins and safe_parse_version(v) is not None
                      and safe_parse_version(pins[k]) is not None
                      and safe_parse_version(v) < safe_parse_version(pins[k])}
            if not traded:
                self.log.emit(Event.NOTE, note="regroup_rejected", package=name,
                              plan=plan, reason="no blocker would step back")
                continue
            if any(safe_parse_version(v) < safe_parse_version(baseline[k])
                   for k, v in plan.items()
                   if k in baseline and safe_parse_version(v)
                   and safe_parse_version(baseline[k])):
                continue                     # never below the project's own start
            # A step back has to be small: a few releases inside the same
            # major line, never a version line abandoned.
            too_far = {k: v for k, v in traded.items()
                       if (steps := self._steps_back(k, v[0], v[1])) is None
                       or steps > self.rc.regroup_max_steps_back}
            if too_far:
                self.log.emit(Event.NOTE, note="regroup_rejected", package=name,
                              plan=plan,
                              reason=f"would give up too much of "
                                     f"{', '.join(sorted(too_far))}")
                continue
            proposed = {**pins, **plan}
            days, latest_now = self._staleness(proposed)
            if latest_now < at_latest:
                self.log.emit(Event.NOTE, note="regroup_rejected", package=name,
                              plan=plan,
                              reason="fewer packages would sit at their newest "
                                     "release")
                continue
            if days >= score:
                self.log.emit(Event.NOTE, note="regroup_rejected", package=name,
                              plan=plan,
                              reason=f"no less stale overall ({days} vs {score} "
                                     f"days behind)")
                continue
            self.log.emit(Event.NOTE, note="regroup_try", package=name,
                          target=target, plan=plan, traded=traded)
            result = self._probe(reqfile.with_pins(plan), package=name,
                                 tier=Tier.GREEDY, pins=plan,
                                 context={"regroup": True})
            if not result.accepted:
                self.log.emit(Event.NOTE, note="regroup_rejected", package=name,
                              plan=plan, reason=result.error_class)
                continue
            reqfile = reqfile.with_pins(plan)
            pins, score, at_latest = dict(reqfile.pins()), days, latest_now
            self.log.bump("regroups_accepted", 1)
            self.log.bump("regroup_days_recovered", max(0, score - days))
            self.log.emit(Event.NOTE, note="regroup_accepted", package=name,
                          plan=plan, traded=traded, behind=days,
                          at_latest=latest_now)
        return reqfile

    # ---------------------------------------------------------------- the run

    def run(self) -> AgentResult:
        reqfile = ReqFile.load(self.requirements_path)

        ok, reqfile, reason = self.establish_baseline(reqfile)
        if not ok:
            result = AgentResult(Outcome.FAILED, reason=reason)
            self.log.finish(Outcome.FAILED, reason=reason)
            return result
        reqfile.save(self.requirements_path)
        # Where the project itself started: the floor for any later trade, and
        # what "advanced" is finally measured against.
        baseline_pins = dict(reqfile.pins())

        baseline_coverage = self.measure_coverage(reqfile, "baseline")
        age_before = version_age_report(self.index, reqfile.pins())
        updated: dict[str, tuple[str, str]] = {}
        attempted: set[str] = set()
        passes_used = 0
        truncated_by_time = False
        changed_in_last_pass = False

        while passes_used < self.rc.max_passes:
            if self.out_of_time:
                truncated_by_time = True
                break
            passes_used += 1
            self._pass = passes_used
            # Counted, so a report can separate "converged" from "ran out of
            # passes" without re-reading the log.
            self.log.bump("passes", 1)
            self.log.emit(Event.PASS_START, pass_num=passes_used)

            self.refresh_graph(reqfile)
            spread, statements = architectural_spread(self.workdir)
            candidates = self.build_candidates(reqfile, spread, statements)
            if not candidates:
                self.log.emit(Event.PASS_END, pass_num=passes_used,
                              changed=0, converged=True)
                changed_in_last_pass = False
                break

            ordered = order_candidates(candidates, self.rc.ordering, self.rc.ordering_seed)
            self.log.emit(
                Event.PRIORITIZATION, pass_num=passes_used, ordering=self.rc.ordering,
                queue=[{"package": c.name, "current": c.current, "target": c.target,
                        "spread": c.spread, "import_statements": c.import_statements,
                        "version_jump": c.version_jump, "is_primary": c.is_primary}
                       for c in ordered],
            )

            changed_this_pass = 0
            processed: set[str] = set()
            for candidate in ordered:
                if self.out_of_time:
                    truncated_by_time = True
                    break
                if candidate.name in processed:
                    continue
                attempted.add(candidate.name)

                pins = self.update_package(candidate, reqfile)
                if not pins:
                    continue
                real = {n: v for n, v in pins.items()
                        if v != reqfile.pins().get(n)}
                if not real:
                    continue
                for name, version in real.items():
                    old = reqfile.pins().get(name, "")
                    updated[name] = (updated.get(name, (old, old))[0], version)
                    processed.add(name)
                reqfile.repin_all(real)
                reqfile.save(self.requirements_path)
                changed_this_pass += len(real)

            self.log.emit(Event.PASS_END, pass_num=passes_used,
                          changed=changed_this_pass, converged=changed_this_pass == 0)
            changed_in_last_pass = changed_this_pass > 0
            if not changed_in_last_pass:
                break

        # A pass cap reached while changes were still landing is truncation,
        # not convergence -- v1 reported both as success.
        if truncated_by_time:
            outcome, reason = Outcome.TRUNCATED_TIMEOUT, f"exceeded {self.rc.timeout_s}s budget"
        elif passes_used >= self.rc.max_passes and changed_in_last_pass:
            outcome, reason = Outcome.TRUNCATED_PASSES, f"hit max_passes={self.rc.max_passes}"
        elif updated:
            outcome, reason = Outcome.MODERNIZED, ""
        else:
            outcome, reason = Outcome.KEPT_BASELINE, "no package could be advanced"

        # Re-measure only when something moved: an unchanged configuration
        # cannot have changed what the oracle executes.
        if updated:
            final_coverage = self.measure_coverage(reqfile, "final")
            if baseline_coverage and final_coverage \
                    and baseline_coverage.ok and final_coverage.ok:
                delta = compare_coverage(baseline_coverage, final_coverage)
                self.log.emit(Event.COVERAGE, phase="delta", **delta)
                if delta["dropped"]:
                    # Less code ran than before. Tests pass and the numbers
                    # match, so nothing else in the pipeline would catch this.
                    self.log.bump("coverage_dropped", len(delta["dropped"]))

        if (self.rc.regroup and Tier.GREEDY.value in self.tiers
                and not self.out_of_time and updated):
            before = dict(reqfile.pins())
            reqfile = self.regroup(reqfile, baseline_pins)
            if reqfile.pins() != before:
                reqfile.save(self.requirements_path)
                # Re-derive what advanced from the project's own starting point:
                # a trade can leave a package below where the scan had it while
                # the environment as a whole is nearer the current releases.
                updated = {name: (baseline_pins[name], version)
                           for name, version in reqfile.pins().items()
                           if name in baseline_pins
                           and version != baseline_pins[name]}
                attempted |= set(updated)

        age_after = version_age_report(self.index, reqfile.pins())
        result = AgentResult(
            outcome=outcome, updated=updated,
            unchanged=sorted(attempted - set(updated)), reason=reason,
            passes_used=passes_used, version_age_before=age_before,
            version_age_after=age_after, counters=self.log.counters,
            baseline_stage=self.baseline_stage,
            baseline_digest=self._baseline_digest,
            baseline_repaired=self.baseline_repaired,
        )
        self.log.finish(outcome, updated=updated, unchanged=result.unchanged,
                        reason=reason,
                        version_age={"before": age_before, "after": age_after},
                        baseline={"stage": self.baseline_stage,
                                  "digest": self._baseline_digest,
                                  "repaired": self.baseline_repaired})
        return result
