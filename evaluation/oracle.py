"""Measuring how much work a subject's validation logic actually does.

Reviewer C: "the reliance on manually authored validation scripts introduces
potential bias." Reviewer B made the same point from the other side: a passing
script "only shows that no failure was observed for the covered executions".

Both objections reduce to one unanswered question -- *is this oracle strong
enough to catch a regression?* -- and it is answerable rather than arguable. We
measure it two ways:

1. **Observed discrimination** (free, from existing logs). Across a run's
   candidate configurations, how many did the oracle reject *on validation*, as
   opposed to the resolver rejecting them at install time? An oracle that never
   once fired did no work, and the subject's result rests entirely on pip.

2. **Mutation detection** (one extra environment per mutation). Deliberately
   damage a working environment and check the oracle notices. The mutations are
   ordered by how certainly they break a project, so the detection rates are
   interpretable:

   - ``missing_dependency`` -- uninstall the most-used package. A working
     project cannot survive this. An oracle that passes is not exercising the
     project at all, and its subject's results should be discarded.
   - ``stale_dependency``  -- pin the most-used package to its oldest release.
     Very likely to break; a strong oracle catches it.
   - ``major_jump``        -- pin it to the newest major. May or may not break;
     reported for context, not as a pass/fail bar.

Only ``missing_dependency`` is treated as a hard floor, because only it is
guaranteed broken independently of the oracle -- which keeps the check from
being circular.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aura.environment import EnvSpec, Environment, ErrorClass
from aura.index import PackageIndex
from aura.prioritize import architectural_spread
from aura.reqfile import ReqFile
from aura.validation import Validator
from evaluation.stats import wilson_interval
from evaluation.arms import FULL_ARM


class Mutation:
    MISSING = "missing_dependency"
    STALE = "stale_dependency"
    MAJOR_JUMP = "major_jump"


HARD_FLOOR = {Mutation.MISSING}


@dataclass
class MutationResult:
    subject: str
    mutation: str
    package: str = ""
    detail: str = ""
    detected: bool = False        # the oracle rejected the damaged environment
    baseline_ok: bool = True
    reason: str = ""
    duration_s: float = 0.0


@dataclass
class OracleReport:
    subject: str
    provenance: str = ""
    baseline_metrics: dict = field(default_factory=dict)
    mutations: list[MutationResult] = field(default_factory=list)
    error: str = ""

    @property
    def passes_floor(self) -> bool:
        """False if the oracle failed to notice a guaranteed-broken environment."""
        floor = [m for m in self.mutations if m.mutation in HARD_FLOOR]
        return bool(floor) and all(m.detected for m in floor)

    @property
    def detection_rate(self) -> float:
        if not self.mutations:
            return 0.0
        return sum(1 for m in self.mutations if m.detected) / len(self.mutations)


def _top_package(workspace: Path, reqfile: ReqFile) -> str | None:
    """The package whose loss should most obviously break this project."""
    spread, _ = architectural_spread(workspace)
    pins = reqfile.pins()
    ranked = sorted(((spread.get(name, 0), name) for name in pins), reverse=True)
    for score, name in ranked:
        if score > 0:
            return name
    return ranked[0][1] if ranked else None


def probe_oracle(subject, workspace: Path, *, index: PackageIndex | None = None,
                 spec: EnvSpec | None = None,
                 mutations: tuple[str, ...] = (Mutation.MISSING, Mutation.STALE,
                                               Mutation.MAJOR_JUMP)) -> OracleReport:
    """Damage a working environment in known ways and see what the oracle notices."""
    index = index or PackageIndex()
    spec = spec or EnvSpec(use_cache=True, build_isolation=True)
    config = subject.agent_config(workspace)
    report = OracleReport(subject=subject.name,
                          provenance=getattr(subject, "validation_provenance", ""))

    reqfile = ReqFile.load(workspace / subject.requirements_file)
    target = _top_package(workspace, reqfile)
    if target is None:
        report.error = "no pinned dependency to mutate"
        return report

    validator = Validator(config, tolerance=subject.metric_tolerance)

    # Reference: the undamaged environment must pass, or nothing below means anything.
    with Environment(workspace / ".aura_oracle_base", spec) as env:
        output, code = env.install_requirements(reqfile, workspace)
        if code != 0:
            report.error = f"baseline install failed: {ErrorClass.NONE}"
            return report
        baseline = validator.record_baseline(env.python)
        report.baseline_metrics = dict(baseline.metrics)
        if not baseline.passed:
            report.error = "baseline does not pass its own validation"
            return report

    for mutation in mutations:
        started = time.time()
        result = MutationResult(subject=subject.name, mutation=mutation, package=target)

        if mutation == Mutation.MISSING:
            result.detail = f"uninstall {target}"
            with Environment(workspace / ".aura_oracle_mut", spec) as env:
                _, code = env.install_requirements(reqfile, workspace)
                if code != 0:
                    result.reason = "install failed before mutation"
                else:
                    env.uninstall(target)
                    candidate = validator.run(env.python)
                    accepted, _, _, _ = validator.judge(candidate)
                    result.detected = not accepted
        else:
            history = index.history(target)
            if history is None or not history.versions:
                result.reason = f"no release history for {target}"
                report.mutations.append(result)
                continue
            version = (history.versions[0] if mutation == Mutation.STALE
                       else history.versions[-1])
            if version == reqfile.pins().get(target):
                result.reason = f"{target} is already at {version}; no mutation possible"
                report.mutations.append(result)
                continue
            result.detail = f"{target}=={version}"
            mutated = reqfile.with_pins({target: version})
            with Environment(workspace / ".aura_oracle_mut", spec) as env:
                _, code = env.install_requirements(mutated, workspace)
                if code != 0:
                    # The resolver caught it; the oracle was never consulted.
                    result.detected = False
                    result.reason = "rejected at install time, not by the oracle"
                else:
                    candidate = validator.run(env.python)
                    accepted, _, _, _ = validator.judge(candidate)
                    result.detected = not accepted

        result.duration_s = round(time.time() - started, 2)
        report.mutations.append(result)

    return report


def observed_discrimination(analysis, arm: str = FULL_ARM) -> dict:
    """How often the oracle -- rather than the resolver -- rejected a candidate.

    Computed from logs already collected, so it costs nothing. A subject whose
    oracle never rejected anything contributed no execution-based evidence.
    """
    per_subject: dict[str, dict] = {}
    for run in analysis.by_arm.get(arm, []):
        installed = [a for a in run.attempts if a.get("install_ok")]
        oracle_rejects = [a for a in installed
                          if not a.get("accepted") and not a.get("validate_ok")]
        drift_rejects = [e for e in run.drifts]
        per_subject[run.subject] = {
            "attempts": len(run.attempts),
            "reached_the_oracle": len(installed),
            "rejected_by_oracle": len(oracle_rejects),
            "rejected_on_metric_drift": len(drift_rejects),
            "oracle_ever_fired": bool(oracle_rejects),
        }
    fired = sum(1 for v in per_subject.values() if v["oracle_ever_fired"])
    n = len(per_subject)
    return {
        "arm": arm, "subjects": n,
        "oracle_fired_at_least_once": str(wilson_interval(fired, n)) if n else "n/a",
        "silent_oracles": sorted(s for s, v in per_subject.items()
                                 if not v["oracle_ever_fired"]),
        "per_subject": per_subject,
    }


def summarize_reports(reports: list[OracleReport]) -> dict:
    by_mutation: dict[str, list[MutationResult]] = {}
    for report in reports:
        for m in report.mutations:
            by_mutation.setdefault(m.mutation, []).append(m)
    rates = {
        name: str(wilson_interval(sum(1 for m in group if m.detected), len(group)))
        for name, group in sorted(by_mutation.items())
    }
    failed_floor = [r.subject for r in reports if r.mutations and not r.passes_floor]
    return {
        "subjects": len(reports),
        "detection_rate_by_mutation": rates,
        "failed_hard_floor": sorted(failed_floor),
        "note": ("Subjects under failed_hard_floor have an oracle that did not "
                 "notice a missing dependency. Their validation logic is not "
                 "exercising the project and their results must not be reported."),
    }


def write_reports(reports: list[OracleReport], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"summary": summarize_reports(reports),
         "reports": [asdict(r) for r in reports]}, indent=2) + "\n", encoding="utf-8")
    return out
