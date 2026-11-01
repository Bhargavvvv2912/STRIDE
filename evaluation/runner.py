"""Executes one arm on one subject in an isolated workspace.

Every arm gets a fresh checkout at the subject's pinned commit, because AURA
rewrites ``requirements.txt`` in place -- running arms in a shared tree would
let one arm's committed upgrades become another arm's starting point. All arms
therefore begin from byte-identical inputs, which is what makes the paired
comparisons between them meaningful.
"""

from __future__ import annotations

import contextlib
import json
import re
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aura.agent import AuraAgent
from aura.baseline import era_build_tools
from aura.baseline import establish as establish_baseline, freeze_once, frozen_baseline
from aura.environment import (
    EnvSpec, Environment, ErrorClass, attempt_details, find_interpreter, probe,
)
from aura.index import PackageIndex, version_age_report
from aura.llm import build_client
from aura.reqfile import ReqFile
from aura.telemetry import Event, Outcome, RunConfig, RunLog
from aura.validation import Validator
from evaluation.arms import Arm

_WINDOWS = os.name == "nt"
from evaluation.subjects import Subject


@dataclass
class ArmResult:
    subject: str
    arm: str
    outcome: str
    updated: dict = field(default_factory=dict)
    reason: str = ""
    wall_clock_s: float = 0.0
    log_path: str = ""
    counters: dict = field(default_factory=dict)
    version_age_before: dict = field(default_factory=dict)
    version_age_after: dict = field(default_factory=dict)
    error: str = ""
    # Which rung of the repair ladder produced a working environment, or "" if
    # none did. The baseline arm reports the *declared* file, so it can fail on
    # a subject that is perfectly usable once rebuilt; this says which happened.
    baseline_stage: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome in (Outcome.MODERNIZED.value, Outcome.KEPT_BASELINE.value,
                                Outcome.TRUNCATED_PASSES.value,
                                Outcome.TRUNCATED_TIMEOUT.value)


def log_path_for(results_dir: Path, subject: str, arm: str) -> Path:
    """Where one (subject, arm) run is recorded.

    The subject is repeated in the filename, not just the directory. A campaign
    produces hundreds of these and they are read in an editor, where a tab
    labelled ``pip_upgrade.log`` says nothing about which of fifty subjects it
    belongs to. ``rake-nltk__pip_upgrade.log`` is unambiguous wherever it is
    opened, and the per-subject directory still keeps the tree navigable.
    """
    return Path(results_dir) / subject / f"{subject}__{arm}.jsonl"


def publish_to_cache(clone: Path, cache_dir: Path, name: str) -> None:
    """Add a clone to the shared cache so other runs need not fetch it again.

    Arms of a subject run as separate jobs at the same time, so several of them
    reach this point at once with the same subject. Copying straight into
    ``cache_dir/name`` loses that race twice over: the second copy raises
    FileExistsError, and -- worse, because it is silent -- a third arm can copy
    *out* of a directory that is still being written into and run against a
    half-populated checkout.

    Staging under a private name and renaming into place fixes both: the rename
    is atomic, so the cache entry either is not there or is complete, and the
    loser of the race simply discards its copy.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / name
    if target.exists():
        return
    staging = cache_dir / f".{name}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        shutil.copytree(clone, staging, symlinks=True)
        os.replace(staging, target)
    except OSError:
        # Another arm published first (rename onto a non-empty directory
        # fails), or the cache is unwritable. Neither is worth failing a run.
        shutil.rmtree(staging, ignore_errors=True)


def prepare_workspace(subject: Subject, dest: Path, cache_dir: Path | None = None) -> Path:
    """Materialise the subject at its pinned commit in a clean directory."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    cached = (cache_dir / subject.name) if cache_dir else None
    if cached and cached.exists():
        shutil.copytree(cached, dest, symlinks=True)
    else:
        subprocess.run(["git", "clone", "--quiet", subject.url, str(dest)],
                       check=True, capture_output=True, text=True, timeout=1800)
        if cache_dir:
            publish_to_cache(dest, Path(cache_dir), subject.name)

    if subject.commit:
        subprocess.run(["git", "checkout", "--quiet", subject.commit],
                       cwd=dest, check=True, capture_output=True, text=True, timeout=300)
    subprocess.run(["git", "clean", "-xfdq"], cwd=dest,
                   capture_output=True, text=True, timeout=300)
    # The relaxed specification is deliberately *not* derived here. Creating it
    # for every arm would make the arms that need it report it as shipped by the
    # project, which is a different claim from one the harness generated.
    return dest


def ensure_primary_requirements(subject: Subject, workspace: Path) -> tuple[Path, bool]:
    """Make sure the relaxed specification the naive-upgrade arms need exists.

    ``pip_upgrade`` answers "what would you get by just asking for these
    packages?", which needs the requirements with their version constraints
    removed. Almost no real project ships such a file, so it is derived from the
    subject's own ``requirements.txt`` -- names, extras and markers kept, version
    specifiers dropped -- unless the project already provides one, in which case
    the project's own file wins and is left untouched.

    Returns ``(path, derived)``. ``derived`` is recorded in the run log, because
    a baseline the harness generated is a different claim from one the project
    published, and the difference should not be invisible in the results.
    """
    target = Path(workspace) / subject.primary_requirements_file
    if target.exists():
        return target, False

    source = Path(workspace) / subject.requirements_file
    if not source.exists():
        return target, False

    reqfile = ReqFile.load(source)
    reqfile.drop_all_constraints()
    # Lines that install the project's own folders ("-e ./tensorly", "-e .")
    # are not dependencies to upgrade: the project's own code is installed by
    # its setup step, exactly as for every other arm. Keeping them made pip
    # fail before upgrading anything (tensorgrad), a failure of this harness,
    # not of pip.
    local = re.compile(r"^\s*(-e\s+|--editable\s+)?\.{1,2}(/|$|\s)")
    text = "".join(l for l in reqfile.to_text().splitlines(keepends=True)
                   if not local.match(l))
    target.write_text(text)
    return target, True


class InterpreterMissing(RuntimeError):
    """The subject's Python is not installed. Never substitute another one."""


def _interpreter(subject: Subject) -> str:
    """Resolve the interpreter this subject must be validated on.

    Raises rather than falling back. A subject pinned in 2021 that is quietly
    validated on 3.13 produces a result that looks ordinary, means nothing, and
    leaves no trace of the substitution in the logs -- the worst kind of error
    this harness can make.
    """
    found = find_interpreter(subject.python_version)
    if found is None:
        raise InterpreterMissing(
            f"{subject.name} declares Python {subject.python_version}, which is "
            f"not installed. Install it (e.g. `pyenv install "
            f"{subject.python_version}`) rather than running the subject on a "
            "different interpreter.")
    return found


def commit_date(subject: Subject, workspace: Path | None = None) -> str:
    """The date of the subject's pinned commit, for rebuilding its environment.

    Stated in the registry when known; otherwise read from the checkout, which
    is the same answer without having to maintain it by hand.
    """
    if subject.commit_date:
        return subject.commit_date
    if workspace is None or not subject.commit:
        return ""
    try:
        out = subprocess.run(
            ["git", "show", "-s", "--format=%cd", "--date=short", subject.commit],
            cwd=workspace, capture_output=True, text=True, timeout=60)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _run_config(subject: Subject, arm: Arm, timeout_s: int, pip_cache: str | None,
                max_passes: int = 20, on_date: str = "") -> RunConfig:
    return RunConfig(
        commit_date=on_date,
        repo=subject.name, arm=arm.name, tiers=arm.tiers, ordering=arm.ordering,
        ordering_seed=arm.ordering_seed, llm_model=arm.llm_spec,
        llm_temperature=0.0, llm_explanation_first=arm.explanation_first,
        llm_prompt_style=arm.prompt_style,
        installer=arm.installer, python_version=subject.python_version,
        max_passes=max_passes,
        python_executable=find_interpreter(subject.python_version) or "",
        timeout_s=timeout_s,
        validation_mode="baseline_relative" if arm.validate else "install_only",
        install_cache=True, build_isolation=True,
        constraint_graph=arm.constraint_graph, routing=arm.routing,
        measure_coverage=arm.measure_coverage,
        batching=arm.batching, batch_k=arm.batch_k,
        notes=arm.description,
    )


def _env_spec(subject: Subject, arm: Arm, workspace: Path, index: PackageIndex,
              pip_cache: str | None) -> EnvSpec:
    """How environments are built for this subject, including its build era.

    A package whose setup.py uses an API today's setuptools removed fails to
    build for a reason that belongs to this machine rather than to the project.
    The era is taken from the pins themselves, and only ever used as a retry
    after such a failure.
    """
    era: tuple[str, ...] = ()
    source = workspace / subject.requirements_file
    if source.exists():
        try:
            era = era_build_tools(index, ReqFile.load(source).pins(),
                                  on_date=subject.commit_date,
                                  python_version=subject.python_version)
        except Exception:
            era = ()
    return EnvSpec(python=_interpreter(subject), installer=arm.installer,
                   use_cache=True, build_isolation=True, cache_dir=pip_cache,
                   build_era_pins=era)


def _reference_baseline(reqfile: ReqFile, *, config, validator, spec, workspace,
                        index, rc, log):
    """The working state of the project, to judge a one-shot arm against.

    Identical to what the AURA arms establish, so "which tests already failed"
    and "how far behind was it" mean the same thing in every arm's numbers.
    """
    seen: dict[str, object] = {}

    def probe_once(candidate: ReqFile):
        result = probe(candidate, config=config, validator=validator, spec=spec,
                       env_path=workspace / ".aura_ref_venv", record_baseline=True)
        seen[candidate.to_text()] = result
        return result

    def closure_of(candidate: ReqFile) -> dict[str, str]:
        try:
            with Environment(workspace / ".aura_closure", spec) as env:
                return env.resolve_closure(candidate, workspace)
        except Exception:
            return {}

    frozen_path = config.get("FROZEN_BASELINE")
    outcome = establish_baseline(
        reqfile, probe=probe_once, closure_of=closure_of, index=index,
        commit_date=rc.commit_date, python_version=rc.python_version,
        repair=rc.repair_baseline,
        on_stage=lambda **kw: log.emit(Event.NOTE, note="baseline_stage", **kw),
        frozen=frozen_baseline(frozen_path))
    if not outcome.ok:
        return probe_once(reqfile), {}
    if freeze_once(frozen_path, outcome.reqfile):
        log.emit(Event.NOTE, note="baseline_frozen", path=frozen_path, digest=outcome.digest)
    if outcome.repaired:
        log.emit(Event.NOTE, note="baseline_repaired", stage=outcome.stage,
                 packages=outcome.repaired, on_date=rc.commit_date)
    log.emit(Event.NOTE, note="baseline_established", stage=outcome.stage,
             digest=outcome.digest, repaired=len(outcome.repaired))
    # The winning rung was already installed and validated while the ladder was
    # walked, and it recorded the reference validation; probing it again would
    # spend a second full install to learn the same thing.
    result = seen.get(outcome.reqfile.to_text()) or probe_once(outcome.reqfile)
    return result, {"pins": outcome.reqfile.pins(), "stage": outcome.stage,
                    "digest": outcome.digest, "reqfile": outcome.reqfile}


def run_baseline_arm(subject: Subject, arm: Arm, workspace: Path, log_path: Path,
                     *, index: PackageIndex, timeout_s: int,
                     pip_cache: str | None = None,
                     max_passes: int = 20) -> ArmResult:
    """One-shot arms: install a requirement set once and validate it."""
    rc = _run_config(subject, arm, timeout_s, pip_cache, max_passes,
                     on_date=commit_date(subject, workspace))
    log = RunLog(log_path, rc, echo=True)
    started = time.time()

    source = (subject.primary_requirements_file if arm.upgrade_source == "primary"
              else subject.requirements_file)
    source_path = workspace / source
    if arm.upgrade_source == "primary" and not source_path.exists():
        ensure_primary_requirements(subject, workspace)
    if not source_path.exists():
        log.emit(Event.NOTE, note="missing_requirements", path=str(source_path))
        log.finish(Outcome.ERROR, reason=f"{source} not found")
        return ArmResult(subject.name, arm.name, Outcome.ERROR.value,
                         reason=f"{source} not found", log_path=str(log_path))

    if arm.upgrade_source == "primary":
        _, derived = ensure_primary_requirements(subject, workspace)
        log.emit(Event.NOTE, note="primary_requirements",
                 derived=derived, path=str(source_path),
                 detail=("version constraints stripped from "
                         f"{subject.requirements_file} by the harness" if derived
                         else "shipped by the project"))

    reqfile = ReqFile.load(source_path)
    config = subject.agent_config(workspace)
    validator = Validator(config, tolerance=subject.metric_tolerance)
    spec = _env_spec(subject, arm, workspace, index, pip_cache)

    # Always measured against the subject's own pins. An arm that starts from
    # the unpinned specification has no pins of its own, and taking its
    # "before" from that file would report zero packages behind and leave the
    # per-package comparison with an empty denominator.
    original_path = workspace / subject.requirements_file
    original = (ReqFile.load(original_path).pins() if original_path.exists()
                else reqfile.pins())
    age_before = version_age_report(index, original)

    # Every arm is judged by the same rule. An upgrade arm is compared against
    # the project's own starting behaviour, exactly as AURA's candidates are:
    # judging it absolutely ("all tests must pass") while AURA is forgiven the
    # failures a subject already had would penalise the baselines for something
    # AURA is not penalised for, and that bias runs one way.
    #
    # The reference is the *established* baseline, the same one the AURA arms
    # start from. For a project whose declared file no longer installs, using
    # that file here would abort the arm for want of a yardstick -- reported as
    # a failure, though the arm never got to try its upgrade. What the arm
    # installs is untouched by this: stripping every constraint resolves to the
    # newest releases whatever the reference was.
    baseline_stage = ""
    if arm.name != "baseline" and arm.validate:
        reference, baseline_info = _reference_baseline(
            ReqFile.load(original_path), config=config, validator=validator,
            spec=spec, workspace=workspace, index=index, rc=rc, log=log)
        original = baseline_info.get("pins", original)
        baseline_stage = baseline_info.get("stage", "")
        age_before = version_age_report(index, original)
        log.attempt(package="<reference>", tier="baseline", plan=original,
                    install_ok=reference.install_ok,
                    validate_ok=bool(reference.validation
                                     and reference.validation.passed),
                    duration_s=reference.duration_s, accepted=reference.accepted,
                    error_class=reference.error_class,
                    error_excerpt=reference.stderr[-2000:],
                    **attempt_details(reference))
        if not reference.accepted:
            reason = f"baseline precondition: {reference.error_class}"
            log.finish(Outcome.FAILED, reason=reason,
                       version_age={"before": age_before, "after": age_before})
            return ArmResult(subject.name, arm.name, Outcome.FAILED.value,
                             reason=reason,
                             wall_clock_s=round(time.time() - started, 2),
                             log_path=str(log_path), version_age_before=age_before)

    result = probe(reqfile, config=config, validator=validator, spec=spec,
                   env_path=workspace / ".aura_arm_venv",
                   record_baseline=(arm.name == "baseline" or not arm.validate))

    stage = baseline_stage if arm.name != "baseline" else (
        "declared" if result.accepted else "")
    if arm.name == "baseline" and result.accepted:
        # The declared file works as written: that is the subject's baseline.
        if freeze_once(config.get("FROZEN_BASELINE"), reqfile):
            log.emit(Event.NOTE, note="baseline_frozen",
                     path=config.get("FROZEN_BASELINE"), stage="declared")
    if arm.name == "baseline" and not result.accepted and rc.repair_baseline:
        # The declared file is dead, which is this arm's finding and stays its
        # outcome. Whether the subject is usable at all is a different question,
        # and the rest of the campaign depends on the answer.
        rebuilt, info = _reference_baseline(
            reqfile, config=config, validator=validator, spec=spec,
            workspace=workspace, index=index, rc=rc, log=log)
        stage = info.get("stage", "") if rebuilt.accepted else ""
        log.emit(Event.NOTE, note="baseline_recoverable", stage=stage,
                 detail=("the declared requirements do not install, but a "
                         f"baseline was established ({stage})" if stage else
                         "no baseline could be established at all"))

    log.attempt(package="<arm>", tier="all",
                plan=result.frozen or reqfile.pins(),
                install_ok=result.install_ok,
                validate_ok=bool(result.validation and result.validation.passed),
                duration_s=result.duration_s, accepted=result.accepted,
                error_class=result.error_class, error_excerpt=result.stderr[-2000:],
                    **attempt_details(result))
    if result.validation is not None:
        log.validation(phase="arm", passed=result.validation.passed,
                       failing=list(result.validation.failing),
                       new_failures=sorted(result.new_failures or []),
                       metrics=result.validation.metrics,
                       duration_s=result.validation.duration_s, mode=rc.validation_mode)

    # Measured over the packages the subject declares, not the whole installed
    # environment. An arm that starts from the unpinned specification freezes
    # dozens of transitive packages the file never named, and counting those
    # compares it against the AURA arms on a different denominator -- which on
    # term-image made the untouched baseline look more modern than the arm that
    # advanced eleven packages.
    measured = {name: version for name, version in (result.frozen or {}).items()
                if name in original} or (result.frozen or reqfile.pins())
    # And only when the environment was kept: a rejected one is installed
    # nowhere, so the arm ends at the baseline it started from.
    age_after = (version_age_report(index, measured)
                 if result.accepted and result.frozen else age_before)

    # What this arm actually advanced: packages whose resolved version differs
    # from the subject's own pin. Reporting the whole frozen environment as
    # "updated" would credit an arm for every package it merely installed, and
    # recording nothing would credit it for none -- pip_upgrade genuinely moves
    # most of them, and the per-package comparison has to see that.
    advanced: dict[str, list] = {}
    if result.accepted and arm.name != "baseline":
        for name, version in (result.frozen or {}).items():
            before = original.get(name)
            if before and version != before:
                advanced[name] = [before, version]

    if not result.accepted:
        outcome = Outcome.FAILED
    elif arm.name == "baseline":
        # The inclusion gate: it establishes that a working baseline exists,
        # it does not advance anything.
        outcome = Outcome.KEPT_BASELINE
    elif advanced:
        outcome = Outcome.MODERNIZED
    else:
        outcome = Outcome.KEPT_BASELINE
    log.finish(outcome, updated=advanced,
               reason="" if result.accepted else result.error_class,
               version_age={"before": age_before, "after": age_after})

    return ArmResult(
        subject.name, arm.name, outcome.value,
        updated=advanced,
        reason="" if result.accepted else result.error_class,
        wall_clock_s=round(time.time() - started, 2), log_path=str(log_path),
        counters=log.counters, version_age_before=age_before, version_age_after=age_after,
        baseline_stage=stage,
    )


def final_requirements_path(log_path: Path) -> Path:
    """Where an arm's final requirements file is kept, next to its log:
    ``<subject>__<arm>.requirements.txt``."""
    log_path = Path(log_path)
    return log_path.with_name(log_path.stem + ".requirements.txt")


def compile_requirements(workspace: Path, source: Path, target: Path,
                         *, timeout_s: int = 1800) -> tuple[bool, str]:
    """Resolve an unpinned specification to a coherent pinned set with pip-tools.

    pip-tools is installed into a throwaway environment rather than assumed to
    be on the machine, so this baseline needs nothing set up in advance and
    resolves identically wherever it runs.
    """
    import venv as _venv

    env_path = Path(workspace) / ".aura_piptools"
    shutil.rmtree(env_path, ignore_errors=True)
    try:
        _venv.create(env_path, with_pip=True, clear=True)
    except Exception as exc:
        return False, f"could not create pip-tools environment: {exc}"

    python = str(env_path / ("Scripts" if _WINDOWS else "bin") / "python")
    install = subprocess.run([python, "-m", "pip", "install", "-q", "pip-tools"],
                             capture_output=True, text=True, timeout=timeout_s)
    if install.returncode != 0:
        return False, f"pip-tools install failed: {install.stderr[-400:]}"

    out = subprocess.run(
        [python, "-m", "piptools", "compile", "--quiet", "--no-header",
         "--resolver=backtracking", "--output-file", str(target), str(source)],
        cwd=str(workspace), capture_output=True, text=True, timeout=timeout_s)
    shutil.rmtree(env_path, ignore_errors=True)
    if out.returncode != 0 or not Path(target).exists():
        return False, f"pip-compile failed: {(out.stderr or out.stdout)[-800:]}"
    return True, ""


def run_compile_arm(subject: Subject, arm: Arm, workspace: Path, log_path: Path,
                    *, index: PackageIndex, timeout_s: int,
                    pip_cache: str | None = None,
                    max_passes: int = 20) -> ArmResult:
    """Resolve the unpinned specification properly, then validate it once.

    A stronger resolver baseline than ``pip install --upgrade``: pip-tools
    backtracks to a mutually consistent set rather than taking whatever the
    first pass produces. Still a single proposal with no test feedback, which
    is the line between it and AURA.
    """
    rc = _run_config(subject, arm, timeout_s, pip_cache, max_passes,
                     on_date=commit_date(subject, workspace))
    log = RunLog(log_path, rc, echo=True)
    started = time.time()

    ensure_primary_requirements(subject, workspace)
    source = workspace / subject.primary_requirements_file
    if not source.exists():
        log.finish(Outcome.ERROR, reason="no unpinned specification to compile")
        return ArmResult(subject.name, arm.name, Outcome.ERROR.value,
                         reason="no unpinned specification", log_path=str(log_path))

    target = workspace / "compiled_requirements.txt"
    ok, error = compile_requirements(workspace, source, target)
    log.emit(Event.NOTE, note="pip_compile", succeeded=ok, detail=error)
    if not ok:
        log.finish(Outcome.ERROR, reason=error)
        return ArmResult(subject.name, arm.name, Outcome.ERROR.value, reason=error,
                         wall_clock_s=round(time.time() - started, 2),
                         log_path=str(log_path))

    original = ReqFile.load(workspace / subject.requirements_file)
    age_before = version_age_report(index, original.pins())
    compiled = ReqFile.load(target)
    config = subject.agent_config(workspace)
    validator = Validator(config, tolerance=subject.metric_tolerance)
    spec = _env_spec(subject, arm, workspace, index, pip_cache)

    # Same rule as every other arm: relative to the project's own baseline.
    reference = probe(ReqFile.load(workspace / subject.requirements_file),
                      config=config, validator=validator, spec=spec,
                      env_path=workspace / ".aura_ref_venv", record_baseline=True)
    if not reference.accepted:
        reason = f"baseline precondition: {reference.error_class}"
        log.finish(Outcome.FAILED, reason=reason,
                   version_age={"before": age_before, "after": age_before})
        return ArmResult(subject.name, arm.name, Outcome.FAILED.value, reason=reason,
                         wall_clock_s=round(time.time() - started, 2),
                         log_path=str(log_path), version_age_before=age_before)

    result = probe(compiled, config=config, validator=validator, spec=spec,
                   env_path=workspace / ".aura_arm_venv", record_baseline=False)
    log.attempt(package="<arm>", tier="all", plan=compiled.pins(),
                install_ok=result.install_ok,
                validate_ok=bool(result.validation and result.validation.passed),
                duration_s=result.duration_s, accepted=result.accepted,
                error_class=result.error_class, error_excerpt=result.stderr[-2000:],
                    **attempt_details(result))

    before = original.pins()
    advanced = {n: [before.get(n, ""), v] for n, v in (result.frozen or {}).items()
                if n in before and v != before.get(n)} if result.accepted else {}
    # What the arm *ends with*, which is the baseline whenever its environment
    # was rejected: a rejected environment is not installed anywhere. The
    # frozen set was being reported regardless, so test-flask's baseline arm
    # failed validation, kept the original pins, and recorded "0.0% -> 57.14%"
    # -- the version age of the environment it had just thrown away.
    age_after = (version_age_report(index, result.frozen)
                 if result.accepted and result.frozen else age_before)
    outcome = (Outcome.MODERNIZED if result.accepted and advanced
               else Outcome.KEPT_BASELINE if result.accepted else Outcome.FAILED)
    log.finish(outcome, updated=advanced,
               reason="" if result.accepted else result.error_class,
               version_age={"before": age_before, "after": age_after})
    return ArmResult(
        subject.name, arm.name, outcome.value, updated=advanced,
        reason="" if result.accepted else result.error_class,
        wall_clock_s=round(time.time() - started, 2), log_path=str(log_path),
        counters=log.counters, version_age_before=age_before,
        version_age_after=age_after,
    )


def run_oneshot_arm(subject: Subject, arm: Arm, workspace: Path, log_path: Path,
                    *, index: PackageIndex, timeout_s: int,
                    pip_cache: str | None = None,
                    max_passes: int = 20) -> ArmResult:
    """Ask the model for the whole upgraded file once, then install and test it.

    Starts from the established baseline, the same one the AURA arms search
    from, and is scored by the same rule as pip_upgrade: the proposal is kept
    only if the project's tests still pass, and otherwise the arm ends where it
    started. What separates it from the aura_llm arms is that nothing comes
    back to the model -- no conflict report, no failing test.
    """
    from evaluation.oneshot import (build_prompt, nonexistent, offered_versions,
                                    parse_proposal)

    rc = _run_config(subject, arm, timeout_s, pip_cache, max_passes,
                     on_date=commit_date(subject, workspace))
    log = RunLog(log_path, rc, echo=True)
    started = time.time()

    def finish(outcome, reason, age_before, age_after, updated=None):
        log.finish(outcome, updated=updated or {}, reason=reason,
                   version_age={"before": age_before, "after": age_after})
        return ArmResult(subject.name, arm.name, outcome.value,
                         updated=updated or {}, reason=reason,
                         wall_clock_s=round(time.time() - started, 2),
                         log_path=str(log_path), counters=log.counters,
                         version_age_before=age_before, version_age_after=age_after)

    original_path = workspace / subject.requirements_file
    config = subject.agent_config(workspace)
    validator = Validator(config, tolerance=subject.metric_tolerance)
    spec = _env_spec(subject, arm, workspace, index, pip_cache)

    reference, info = _reference_baseline(
        ReqFile.load(original_path), config=config, validator=validator,
        spec=spec, workspace=workspace, index=index, rc=rc, log=log)
    baseline = info.get("reqfile") or ReqFile.load(original_path)
    before = baseline.pins()
    age_before = version_age_report(index, before)
    if not reference.accepted:
        return finish(Outcome.FAILED, f"baseline precondition: {reference.error_class}",
                      age_before, age_before)

    offers = {name: offered_versions(index, name, version, rc.python_version)
              for name, version in before.items()}
    prompt = build_prompt(before, offers, rc.python_version)
    llm = build_client(arm.llm_spec, temperature=0.0)
    # Room for the whole file even on the largest subject, plus whatever the
    # model says around it.
    response = llm.generate(prompt, max_output_tokens=4096)
    proposed, invented = parse_proposal(response.text, set(before))
    hallucinated = nonexistent(index, proposed)
    usage = response.usage
    # Counted exactly as the AURA arms count theirs, so llm_calls,
    # prompt_tokens and completion_tokens in result.json compare one answer
    # with an iterated search on the same footing.
    log.llm_call(usage, purpose="oneshot",
                 proposed=[f"{n}=={v}" for n, v in proposed.items()],
                 raw=(response.text or "")[-8000:])
    # The model's own count of what it read, against ours. Ollama drops the
    # front of a prompt longer than its context, which here would be the
    # requirements themselves -- so a large shortfall is recorded, not assumed
    # away.
    counted = getattr(usage, "prompt_tokens", 0) or 0
    expected = len(prompt) // 4
    log.emit(Event.NOTE, note="oneshot_proposal",
             prompt_chars=len(prompt), prompt_tokens=counted,
             expected_prompt_tokens=expected,
             possibly_truncated=bool(counted and counted < 0.7 * expected),
             completion_tokens=getattr(usage, "completion_tokens", 0),
             latency_s=getattr(usage, "latency_s", 0),
             llm_ok=getattr(usage, "ok", True), llm_error=getattr(usage, "error", ""),
             proposed=proposed, invented_packages=invented,
             nonexistent_versions=hallucinated,
             raw_response=(response.text or "")[-8000:])

    if not getattr(usage, "ok", True) or not response.text:
        return finish(Outcome.ERROR, f"model call failed: {getattr(usage, 'error', '')}",
                      age_before, age_before)
    changed = {n: v for n, v in proposed.items() if v != before.get(n)}
    if not changed:
        return finish(Outcome.KEPT_BASELINE, "the model proposed no upgrade",
                      age_before, age_before)

    # One attempt, taken as proposed: a version that was never released goes
    # into the file and fails the install, as it would for anyone who took the
    # model at its word.
    candidate = baseline.with_pins(changed)
    result = probe(candidate, config=config, validator=validator, spec=spec,
                   env_path=workspace / ".aura_arm_venv", record_baseline=False)
    log.attempt(package="<arm>", tier="llm_oneshot", plan=candidate.pins(),
                install_ok=result.install_ok,
                validate_ok=bool(result.validation and result.validation.passed),
                duration_s=result.duration_s, accepted=result.accepted,
                error_class=result.error_class, error_excerpt=result.stderr[-2000:],
                **attempt_details(result))
    if not result.accepted:
        return finish(Outcome.FAILED, result.error_class, age_before, age_before)

    frozen = result.frozen or candidate.pins()
    advanced = {n: [before[n], v] for n, v in frozen.items()
                if n in before and v != before[n]}
    age_after = version_age_report(index, {n: frozen.get(n, v) for n, v in before.items()})
    candidate.save(final_requirements_path(log_path))
    return finish(Outcome.MODERNIZED if advanced else Outcome.KEPT_BASELINE, "",
                  age_before, age_after, advanced)


def run_renovate_replay_arm(subject: Subject, arm: Arm, workspace: Path, log_path: Path,
                            *, index: PackageIndex, timeout_s: int,
                            pip_cache: str | None = None,
                            max_passes: int = 20) -> ArmResult:
    """Renovate's own proposals, tested and merged as its pull requests were.

    The proposals come from the real Renovate CLI, run offline on the frozen
    baseline with the renovate arm's configuration
    (scripts/renovate_proposals.py writes <subject>__renovate.proposals.json):
    every branch it would open, minor and major separately, grouped as Renovate
    groups them. What GitHub did with them is replayed here: each round, every
    open branch is tested against the current main, with the install, tests and
    acceptance rule every other arm uses; every branch that passed is merged;
    a branch the merges have already reached is closed, as Renovate closes it.
    Rounds repeat until one merges nothing, and the final main is tested once,
    as the GitHub arms test it.
    """
    from packaging.version import InvalidVersion, Version

    rc = _run_config(subject, arm, timeout_s, pip_cache, max_passes,
                     on_date=commit_date(subject, workspace))
    log = RunLog(log_path, rc, echo=True)
    started = time.time()

    def finish(outcome, reason, age_before, age_after, updated=None):
        log.finish(outcome, updated=updated or {}, reason=reason,
                   version_age={"before": age_before, "after": age_after})
        return ArmResult(subject.name, arm.name, outcome.value,
                         updated=updated or {}, reason=reason,
                         wall_clock_s=round(time.time() - started, 2),
                         log_path=str(log_path), counters=log.counters,
                         version_age_before=age_before, version_age_after=age_after)

    def newer(a: str, b: str) -> bool:
        try:
            return Version(a) > Version(b)
        except InvalidVersion:
            return a != b

    original_path = workspace / subject.requirements_file
    config = subject.agent_config(workspace)
    validator = Validator(config, tolerance=subject.metric_tolerance)
    spec = _env_spec(subject, arm, workspace, index, pip_cache)

    reference, info = _reference_baseline(
        ReqFile.load(original_path), config=config, validator=validator,
        spec=spec, workspace=workspace, index=index, rc=rc, log=log)
    baseline = info.get("reqfile") or ReqFile.load(original_path)
    before = baseline.pins()
    age_before = version_age_report(index, before)
    if not reference.accepted:
        return finish(Outcome.FAILED, f"baseline precondition: {reference.error_class}",
                      age_before, age_before)

    proposals_path = log_path.parent / f"{subject.name}__renovate.proposals.json"
    if not proposals_path.exists():
        return finish(Outcome.ERROR, f"no Renovate proposals: run scripts/"
                      f"renovate_proposals.py {subject.name}", age_before, age_before)
    branches = json.loads(proposals_path.read_text())["branches"]
    log.emit(Event.NOTE, note="renovate_proposals", branches=len(branches),
             grouped=sum(len(b["updates"]) > 1 for b in branches),
             proposals=[{b["branch"]: {u["package"]: u["to"] for u in b["updates"]}}
                        for b in branches])

    main = dict(before)
    open_ = {b["branch"]: {u["package"]: u["to"] for u in b["updates"]
                           if u["package"] in main} for b in branches}
    open_ = {name: ups for name, ups in open_.items() if ups}
    merged_order: list[str] = []
    env_path = workspace / ".aura_arm_venv"
    for round_no in range(1, 100):
        # Closed by Renovate once main has reached every version it proposes
        # (a minor branch after its major was merged).
        for name in [n for n, ups in open_.items()
                     if not any(newer(v, main[p]) for p, v in ups.items())]:
            log.emit(Event.NOTE, note="renovate_branch_closed", branch=name,
                     round=round_no, reason="main already at or past its versions")
            del open_[name]
        # One at a time: a branch is tested on the main the merges before it
        # left, and merged at once if it passes. Testing a round together and
        # merging every pass merged sphinx 2.4.5 and docutils 0.23, each green
        # alone and unresolvable together; on GitHub the first merge leaves
        # the second conflicting, and Renovate rebases it and CI runs again.
        green = []
        for name in sorted(open_):
            ups = open_[name]
            if not any(newer(v, main[p]) for p, v in ups.items()):
                continue    # reached by an earlier merge this round
            candidate = baseline.with_pins({**main, **ups})
            result = probe(candidate, config=config, validator=validator, spec=spec,
                           env_path=env_path, record_baseline=False)
            log.attempt(package=",".join(sorted(ups)), tier="renovate", plan=ups,
                        install_ok=result.install_ok,
                        validate_ok=bool(result.validation and result.validation.passed),
                        duration_s=result.duration_s, accepted=result.accepted,
                        error_class=result.error_class,
                        error_excerpt=result.stderr[-2000:], branch=name,
                        round=round_no, **attempt_details(result))
            if result.accepted:
                green.append(name)
                for pkg, version in open_.pop(name).items():
                    if newer(version, main[pkg]):
                        main[pkg] = version
                merged_order.append(name)
        if not green:
            break
        log.emit(Event.NOTE, note="merge_round", round=round_no, merged_now=len(green),
                 merged_total=len(merged_order), still_open=sorted(open_))

    final = baseline.with_pins(main)
    updated = {n: [before[n], v] for n, v in main.items() if v != before[n]}
    log.emit(Event.NOTE, note="github_merge_order", order=merged_order)
    if updated:
        result = probe(final, config=config, validator=validator, spec=spec,
                       env_path=env_path, record_baseline=False)
        log.attempt(package="<main>", tier="renovate", plan=main,
                    install_ok=result.install_ok,
                    validate_ok=bool(result.validation and result.validation.passed),
                    duration_s=result.duration_s, accepted=result.accepted,
                    error_class=result.error_class, error_excerpt=result.stderr[-2000:],
                    **attempt_details(result))
        if not result.accepted:
            # Merges that each passed but break together: the project is left
            # broken, as main was on GitHub.
            final.save(final_requirements_path(log_path))
            return finish(Outcome.FAILED, f"the merged main fails: {result.error_class}",
                          age_before, version_age_report(index, main), updated)
    final.save(final_requirements_path(log_path))
    return finish(Outcome.MODERNIZED if updated else Outcome.KEPT_BASELINE,
                  "" if updated else "no Renovate branch passed",
                  age_before, version_age_report(index, main), updated)


def run_aura_arm(subject: Subject, arm: Arm, workspace: Path, log_path: Path,
                 *, index: PackageIndex, timeout_s: int,
                 pip_cache: str | None = None,
                 max_passes: int = 20) -> ArmResult:
    rc = _run_config(subject, arm, timeout_s, pip_cache, max_passes,
                     on_date=commit_date(subject, workspace))
    log = RunLog(log_path, rc, echo=True)

    config = subject.agent_config(workspace)
    config["REQUIRE_METRIC_PARITY"] = arm.require_metric_parity
    if not arm.validate:
        # The install-only regime: accept whatever the resolver accepts.
        config["VALIDATION_CONFIG"] = {"type": "none"}

    agent = AuraAgent(
        config, run_config=rc,
        llm=build_client(arm.llm_spec, temperature=0.0),
        index=index, run_log=log,
        env_spec=_env_spec(subject, arm, workspace, index, pip_cache),
        workdir=workspace,
    )
    started = time.time()
    try:
        result = agent.run()
    except Exception as exc:  # a crash is an outcome, not a silent zero
        log.emit(Event.NOTE, note="agent_crashed", error=f"{type(exc).__name__}: {exc}")
        log.finish(Outcome.ERROR, reason=f"{type(exc).__name__}: {exc}")
        return ArmResult(subject.name, arm.name, Outcome.ERROR.value,
                         reason=str(exc), wall_clock_s=round(time.time() - started, 2),
                         log_path=str(log_path), error=f"{type(exc).__name__}: {exc}")

    if not arm.validate:
        _post_hoc_validate(subject, arm, workspace, log, pip_cache, index)

    return ArmResult(
        subject.name, arm.name, result.outcome.value,
        updated={k: list(v) for k, v in result.updated.items()},
        reason=result.reason, wall_clock_s=round(time.time() - started, 2),
        log_path=str(log_path), counters=result.counters,
        version_age_before=result.version_age_before,
        version_age_after=result.version_age_after,
        baseline_stage=result.baseline_stage,
    )


def _post_hoc_validate(subject: Subject, arm: Arm, workspace: Path,
                       log: RunLog, pip_cache: str | None,
                       index: PackageIndex) -> None:
    """Run the project's tests against an install-only arm's final answer.

    The install-only arm accepts whatever the resolver accepts, exactly as a
    static safe-update analysis would. Validating its output afterwards -- with
    the result never fed back into the search -- measures how often a
    structurally valid configuration is actually broken. That is the number
    Reviewer A asked for: how often the unverified regime produces a
    false positive that execution-validated search avoids.
    """
    config = subject.agent_config(workspace)
    validator = Validator(config, tolerance=subject.metric_tolerance)
    spec = _env_spec(subject, arm, workspace, index, pip_cache)
    try:
        final = ReqFile.load(workspace / subject.requirements_file)
        result = probe(final, config=config, validator=validator, spec=spec,
                       env_path=workspace / ".aura_posthoc_venv", record_baseline=True)
    except Exception as exc:
        log.emit(Event.NOTE, note="post_hoc_validation_error", error=str(exc))
        return
    log.validation(
        phase="post_hoc", passed=result.accepted,
        failing=list(result.validation.failing) if result.validation else [],
        new_failures=[], metrics=result.validation.metrics if result.validation else {},
        duration_s=result.duration_s, mode="post_hoc_unverified_regime",
    )
    log.emit(Event.NOTE, note="post_hoc_verdict",
             passed=result.accepted, error_class=result.error_class)


@contextlib.contextmanager
def subject_environment(subject: Subject):
    """The subject's declared environment variables, for the length of one arm.

    Set on the process rather than threaded to each test launch, because every
    install, probe and test run inherits it -- including the probes that run
    concurrently -- and one cluster task is one arm of one subject. Restored
    afterwards so a local run of several subjects does not leak one's settings
    into the next.
    """
    saved = {k: os.environ.get(k) for k in subject.environment}
    os.environ.update({k: str(v) for k, v in subject.environment.items()})
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def frozen_baseline_path(results_dir: str | Path, subject: str) -> Path:
    """Where a subject's baseline is frozen, beside its results."""
    return Path(results_dir) / subject / f"{subject}.baseline.txt"


def run_arm(subject: Subject, arm: Arm, *, results_dir: str | Path,
            workspaces_dir: str | Path, repo_cache: str | Path | None = None,
            timeout_s: int = 6 * 3600, pip_cache: str | None = None,
            max_passes: int = 20) -> ArmResult:
    """Prepare a clean workspace and execute one arm on one subject."""
    results_dir, workspaces_dir = Path(results_dir), Path(workspaces_dir)
    log_path = log_path_for(results_dir, subject.name, arm.name)
    workspace = workspaces_dir / subject.name / arm.name

    if arm.upgrade_source == "github":
        # Collected from GitHub by scripts/real_dependabot.py. Never touch its
        # result files from here.
        return ArmResult(subject.name, arm.name, Outcome.ERROR.value,
                         reason="runs on GitHub: scripts/real_dependabot.py",
                         log_path=str(log_path))

    try:
        _interpreter(subject)          # fail before cloning if the Python is absent
    except InterpreterMissing as exc:
        return ArmResult(subject.name, arm.name, Outcome.ERROR.value,
                         reason=str(exc), log_path=str(log_path), error=str(exc))

    try:
        prepare_workspace(subject, workspace, Path(repo_cache) if repo_cache else None)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return ArmResult(subject.name, arm.name, Outcome.ERROR.value,
                         reason="workspace_preparation_failed",
                         error=str(exc), log_path=str(log_path))

    # Every arm of a subject starts from one baseline, frozen by the first --
    # or seeded from baselines/<subject>.txt (tracked in git) when a subject's
    # baseline was fixed by hand, as for the subjects whose Dependabot run
    # started from a different reconstruction than their arms.
    frozen = frozen_baseline_path(results_dir, subject.name)
    seed = Path(__file__).resolve().parent.parent / "baselines" / f"{subject.name}.txt"
    if not frozen.exists() and seed.exists():
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(seed, frozen)
    subject.frozen_baseline = str(frozen)

    if arm.kind == "aura":
        runner = run_aura_arm
    elif arm.upgrade_source == "compiled":
        runner = run_compile_arm
    elif arm.upgrade_source == "llm_oneshot":
        runner = run_oneshot_arm
    elif arm.upgrade_source == "renovate_replay":
        runner = run_renovate_replay_arm
    else:
        runner = run_baseline_arm
    with subject_environment(subject):
        result = runner(subject, arm, workspace, log_path,
                        index=PackageIndex(), timeout_s=timeout_s, pip_cache=pip_cache,
                        max_passes=max_passes)
    if runner is run_aura_arm:
        # AURA rewrites the subject's requirements file in place as it goes;
        # what is left there at the end is the configuration it accepted.
        edited = workspace / subject.requirements_file
        if edited.exists():
            shutil.copy(edited, final_requirements_path(log_path))

    (results_dir / subject.name).mkdir(parents=True, exist_ok=True)
    log_path.with_suffix(".result.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return result


def run_subject(subject: Subject, arms: list[Arm], **kwargs) -> list[ArmResult]:
    """Run every arm on one subject, stopping if no working baseline exists.

    The precondition is that *some* working environment exists to measure
    against, not that the project's declared file still installs: most published
    projects pin part of their dependencies and no longer install at all, and
    for those the baseline is reconstructed at the project's commit date. The
    baseline arm reports the declared file, so its failure is evidence about the
    repository rather than a reason to skip the subject; what stops the run is
    the baseline arm reporting that nothing could be established.
    """
    results: list[ArmResult] = []
    for arm in arms:
        result = run_arm(subject, arm, **kwargs)
        results.append(result)
        if arm.name == "baseline" and not result.passed and not result.baseline_stage:
            break
    return results
