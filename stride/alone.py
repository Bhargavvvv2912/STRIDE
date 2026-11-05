"""agent_alone: a coding agent upgrades the project on its own.

It gets the repository, a working environment, pip (capped at the measurement
date) and one instruction: upgrade the declared dependencies as far as you can,
changing code and test setup as needed, with the tests still passing. It picks
the versions itself and may edit the requirements file.

After each call the harness checks the result the same way as every other arm:
a fresh environment from the agent's requirements file, the project's tests
against the same baseline, the test rule, the structure check. Accepted states
are kept; the agent is then asked to go further. Calls stop when the agent's
answer stops improving (``patience`` calls in a row), a call changes nothing,
or ``max_attempts`` calls were made. The final state is the last accepted one.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from pathlib import Path

from aura.index import MEASURED_AS_OF, version_age_report
from aura.reqfile import ReqFile
from aura.validation import ValidationResult, Validator

from stride import guard
from stride.repair import build_environment, code_diff, diff_stats, git


def alone_prompt(*, project: str, requirements: str, python: str, test_command: str,
                 known_failing: list[str], pins: dict[str, str]) -> str:
    listed = "\n".join(f"  - {k}=={v}" for k, v in sorted(pins.items()))
    already = ""
    if known_failing:
        already = ("\nThese tests already fail before any upgrade and are not part of "
                   "the task:\n" + "\n".join(f"  - {t}" for t in known_failing[:20]) + "\n")
    return f"""Upgrade the dependencies of the project `{project}` as far as you can.

The project declares its dependencies in `{requirements}`:
{listed}

Goal: move as many of these packages as possible to newer releases, ideally
their latest, while the project's tests keep passing. Change the project's code
where a new release needs it. Python stays at {python}; releases newer than
{MEASURED_AS_OF} are not available (pip is capped at that date).

You may:
  - edit `{requirements}` (keep every line an exact pin, `name==version`);
  - install packages with pip into the current environment to try versions;
  - change the project's source code, keeping its behaviour;
  - change how tests are set up (imports, fixtures, mocks, conftest.py) when an
    upgrade requires it.
You may not change what tests check (assertions, pytest.raises, test names),
delete tests, add skips, or delete existing functions, classes or methods.
{already}
Run the tests with:
    {test_command}

When you have upgraded as far as you can and the tests pass, finish with:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""


def _feedback(prompt: str, k: int, n: int, accepted: bool, reason: str, detail: list[str],
              mean_behind: float | None) -> str:
    if accepted:
        status = (f"Your last result was checked and accepted (mean releases behind "
                  f"latest: {mean_behind}). If any package can go further, continue; "
                  "otherwise finish.")
    else:
        lines = "\n".join(f"  - {d}" for d in detail[:30])
        status = (f"Your last result was not accepted: {reason}.\n{lines}\n"
                  "Fix this, then finish.")
    return f"{prompt}\n--- Call {k} of {n} ---\n{status}\n"


def agent_alone(run, baseline: dict, caps: dict[str, str]) -> None:
    ws, reqname = run.workspace, run.subject.requirements_file
    config = run.config()
    vc = config.get("VALIDATION_CONFIG", {})
    targets = vc.get("pytest_target") or vc.get("unittest_target") or "tests"
    targets = targets if isinstance(targets, list) else str(targets).split()
    validator = Validator(config, tolerance=run.subject.metric_tolerance)
    validator.baseline = ValidationResult(
        passed=not baseline.get("failing"), failing=set(baseline.get("failing") or []),
        metrics=dict(baseline.get("metrics") or {}), counts=dict(baseline.get("counts") or {}))
    test_command = (f"python -m unittest discover -v -s {targets[0]}"
                    if validator.kind == "unittest"
                    else f"python -m pytest {' '.join(targets)} -q")

    # The agent's working environment: the baseline, pip allowed, capped by date.
    work_env, error = build_environment(ws.parent / f".stride_env_{ws.name}", run.spec(),
                                        ReqFile.load(ws / reqname), ws, config)
    if work_env is None:
        run.summary["alone"] = {"stop": "baseline_install_failed", "error": error[-2000:]}
        return
    shims = guard.pip_blocker(ws.parent / f".stride_shims_{ws.name}", block_pip=False)
    agent_env = {
        "PATH": os.pathsep.join([str(shims), str(Path(work_env.python).parent),
                                 os.environ.get("PATH", "")]),
        "VIRTUAL_ENV": str(Path(work_env.python).parent.parent),
        "PIP_CONSTRAINT": run.constraints or "",
        "PAGER": "cat", "MANPAGER": "cat", "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    }
    protected = guard.protected_files(ws, targets, [reqname])
    test_rel = [str(p.relative_to(ws)) for p in protected if p.name != Path(reqname).name]
    tests_start = guard.test_shape(ws, test_rel)
    shape_start = guard.code_shape(ws, test_rel + [reqname])
    start_commit = git(ws, "rev-parse", "HEAD").strip()
    accepted_commit = start_commit
    prompt = base = alone_prompt(project=run.subject.name, requirements=reqname,
                                 python=run.subject.python_version,
                                 test_command=test_command,
                                 known_failing=sorted(baseline.get("failing") or []),
                                 pins=ReqFile.load(ws / reqname).pins())
    declared = list(ReqFile.load(ws / reqname).pins())
    best = version_age_report(run.index, ReqFile.load(ws / reqname).pins())[
        "mean_releases_behind"]
    stale, stop, attempts = 0, "", []
    for k in range(1, run.max_attempts + 1):
        here = run.out / f"alone{k}"
        here.mkdir(parents=True, exist_ok=True)
        (here / "task.md").write_text(prompt)
        agent_run = run.agent.run(prompt, ws, agent_env, here)
        git(ws, "add", "--intent-to-add", "--", ".")
        step = git(ws, "diff", "HEAD")
        (here / "patch.diff").write_text(step)
        entry = {"attempt": k, "agent": {**asdict(agent_run), "calls": 1},
                 "changed": bool(step)}
        if not step:
            entry["reason"] = "no_change"
            attempts.append(entry)
            stop = "no_change"
            break
        reason, detail, mean = _check(run, ws, reqname, declared, validator, config,
                                      tests_start, test_rel, shape_start, start_commit)
        entry.update({"reason": reason, "detail": detail[:30], "mean_behind": mean})
        attempts.append(entry)
        ok = reason == "accepted"
        if ok:
            git(ws, "add", "-A")
            git(ws, "commit", "-qm", f"STRIDE: agent_alone call {k} accepted")
            accepted_commit = git(ws, "rev-parse", "HEAD").strip()
            if mean is not None and (best is None or mean < best):
                best, stale = mean, 0
            else:
                stale += 1
        else:
            stale += 1
        if stale >= run.patience:
            stop = "no_progress"
            break
        if k == run.max_attempts:
            stop = "max_attempts"
            break
        prompt = _feedback(base, k + 1, run.max_attempts, ok, reason, detail, mean)
    # The final state is the last accepted one.
    git(ws, "reset", "-q", "--hard", accepted_commit)
    git(ws, "clean", "-fdq", "-e", ".aura*", "-e", ".stride*")
    work_env.destroy()
    shutil.rmtree(shims, ignore_errors=True)
    run.summary["alone_attempts"] = attempts
    run.summary["alone"] = {"stop": stop, "accepted_any": accepted_commit != start_commit}


def _check(run, ws: Path, reqname: str, declared: list[str], validator: Validator,
           config: dict, tests_start: dict, test_rel: list[str], shape_start: dict,
           start_commit: str) -> tuple[str, list[str], float | None]:
    """The harness's verdict on the agent's current state."""
    try:
        req = ReqFile.load(ws / reqname)
    except Exception as exc:
        return "unreadable_requirements", [str(exc)], None
    loose = [l.name for l in req.requirements() if l.name and not l.pinned_version]
    if loose:
        return "unpinned_requirements", loose, None
    weakened = guard.weakened_tests(tests_start, guard.test_shape(ws, test_rel))
    if weakened:
        return "weakened_tests", weakened, None
    diff = git(ws, "diff", start_commit)
    if guard.added_skips(diff):
        return "added_skips", guard.added_skips(diff), None
    lost = guard.lost_symbols(shape_start, guard.code_shape(ws, test_rel + [reqname]))
    if lost:
        return "removed_code", lost, None
    env, error = build_environment(ws.parent / f".stride_check_{ws.name}", run.spec(),
                                   req, ws, config)
    if env is None:
        return "install_failed", [error[-1500:]], None
    try:
        after = validator.run(env.python)
        ok, new, drifted, _ = validator.judge(after)
    finally:
        env.destroy()
    if not ok:
        return "tests_fail", sorted(new + drifted), None
    pins = {k: v for k, v in req.pins().items() if k in declared}
    mean = version_age_report(run.index, pins)["mean_releases_behind"]
    return "accepted", [], mean
