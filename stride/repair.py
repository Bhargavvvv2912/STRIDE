"""One repair round: cross one barrier with a code change, or roll back.

1. Pin the barrier's plan into the requirements and build a persistent
   environment with AURA's own installer.
2. Run the tests there to see what the new versions break.
3. Lock tests and requirements, block pip, and give the code agent the task.
4. Accept only if: protected files untouched, no skip markers added, installed
   packages unchanged, and AURA's judge sees no new failure against the
   original baseline. Otherwise restore the code.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aura.environment import Environment, EnvSpec
from aura.index import MEASURED_AS_OF
from aura.reqfile import ReqFile
from aura.validation import ValidationResult, Validator

from stride import guard, triage
from stride.coders import AgentRun, CodeAgent


@dataclass
class RepairResult:
    label: str
    accepted: bool = False
    reason: str = ""
    install_ok: bool = False
    failing_before: list[str] = field(default_factory=list)
    failing_after: list[str] = field(default_factory=list)
    new_failures: list[str] = field(default_factory=list)
    tampered: list[str] = field(default_factory=list)
    skips_added: int = 0
    packages_changed: list[str] = field(default_factory=list)
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    agent: dict = field(default_factory=dict)
    seconds: float = 0.0
    pins: dict[str, str] = field(default_factory=dict)
    triage: dict = field(default_factory=dict)
    transitive_pins: dict[str, str] = field(default_factory=dict)
    lost_symbols: list[str] = field(default_factory=list)    # code the patch removed
    added_symbols: list[str] = field(default_factory=list)   # new top-level code
    fixed_baseline: list[str] = field(default_factory=list)  # failed before any upgrade, pass now
    scope_creep: bool = False      # added code and made already-failing tests pass
    dropped: list[str] = field(default_factory=list)         # packages removed to cross


def git(root: Path, *args: str, check: bool = False) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                         check=check)
    return out.stdout


def ensure_repo(root: Path) -> None:
    """The workspace is a clone; give STRIDE an identity to commit its steps."""
    git(root, "config", "user.email", "stride@localhost")
    git(root, "config", "user.name", "STRIDE")


def code_diff(root: Path, protected: list[str]) -> str:
    excludes = [f":(exclude){p}" for p in protected]
    return git(root, "diff", "HEAD", "--", ".", *excludes)


def diff_stats(diff: str) -> tuple[int, int, int]:
    files = sum(1 for l in diff.splitlines() if l.startswith("diff --git"))
    added = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    return files, added, removed


def build_environment(path: Path, spec: EnvSpec, reqfile: ReqFile, workdir: Path,
                      config: dict) -> tuple[Environment | None, str]:
    """A persistent environment (not destroyed) the agent can run tests in."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    env = Environment(path, spec).create()
    output, code = env.install_requirements(reqfile, workdir)
    if code != 0:
        return None, output[-6000:]
    if config.get("IS_INSTALLABLE_PACKAGE"):
        output, code = env.install_project(
            config.get("VALIDATION_CONFIG", {}).get("project_dir") or str(workdir),
            config.get("PROJECT_EXTRAS", ""), editable=True)
        if code != 0:
            return None, output[-6000:]
    for command in config.get("ENVIRONMENT_SETUP") or []:
        output, code = env.run_setup(command.replace("{python}", env.python),
                                     cwd=str(workdir))
        if code != 0:
            return None, output[-6000:]
    return env, ""


def task_prompt(*, project: str, jump: str, test_command: str, failing: list[str],
                output_tail: str, call_sites: list[str], known_failing: list[str] = (),
                dropped: list[str] = ()) -> str:
    sites = "\n".join(f"  - {s}" for s in call_sites[:40]) or "  (none found)"
    fails = "\n".join(f"  - {t}" for t in failing[:40]) or "  (the suite does not run)"
    removed = ""
    if dropped:
        names = ", ".join(f"`{d}`" for d in dropped)
        removed = f"""
{names} has been uninstalled: it does not support the new versions, so the
upgrade is only possible without it. Replace the project's uses of it with
equivalent code (for example, re-implement the one function the project uses,
with the same behaviour and defaults). Do not change what the project does.
"""
    out_of_scope = ""
    if known_failing:
        listed = "\n".join(f"  - {t}" for t in known_failing[:20])
        out_of_scope = f"""
These tests already failed before the upgrade and are NOT part of this task.
Do not add code just to make them pass:
{listed}
"""
    return f"""The project `{project}` is being upgraded: {jump}.
The new versions are already installed. The project's tests now fail because the
project's code uses APIs that changed. Update the project's source code so that
these tests pass again with the installed versions, keeping the project's behaviour.
Do not delete or disable existing functions, classes or methods.
{removed}{out_of_scope}
Run the tests with:
    {test_command}

Tests that fail now (they passed before the upgrade):
{fails}

Files that import the upgraded packages:
{sites}

Last part of the test output:
```
{output_tail[-3500:]}
```

When the tests pass, finish with: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""


def call_sites(root: Path, packages: list[str]) -> list[str]:
    names = {p.replace("-", "_").lower() for p in packages}
    aliases = {"pytorch_lightning": "pytorch_lightning", "scikit_learn": "sklearn",
               "opencv_python": "cv2", "pillow": "PIL", "pyyaml": "yaml",
               "beautifulsoup4": "bs4", "elasticsearch": "elasticsearch"}
    names |= {aliases[n] for n in list(names) if n in aliases}
    hits = []
    for path in root.rglob("*.py"):
        if any(part.startswith(".") or part in ("site-packages", "tests", "test")
               for part in path.relative_to(root).parts):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            s = line.strip()
            if s.startswith(("import ", "from ")):
                mod = s.split()[1].split(".")[0].lower()
                if mod in names:
                    hits.append(str(path.relative_to(root)))
                    break
    return sorted(set(hits))


def unpinned(reqfile: ReqFile, names) -> ReqFile:
    """A copy with these packages' version specifiers removed, names kept, so
    the installer still installs them and picks the versions. (AURA's
    ``without_pins`` deletes the lines, which installs nothing for them.)"""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    drop = {canonicalize_name(n) for n in names}
    out = ReqFile.from_text(reqfile.to_text())
    for line in out.requirements():
        req = line.requirement
        if req is None or canonicalize_name(req.name) not in drop:
            continue
        extras = f"[{','.join(sorted(req.extras))}]" if req.extras else ""
        marker = f" ; {req.marker}" if req.marker else ""
        indent = line.raw[: len(line.raw) - len(line.raw.lstrip())]
        line._override = f"{indent}{req.name}{extras}{marker}{line.inline_comment}"
        line.requirement = Requirement(
            f"{req.name}{extras}" + (f"; {req.marker}" if req.marker else ""))
    return ReqFile.from_text(out.to_text())


def repair_round(*, label: str, project: str, workspace: Path, requirements_file: str,
                 pins: dict[str, str], moved: list[str], config: dict, spec: EnvSpec,
                 unpin: list[str] | None = None, index=None, python_version: str = "",
                 baseline: dict, agent: CodeAgent, out_dir: Path,
                 tolerance: float = 0.01, use_triage: bool = True,
                 drop: list[str] | None = None) -> RepairResult:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = RepairResult(label=label, pins=dict(pins))
    ensure_repo(workspace)

    req_path = workspace / requirements_file
    reqfile = ReqFile.load(req_path)
    present = {n.lower() for n in reqfile.names()}
    target = reqfile.with_pins(pins).with_added_pins(
        {k: v for k, v in pins.items() if k.lower() not in present})
    if unpin:
        target = unpinned(target, [u for u in unpin if u.lower() in present])
    if drop:   # an install barrier: the blocking package goes, its uses get replaced
        target = target.without_pins([d for d in drop if d.lower() in present])
        result.dropped = sorted(d for d in drop if d.lower() in present)
    target.save(req_path)
    git(workspace, "add", requirements_file)
    git(workspace, "commit", "-qm", f"STRIDE: pin {label}", "--allow-empty")

    env_path = workspace.parent / f".stride_env_{workspace.name}"
    env, error = build_environment(env_path, spec, target, workspace, config)
    if env is None:
        result.reason = "install_failed"
        (out_dir / "install_error.txt").write_text(error)
        git(workspace, "reset", "-q", "--hard", "HEAD~1")
        result.seconds = round(time.time() - started, 1)
        return _save(result, out_dir)
    result.install_ok = True
    if unpin:
        # The installer chose these; record and pin what it chose, so the next
        # round of version search starts from an exact configuration.
        norm = lambda n: n.lower().replace("_", "-").replace(".", "-")
        # --all: plain freeze hides setuptools, which would then stay unpinned.
        frozen = {norm(k): v for k, v in _freeze_all(env).items()}
        chosen = {u: frozen[norm(u)] for u in unpin if norm(u) in frozen}
        target = target.with_pins(chosen)
        target.save(req_path)
        result.pins.update(chosen)
        git(workspace, "add", requirements_file)
        git(workspace, "commit", "-q", "--amend", "--no-edit")

    validator = Validator(config, tolerance=tolerance)
    validator.baseline = ValidationResult(
        passed=not baseline.get("failing"), failing=set(baseline.get("failing") or []),
        metrics=dict(baseline.get("metrics") or {}), counts=dict(baseline.get("counts") or {}))
    before = validator.run(env.python)
    result.failing_before = sorted(before.failing)
    accepted, new, _, _ = validator.judge(before)
    if accepted:
        result.accepted, result.reason = True, "no_code_change_needed"
        env.destroy()
        result.seconds = round(time.time() - started, 1)
        return _save(result, out_dir)
    vc = config.get("VALIDATION_CONFIG", {})
    targets = vc.get("pytest_target") or vc.get("unittest_target") or "tests"
    targets = targets if isinstance(targets, list) else str(targets).split()

    # Triage: does the failure touch the project's code at all?
    traced = before.output
    if validator.kind == "pytest" and use_triage:
        traced = triage.run_with_tracebacks(env.python, targets, validator.cwd)
        (out_dir / "tests_before_tb.txt").write_text(traced[-30000:])
        test_dirs = [str(Path(t).parent) if t.endswith(".py") else t for t in targets]
        verdict = triage.classify(traced, test_dirs + ["tests", "test"], only=new,
                                  stepped=set(pins) | set(moved))
        result.triage = asdict(verdict)
        if verdict.kind == "dependency" and index is not None:
            pinned, passed = _try_transitive_pins(env, verdict.implicated, target, validator,
                                                  index, python_version, out_dir,
                                                  stepped=set(pins))
            if pinned:
                target = target.with_added_pins(pinned)
                target.save(req_path)
                result.transitive_pins = pinned
                result.pins.update(pinned)
                git(workspace, "add", requirements_file)
                git(workspace, "commit", "-q", "--amend", "--no-edit")
            if passed:
                result.accepted, result.reason = True, "transitive_pin"
                env.destroy()
                result.seconds = round(time.time() - started, 1)
                return _save(result, out_dir)
            if pinned:   # the agent starts from the moved package
                before = validator.run(env.python)
                _, new, _, _ = validator.judge(before)
                traced = triage.run_with_tracebacks(env.python, targets, validator.cwd)
                (out_dir / "tests_before_tb.txt").write_text(traced[-30000:])
    frozen_before = env.freeze()
    if validator.kind == "unittest":
        test_command = f"python -m unittest discover -v -s {targets[0]}"
    else:
        test_command = f"python -m pytest {' '.join(targets)} -q -x"
    protected = guard.protected_files(workspace, targets, [requirements_file])
    protected_rel = [str(p.relative_to(workspace)) for p in protected]

    shims = guard.pip_blocker(workspace.parent / f".stride_shims_{workspace.name}")
    agent_env = guard.agent_environment(str(Path(env.python).parent), shims)
    known_failing = sorted(baseline.get("failing") or [])
    task = task_prompt(project=project, jump=label, test_command=test_command,
                       failing=sorted(new or before.failing), output_tail=traced,
                       call_sites=call_sites(workspace, list(moved) + list(drop or [])),
                       known_failing=known_failing, dropped=result.dropped)
    (out_dir / "task.md").write_text(task)

    shape_before = guard.code_shape(workspace, protected_rel)
    with guard.Lock(workspace, protected) as lock:
        run: AgentRun = agent.run(task, workspace, agent_env, out_dir)
        tampered = lock.tampered()
    result.agent = asdict(run)

    diff = code_diff(workspace, protected_rel)
    (out_dir / "patch.diff").write_text(diff)
    result.files_changed, result.lines_added, result.lines_removed = diff_stats(diff)
    result.tampered = tampered
    result.skips_added = len(guard.added_skips(diff))
    frozen_after = env.freeze()
    result.packages_changed = sorted(k for k in set(frozen_before) | set(frozen_after)
                                     if frozen_before.get(k) != frozen_after.get(k))

    after = validator.run(env.python)
    (out_dir / "tests_after.txt").write_text(after.output[-20000:])
    result.failing_after = sorted(after.failing)
    ok, new_failures, drifted, _ = validator.judge(after)
    result.new_failures = new_failures + drifted
    shape_after = guard.code_shape(workspace, protected_rel)
    result.lost_symbols = guard.lost_symbols(shape_before, shape_after)
    result.added_symbols = guard.added_symbols(shape_before, shape_after)
    result.fixed_baseline = sorted(set(known_failing) - set(after.failing))
    result.scope_creep = bool(result.fixed_baseline and result.added_symbols)

    if run.exit_status == "crashed" and not result.files_changed:
        result.reason = "agent_error"          # infrastructure, not a failed repair
    elif tampered:
        result.reason = "edited_protected_files"
    elif result.skips_added:
        result.reason = "added_skips"
    elif result.packages_changed:
        result.reason = "changed_packages"
    elif result.lost_symbols:
        result.reason = "removed_code"         # tests pass, but code went missing
    elif not ok:
        result.reason = "tests_still_fail"
    else:
        result.accepted, result.reason = True, "repaired"

    if result.accepted:
        git(workspace, "add", "-A", "--", ".", *[f":(exclude){p}" for p in protected_rel])
        git(workspace, "commit", "-qm", f"STRIDE: code change for {label}")
    else:
        guard.restore(workspace, tampered)
        git(workspace, "reset", "-q", "--hard", "HEAD~1")   # drop code and the pin commit
        git(workspace, "clean", "-fdq", "-e", ".aura*", "-e", ".stride*")
    env.destroy()
    shutil.rmtree(shims, ignore_errors=True)
    result.seconds = round(time.time() - started, 1)
    return _save(result, out_dir)


MODULE_TO_DIST = r"""
import importlib.metadata as md, json
out = {}
for d in md.distributions():
    name = d.metadata["Name"]
    tops = set()
    txt = d.read_text("top_level.txt")
    if txt:
        tops.update(t.strip() for t in txt.split() if t.strip())
    for f in d.files or []:
        part = str(f).split("/")[0]
        if part.endswith(".py"):
            part = part[:-3]
        if part and not part.endswith((".dist-info", ".egg-info", ".data")) and part != "..":
            tops.add(part)
    for t in tops:
        out.setdefault(t, name)
print(json.dumps(out))
"""


def _module_to_dist(python: str) -> dict[str, str]:
    """Import name -> distribution name in the subject's environment
    (pl_bolts -> lightning-bolts, PIL -> pillow)."""
    try:
        out = subprocess.run([python, "-c", MODULE_TO_DIST], capture_output=True,
                             text=True, timeout=120)
        return json.loads(out.stdout or "{}")
    except Exception:
        return {}


def _norm(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _try_transitive_pins(env, implicated: list[str], target: ReqFile, validator: Validator,
                         index, python_version: str, out_dir: Path,
                         stepped: set[str] | None = None) -> tuple[dict[str, str], bool]:
    """Move a package the failure was raised in to another release.

    Any installed package except the ones being stepped: an undeclared
    transitive package (anyio under fastapi) or a declared one that has to
    catch up (lightning-bolts 0.5 still imports torch._six, removed in torch
    2.0). Every other pin is held, so pip cannot quietly move the step back.
    Returns (pins, passed): the pins that made the suite pass; else the move that
    left the fewest new failures, installed, for the agent to finish; else {}.
    """
    stepped = {_norm(n) for n in (stepped or set())}
    frozen = {_norm(k): v for k, v in env.freeze().items()}
    modules = {k: _norm(v) for k, v in _module_to_dist(env.python).items()}
    hold = {_norm(k): v for k, v in target.pins().items()}
    log = []
    seen = set()
    best: tuple[int, dict[str, str]] | None = None     # fewest new failures so far
    start_new = len(validator.judge(validator.run(env.python))[1])
    for name in implicated:
        dist = modules.get(name) or _norm(name)
        if dist in seen or dist in stepped or dist not in frozen:
            continue
        seen.add(dist)
        current = frozen[dist]
        held = [f"{k}=={v}" for k, v in hold.items() if k != dist]
        candidates = triage.candidate_pins(index, dist, current, python_version,
                                           as_of=MEASURED_AS_OF)
        if dist in hold:   # a declared package only moves forward: this is an upgrade
            candidates = [v for v in candidates if _newer(v, current)]
        for version in candidates:
            moved = _install_moving(env, dist, version, held, hold, stepped, log)
            if moved is None:
                continue
            ok, new, _, _ = validator.judge(validator.run(env.python))
            log[-1].update({"accepted": ok, "new_failures": new[:5]})
            if ok:
                (out_dir / "transitive_search.json").write_text(json.dumps(log, indent=2))
                return moved, True
            if len(new) < start_new and (best is None or len(new) < best[0]):
                best = (len(new), moved)
            _restore(env, hold, frozen, dist, current)
        _restore(env, hold, frozen, dist, current)
    (out_dir / "transitive_search.json").write_text(json.dumps(log, indent=2))
    if best:
        # Fewer failures, not none: keep the move and let the agent fix the rest.
        keep = dict(hold)
        keep.update(best[1])
        env.pip_install([f"{k}=={v}" for k, v in keep.items()])
        return best[1], False
    return {}, False


def _install_moving(env, dist, version, held, hold, stepped, log):
    """Install ``dist==version`` holding every pin; if that conflicts, hold only
    the stepped packages and let pip move what must move with it
    (lightning-bolts 0.6 needs pytorch-lightning 1.7). Returns all pins that
    changed, or None when it cannot install."""
    out, code = env.pip_install([f"{dist}=={version}", *held])
    entry = {"package": dist, "version": version, "install": code == 0, "held": "all"}
    if code != 0:   # partners may move, within their own major series
        same_major = [f"{k}=={v}" if k in stepped else _within_major(k, v)
                      for k, v in hold.items() if k != dist]
        out, code = env.pip_install([f"{dist}=={version}", *same_major])
        entry.update({"install": code == 0, "held": "same major"})
    if code != 0:
        loose = [f"{k}=={v}" for k, v in hold.items() if k in stepped and k != dist]
        out, code = env.pip_install([f"{dist}=={version}", *loose])
        entry.update({"install": code == 0, "held": "stepped only"})
    log.append(entry)
    if code != 0:
        entry["error"] = out[-400:]
        return None
    now = {_norm(k): v for k, v in env.freeze().items()}
    moved = {dist: version}
    moved.update({k: now[k] for k, v in hold.items() if k in now and now[k] != v})
    entry["moved"] = moved
    backward = [k for k, v in moved.items() if k in hold and not _newer(v, hold[k])
                and k != dist]
    if backward:   # pip downgraded a declared package to make room: not an upgrade
        entry.update({"install": False, "error": f"would downgrade {backward}"})
        return None
    return moved


def _freeze_all(env) -> dict[str, str]:
    """pip freeze including pip, setuptools and wheel."""
    import subprocess
    from aura.reqfile import parse_freeze
    out = subprocess.run([env.python, "-m", "pip", "freeze", "--all"],
                         capture_output=True, text=True, timeout=300)
    return parse_freeze(out.stdout) if out.returncode == 0 else env.freeze()


def _newer(a: str, b: str) -> bool:
    from aura.reqfile import safe_parse_version
    pa, pb = safe_parse_version(a), safe_parse_version(b)
    return pa is not None and pb is not None and pa > pb


def _within_major(name: str, version: str) -> str:
    """``name>=version,<next major`` (for 0.x, the next minor is the break)."""
    from aura.reqfile import safe_parse_version
    p = safe_parse_version(version)
    if p is None or not p.release:
        return f"{name}=={version}"
    r = p.release
    upper = f"0.{(r[1] if len(r) > 1 else 0) + 1}" if r[0] == 0 else f"{r[0] + 1}"
    return f"{name}>={version},<{upper}"


def _restore(env, hold, frozen, dist, current):
    """Back to the step's environment after a candidate that did not help."""
    pins = dict(hold)
    pins[dist] = current
    env.pip_install([f"{k}=={v}" for k, v in pins.items()])


def _save(result: RepairResult, out_dir: Path) -> RepairResult:
    (out_dir / "repair.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result
