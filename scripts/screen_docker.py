#!/usr/bin/env python
"""Screen candidate subjects on this machine, in the platform the cluster uses.

Screening decides a candidate by installing its pinned requirements and running
its tests, so it has to happen where those installs behave the way they will in
the experiment. On an Apple Silicon laptop they do not: the old pins this work
is about ship manylinux x86_64 wheels and no macOS arm64 wheels, so pip compiles
numpy 1.20 from source, the compile fails, and a perfectly good subject is
recorded as having no working baseline.

This runs each trial in a linux/amd64 container -- the cluster's platform -- so
a verdict produced here means the same thing as one produced there, without
waiting for a Slurm queue.

Needs Docker running (Docker Desktop on a Mac). Nothing is pushed anywhere; the
containers are removed as they finish.

Usage:
    python scripts/screen_docker.py AlexandrovRoman/Flask-DJ
    python scripts/screen_docker.py --repos screening/to_screen.txt
    python scripts/screen_docker.py --pythons 3.10,3.9 owner/repo

Verdicts are appended to screening/screening.jsonl, the same file the cluster
job writes, each tagged with the platform it was produced on.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# The full image, not -slim: slim has no C compiler, so any package without a
# wheel fails with "command 'gcc' failed" -- a property of the image rather
# than of the project, and the cluster the runs happen on has a compiler.
#
# Pinned to bookworm (glibc 2.36) rather than the floating tag, which now means
# trixie (glibc 2.41). glibc 2.41 refuses to load a shared object that asks for
# an executable stack, which every torch wheel of a certain age does: screening
# on trixie rejected GabrielDornelles/pytorch-ocr for an ImportError the cluster
# would never raise. Screening must not be stricter than the machine the runs
# happen on, or it discards subjects for the screener's properties.
IMAGE = "python:{version}-bookworm"
# Below this, a passing suite is not evidence: one green test cannot
# tell an upgrade that broke the project from one that did not.
MIN_PASSING_TESTS = 3

INSTALL_FAILED = "AURA-INSTALL-FAILED"
# A project that declares itself a package but does not build as one.
SELF_INSTALL_FAILED = "AURA-SELF-INSTALL-FAILED"

# The screening logic (what to install, what to run, how to classify) lives in
# screen_candidates.py and is shared: only *where* the trial runs differs.
_spec = importlib.util.spec_from_file_location(
    "screen_candidates", ROOT / "scripts" / "screen_candidates.py")
screen_candidates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(screen_candidates)

from aura.baseline import (                              # noqa: E402
    LADDER, TEST_TOOLING, closure_at_date, era_build_tools, reconstruct,
)
from aura.environment import ERA_PIP, _BUILD_TOOL_ERROR  # noqa: E402
from aura.index import PackageIndex               # noqa: E402
from aura.reqfile import ReqFile                  # noqa: E402

count_tests = screen_candidates.count_tests
find_test_targets = screen_candidates.find_test_targets
install_error = screen_candidates.install_error
requirement_lines = screen_candidates.requirement_lines
unpin = screen_candidates.unpin


def docker() -> str:
    for candidate in ("docker", str(Path.home() / ".docker/bin/docker"),
                      "/Applications/Docker.app/Contents/Resources/bin/docker"):
        if shutil.which(candidate) or Path(candidate).exists():
            return candidate
    sys.exit("Docker not found: install Docker Desktop and start it")


DOCKER = None
# Placeholder environment variables for the project's tests, from --env. The
# same values go into the subject's registry entry, so the screen and the run
# see the same environment -- see Subject.environment.
ENVIRONMENT: dict[str, str] = {}


# One pip cache for every container: without it each trial re-downloads the
# whole environment, and a repository is tried on up to three interpreters.
CACHE = Path.home() / ".aura-screen-cache"


def run_in_container(version: str, repo: Path, reqs: Path, script: str,
                     timeout: int) -> tuple[int, str]:
    """Run a shell script against this repo inside a linux/amd64 container."""
    CACHE.mkdir(exist_ok=True)
    command = [
        DOCKER, "run", "--rm", "--platform", "linux/amd64",
        "-v", f"{repo.resolve()}:/repo",
        "-v", f"{reqs.resolve()}:/reqs.txt:ro",
        "-v", f"{CACHE.resolve()}:/root/.cache/pip",
        *[a for k, v in ENVIRONMENT.items() for a in ("-e", f"{k}={v}")],
        "-w", "/repo", IMAGE.format(version=version),
        "bash", "-lc", script,
    ]
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timed out"


def install_script(era: tuple[str, ...] = ()) -> str:
    """The install half of a trial, with the retries the harness itself performs.

    Screening has to agree with the run, or a subject screens in and then fails
    for a reason already known -- or, as happened to colindecourt/record, is
    rejected here for something the run would have recovered from.
    """
    base = ("pip install -q --upgrade pip >/dev/null 2>&1; "
            "pip install -r /reqs.txt")
    era_tools = ""
    if era:
        # aura.environment._retry_with_era_build_tools.
        era_tools = (f"pip install -q {' '.join(era)} >/dev/null 2>&1; "
                     "pip install --no-build-isolation -r /reqs.txt && exit 0; ")
    # aura.environment._retry_with_era_pip: pip 24.1 skips any release whose
    # metadata is not valid PEP 440, and then reports it as missing.
    era_pip = (f"pip install -q {ERA_PIP} >/dev/null 2>&1; "
               "pip install -r /reqs.txt")
    return f"{base} && exit 0; {era_tools}{era_pip}"


# Newest first, because a project that runs on a current interpreter should be
# measured on one; the older versions are for pins that predate them. 3.7 is
# past end of life but included: torch 1.4 published Linux wheels only up to
# 3.7, and 2015-2019 projects refuse 3.8 outright or depend on releases that
# predate it -- in one batch it was the direct cause of five rejections.
DEFAULT_PYTHONS = "3.13,3.12,3.11,3.10,3.9,3.8,3.7"


def declared_python(repo: Path) -> str:
    """The interpreter the project itself asks for, if it says.

    A project that states 3.11 is not evidence about 3.9, and trying the
    others first spends an install per version to rediscover what the
    repository already told us.
    """
    for name in (".python-version", "runtime.txt"):
        path = repo / name
        if path.exists():
            found = re.search(r"(\d+\.\d+)", path.read_text())
            if found:
                return found.group(1)
    for name in ("pyproject.toml", "setup.cfg", "setup.py"):
        path = repo / name
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        found = re.search(r"requires[-_]python\s*=?\s*[\"']([^\"']+)", text)
        if found:
            floor = re.search(r">=\s*(\d+\.\d+)", found.group(1))
            if floor:
                return floor.group(1)
    return ""


def installs_itself(repo: Path) -> bool:
    """Whether the project is a package its own tests would import.

    A bare setup.cfg does not make one: it is as often flake8 or coverage
    configuration, and treating it as packaging metadata makes every install
    fail with "does not appear to be a Python project".

    Nor does a bare pyproject.toml. pytorch-ocr's holds ``[tool.black]`` and
    nothing else -- a formatter's line length, declaring no package and naming
    no build backend. The file has become where Python tools keep their
    settings, so its presence says only that the project uses a tool.
    """
    if (repo / "setup.py").exists():
        return True
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(errors="replace")
        # Either a package is declared, or a backend is named to build one.
        if re.search(r"^\s*\[(project|build-system|tool\.(poetry|flit|pdm|hatch))",
                     text, re.M):
            return True
    cfg = repo / "setup.cfg"
    return cfg.exists() and "[metadata]" in cfg.read_text(errors="replace")


# pytest ends with a line like "3 passed, 1 failed in 0.4s". It is not the last
# line of the output: stderr is appended after stdout and pip's warnings land
# there, so taking the last line reads a pip notice and every suite parses as
# zero tests passed.
_PYTEST_SUMMARY = re.compile(
    r"^=*\s*(?:\d+ (?:passed|failed|error|errors|skipped|xfailed|xpassed)\b.*"
    r"|no tests ran.*)$", re.M)


def summary_line(output: str) -> str:
    """pytest's own summary line, or "" when the suite never produced one."""
    found = [m.strip(" =") for m in _PYTEST_SUMMARY.findall(output or "")]
    if found:
        return found[-1]
    counts = [l.strip() for l in (output or "").splitlines()
              if re.search(r"\d+ (passed|failed|error)", l)]
    return counts[-1] if counts else ""


def trial(version: str, lines: list[str], repo: Path, targets: list[str],
          work: Path, era: tuple[str, ...] = ()) -> dict:
    """Install this requirement set and run the project's tests, in a container."""
    body = list(lines)
    if not any(l.lower().startswith("pytest") for l in body):
        # The runner is not part of the subject, but the harness installs only
        # what the project declares: a subject that does not name pytest screens
        # in here and then fails every arm with "No module named pytest". The
        # verdict records it so the registry entry can declare the runner.
        body.append("pytest")
    reqs = work / "reqs.txt"
    reqs.write_text("\n".join(body) + "\n")

    started = time.time()
    # Install and test in one container. Doing them in two means installing
    # the whole environment twice, and these installs are the entire cost of
    # screening. The marker keeps the two verdicts apart: an install that fails
    # is a different finding from a test suite that fails, and the screen
    # exists to tell them apart.
    # A project that is itself a package has tests that import it, so it has to
    # be installed like one -- without touching the dependency set being
    # measured, which is what --no-deps guarantees. A project that is not a
    # package simply fails this harmlessly.
    # Whether it installs is answered by installing it, not by which files are
    # present: the runner treats a failed project build as fatal for every arm,
    # so a flag that is optimistic here voids the whole subject there. The
    # failure is tolerated for the rest of the trial -- the tests may well not
    # need it -- but it is recorded rather than swallowed.
    declares_package = installs_itself(repo)
    editable = (f"pip install -q -e . --no-deps >/dev/null 2>&1 "
                f"|| echo '{SELF_INSTALL_FAILED}'; " if declares_package else "")
    code, out = run_in_container(
        version, repo, reqs,
        f"({install_script(era)}) || {{ echo '{INSTALL_FAILED}'; exit 1; }}; "
        + editable +
        f"python -m pytest -q -p no:cacheprovider -p no:typeguard {' '.join(targets)}",
        timeout=3600)
    self_installs = declares_package and SELF_INSTALL_FAILED not in out
    seconds = round(time.time() - started, 1)
    if INSTALL_FAILED in out:
        return {"stage": "install", "ok": False, "detail": install_error(out),
                "seconds": seconds, "self_installs": self_installs,
                "build_tools_retried": bool(era and _BUILD_TOOL_ERROR.search(out))}

    import re
    tail = summary_line(out)
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
    first_error = next((l.strip() for l in out.splitlines()
                        if re.match(r"^E\s", l)), "")
    if not first_error and not tail:
        # pytest died before it could report: a crash at import or collection
        # prints a traceback and no "E " line, which left every such rejection
        # with a blank reason. The traceback's last line is the error itself.
        first_error = next((l.strip() for l in reversed(out.splitlines())
                            if re.match(r"^\s*\w*(Error|Exception)\b", l.strip())
                            or re.match(r"^\w+(\.\w+)*: ", l.strip())), "")
    # A baseline does not have to be green, only runnable: the judge rejects a
    # candidate for tests that *newly* fail, so a suite with pre-existing
    # failures still constrains an upgrade through the tests that pass. What
    # cannot be used is a suite that never ran.
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
    # Where the first failure was raised: the frames pytest prints before its
    # first "E" line. A break inside site-packages is one library tripping
    # over another; a break in the project's files is the project's own code.
    frames = []
    for line in out.splitlines():
        if re.match(r"^E\s", line):
            if frames:
                break
            continue
        if re.match(r"^\S+\.py:\d+: in ", line.strip()) or re.match(r'^\s*File "[^"]+", line \d+', line):
            frames.append(line.strip()[:200])
    return {"stage": "tests", "ok": passed >= MIN_PASSING_TESTS,
            "failing": failed, "self_installs": self_installs,
            "summary": tail[:160], "passed": passed, "detail": first_error[:300],
            "frames": frames[-6:], "seconds": seconds}


def closure_in_container(version: str, repo: Path, reqs: Path) -> dict[str, str]:
    """What pip would install from this requirement set, names and versions."""
    code, out = run_in_container(
        version, repo, reqs,
        "pip install -q --dry-run --report /tmp/report.json -r /reqs.txt "
        ">/dev/null 2>&1; cat /tmp/report.json 2>/dev/null", timeout=1200)
    if code != 0 or not out.strip().startswith("{"):
        return {}
    try:
        payload = json.loads(out[out.index("{"):])
    except (ValueError, json.JSONDecodeError):
        return {}
    closure = {}
    for item in payload.get("install", []):
        meta = item.get("metadata") or {}
        if meta.get("name") and meta.get("version"):
            closure[meta["name"].lower().replace("_", "-")] = meta["version"]
    return closure


def repaired_lines(version: str, repo: Path, lines: list[str], work: Path,
                   on_date: str, index) -> tuple[list[str], dict]:
    """The same requirements with the closure dated back to ``on_date``.

    The screen has to answer the question the experiment will actually face: a
    project whose file pins some dependencies and not others does not install
    today, and the run will rebuild it this same way, so screening it on the
    declared file alone rejects subjects the experiment could use.
    """
    reqs = work / "closure_reqs.txt"
    reqs.write_text("\n".join(lines) + "\n")
    declared = ReqFile.from_text("\n".join(lines) + "\n")
    closure = closure_in_container(version, repo, reqs)
    if not closure:
        # pip cannot report a closure for a set it cannot resolve -- which is
        # the state of a project whose unpinned dependency has moved somewhere
        # incompatible with a pinned one. Declared metadata still describes the
        # environment, so walk that instead, exactly as the run does.
        closure = closure_at_date(index, declared.pins(), on_date,
                                  python_version=version)
    if not closure:
        return []
    # Both rungs the run climbs, in its order: the closure dated back, then the
    # same with the project's own test tooling unpinned. Screening used to stop
    # after the first and to skip even that when nothing had moved, which made
    # it stricter than the experiment -- it rejected subjects the run would
    # have established a baseline for.
    out = []
    for stage, skip, wheels in LADDER:
        fixed, moved = reconstruct(declared, closure, index=index, on_date=on_date,
                                   python_version=version, skip=skip,
                                   prefer_wheels=wheels)
        if stage.startswith("relaxed"):
            fixed = fixed.without_pins([n for n in fixed.pins() if n in TEST_TOOLING])
        if fixed.to_text() == declared.to_text():
            continue
        lines = [l for l in fixed.to_text().splitlines() if l.strip()]
        # A wheel rung that moved nothing is an earlier rung again.
        if wheels and any(lines == earlier for _, earlier, _ in out):
            continue
        out.append((stage, lines, moved))
    return out


def screen(slug: str, pythons: list[str], work_root: Path) -> dict:
    name = slug.replace("/", "_")
    repo = work_root / "repos" / name
    record = {"repo": slug, "platform": "linux/amd64 (docker)",
              "screened": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not repo.exists():
        clone = subprocess.run(["git", "clone", "-q", "--depth", "1",
                                f"https://github.com/{slug}", str(repo)],
                               capture_output=True, text=True, timeout=600)
        if clone.returncode:
            return {**record, "verdict": "reject", "reason": "clone failed"}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True)
    record["commit"] = head.stdout.strip()

    if not (repo / "requirements.txt").exists():
        return {**record, "verdict": "reject", "reason": "no requirements.txt"}
    targets = find_test_targets(repo)
    if not targets:
        return {**record, "verdict": "reject", "reason": "no tests found"}
    app, test, test_files = requirement_lines(repo)
    record.update(requirements=app, test_requirement_files=test_files,
                  test_requirements=test)
    record.update(test_targets=targets, test_functions=count_tests(repo, targets),
                  is_installable=installs_itself(repo),
                  declares_test_runner=any(l.lower().startswith("pytest")
                                           for l in app + test))

    date = subprocess.run(["git", "log", "-1", "--format=%cd", "--date=short"],
                          cwd=repo, capture_output=True, text=True)
    record["commit_date"] = date.stdout.strip()
    index = PackageIndex()
    # A pin to a release PyPI does not have can never install, on any
    # interpreter and at any rung -- the ladder does not rewrite the project's
    # own pins. Found here from the index in a second, instead of after
    # minutes of containers: install==1.3.5, a deleted package, cost two
    # candidates a full screen each.
    missing = []
    for line in app + test:
        m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._\-]*)(?:\[[^\]]*\])?\s*==\s*([^\s;#,]+)\s*$",
                     line)
        if not m:
            continue
        try:
            history = index.history(m.group(1))
        except Exception as exc:
            # A project PyPI has never heard of is reported as an error, not as
            # None -- which is how install==1.3.5 slipped past this check.
            # Only that is a verdict; any other failure is the network.
            if "No details about project" in str(exc):
                missing.append(f"{m.group(1)}=={m.group(2)}")
            continue
        if history is None or not history.contains(m.group(2)):
            missing.append(f"{m.group(1)}=={m.group(2)}")
    if missing:
        return {**record, "verdict": "reject",
                "reason": f"pins a release PyPI does not have: {', '.join(missing[:4])}"}
    # The build tools of this environment's era, from the pins themselves.
    try:
        era = era_build_tools(index, ReqFile.from_text("\n".join(app + test) + "\n").pins(),
                              on_date=record["commit_date"])
    except Exception:
        era = ()
    record["build_era"] = list(era)

    work = work_root / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    # What the project asks for goes first; the rest of the ladder follows.
    asked = declared_python(repo)
    if asked and asked in pythons:
        pythons = [asked] + [v for v in pythons if v != asked]
        record["declared_python"] = asked
    for version in pythons:
        lines, stage, repaired = app + test, "declared", {}
        base = trial(version, lines, repo, targets, work, era)
        if not base["ok"] and record["commit_date"]:
            # The same repair the run itself would perform, rung by rung, so the
            # screen and the experiment agree about which subjects are usable.
            # Every rung is recorded, not only the one kept: when reconstruction
            # fails, what it pinned and where it stopped is the diagnosis, and
            # without it negativei2-server read as "stage: None" and nothing
            # else, which sent the investigation to peeling layers by hand.
            ladder = record.setdefault("ladder", {}).setdefault(version, [])
            rungs = repaired_lines(version, repo, app + test, work,
                                   record["commit_date"], index)
            if not rungs:
                ladder.append({"rung": "none", "detail": "nothing to reconstruct: "
                               "no closure could be established, or dating it back "
                               "changed no pin"})
            for rung, fixed, moved in rungs:
                attempt = trial(version, fixed, repo, targets, work, era)
                ladder.append({"rung": rung, "pins": len(fixed),
                               "dated_back": sorted(moved)[:40],
                               "stage": attempt.get("stage"), "ok": attempt.get("ok"),
                               "summary": (attempt.get("summary") or "")[:160],
                               "detail": (attempt.get("detail") or "")[:300]})
                if attempt["ok"] or attempt["stage"] == "tests":
                    lines, stage, repaired, base = fixed, rung, moved, attempt
                if attempt["ok"]:
                    break
        if stage != "declared":
            record.update(baseline_stage=stage, baseline_repaired=repaired,
                          reconstructed_requirements=lines)
        if base["ok"]:
            record.update(python_version=version, baseline=base)
            break
        if base["stage"] == "tests" and len(targets) > 1:
            # A suite can be part sound and part unrunnable -- a file that
            # shells out to its author's own machine, say. Rejecting the
            # repository loses a usable oracle, so each file is tried on its
            # own and the ones that pass are reported for a human to accept.
            usable, per_file = [], {}
            for target in targets:
                # ``lines``, not the declared set: if the baseline had to be
                # rebuilt, each file has to be judged in that same environment.
                one = trial(version, lines, repo, [target], work, era)
                per_file[target] = {"ok": one["ok"], "passed": one.get("passed", 0),
                                    "summary": one.get("summary", ""),
                                    "detail": one.get("detail", "")}
                if one["ok"]:
                    usable.append(target)
            base["per_file"] = per_file
            if usable:
                # The baseline on record has to be the one the subject will
                # actually be judged against, not the mixed run that included
                # a file we are about to exclude.
                clean = trial(version, lines, repo, usable, work, era)
                clean["per_file"] = per_file
                record.update(python_version=version, baseline=clean,
                              usable_test_targets=usable)
                break
        record.setdefault("baseline_attempts", {})[version] = base
    else:
        detail = ""
        attempts = record.get("baseline_attempts", {})
        if attempts:
            first = next(iter(attempts.values()))
            detail = f": {first.get('detail', '')[:120]}"
        return {**record, "verdict": "reject",
                "reason": f"no working baseline on {', '.join(pythons)}{detail}"}

    # Replace the guess with what the container observed. A project whose
    # pyproject.toml only configures a formatter declares no package, and the
    # runner would fail every arm on it with project_build_failure.
    record["is_installable"] = bool(record["baseline"].get("self_installs"))

    # Whatever the baseline was judged on is what the upgrade is judged on.
    targets = record.get("usable_test_targets", targets)
    upgraded = trial(record["python_version"], unpin(lines), repo, targets, work, era)
    # Judged the way the arms are judged: against the baseline, not in
    # isolation. An upgrade that leaves three tests passing has not "passed" if
    # it broke five that were passing before.
    if upgraded["ok"] and upgraded.get("failing", 0) > (record["baseline"].get("failing") or 0):
        upgraded["ok"] = False
        upgraded["detail"] = (upgraded.get("detail")
                              or f"{upgraded['failing']} failing, "
                                 f"{record['baseline'].get('failing', 0)} at baseline")
    record["upgrade"] = upgraded
    if upgraded["ok"]:
        # Not a rejection. Requiring the naive upgrade to break would select
        # exactly the subjects where the baselines fail, and a comparison run
        # only on those is not evidence about a population of projects. A
        # subject where upgrading everything already works is one where the
        # search is expected to tie, and tying is a result.
        record.update(verdict="weak", headroom=False,
                      reason="the naive upgrade also passes: a subject where "
                             "no search is needed")
    else:
        record.update(verdict="keep", headroom=True,
                      reason=f"naive upgrade breaks at {upgraded['stage']}: "
                             f"{upgraded['detail'][:120] or upgraded.get('summary','')[:120]}")
    return record


def main() -> int:
    global DOCKER
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repos", nargs="*", help="owner/name")
    ap.add_argument("--repos", dest="repo_file", help="file of owner/name lines")
    ap.add_argument("--pythons", default=DEFAULT_PYTHONS)
    ap.add_argument("--out", default="screening/screening.jsonl")
    ap.add_argument("--jobs", type=int, default=3,
                    help="candidates screened at once (default 3)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="a placeholder variable the tests read, e.g. "
                         "AWS_DEFAULT_REGION=us-east-1 (repeatable); recorded "
                         "with the verdict")
    args = ap.parse_args()

    DOCKER = docker()
    if subprocess.run([DOCKER, "info"], capture_output=True).returncode != 0:
        sys.exit("Docker is not running: start Docker Desktop")

    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            ap.error(f"--env wants KEY=VALUE, got {item!r}")
        ENVIRONMENT[key] = value

    slugs = list(args.repos)
    if args.repo_file:
        slugs += [l.strip() for l in Path(args.repo_file).read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
    if not slugs:
        ap.error("give at least one owner/name")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pythons = [p.strip() for p in args.pythons.split(",") if p.strip()]
    marks = {"keep": "KEEP  ", "weak": "weak  ", "reject": "reject"}

    lock = threading.Lock()

    def run_one(i: int, slug: str) -> None:
        with lock:
            print(f"[{i}/{len(slugs)}] {slug} ...", flush=True)
        started = time.time()
        try:
            rec = screen(slug, pythons, out.parent)
            if ENVIRONMENT:
                rec["environment"] = dict(ENVIRONMENT)
        except Exception as exc:
            rec = {"repo": slug, "verdict": "reject",
                   "reason": f"screening crashed: {type(exc).__name__}: {exc}",
                   "platform": "linux/amd64 (docker)"}
        rec["screen_seconds"] = round(time.time() - started, 1)
        extra = ""
        if rec.get("python_version"):
            extra = (f"  py{rec['python_version']}, "
                     f"{rec['baseline'].get('passed', 0)} tests pass at baseline")
            if rec.get("declares_test_runner") is False:
                extra += ("\n           NOTE: does not declare pytest -- the "
                          "subject entry needs environment_setup to install it, "
                          "or every arm fails with 'No module named pytest'")
            if rec.get("usable_test_targets"):
                broken = [t for t in rec.get("test_targets", [])
                          if t not in rec["usable_test_targets"]]
                extra += (f"\n           usable targets: "
                          f"{', '.join(rec['usable_test_targets'])}"
                          f"   (excluded: {', '.join(broken)})")
        with lock:
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            # The repository is named on the verdict line too: with several
            # screened at once, the line above it is not necessarily its own.
            print(f"    {marks[rec['verdict']]} {rec['reason'][:130]}{extra}"
                  f"  [{slug}]", flush=True)

    # Several at once. A screen spends its time waiting on downloads and
    # installs, not on the CPU, so one at a time left eleven of twelve cores
    # idle. Each has its own container and working directory; the pip cache
    # they share is safe for concurrent use.
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        list(pool.map(lambda pair: run_one(*pair), enumerate(slugs, 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
