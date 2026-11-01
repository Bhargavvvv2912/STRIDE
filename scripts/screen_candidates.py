#!/usr/bin/env python
"""Screen candidate subjects before committing hours of compute to them.

For each repository this does the two cheap things that decide whether a
subject is worth a full campaign:

  baseline   install the pinned requirements and run the project's own tests.
             If this fails, the subject has no working baseline (criterion C2).
  upgrade    install the same requirements with every version constraint
             removed and run the tests again.

and classifies the result:

  reject     no working baseline: nothing to measure against
  weak       the baseline passes and so does the naive upgrade -- a subject
             where no search is needed, kept because excluding these would
             select the cases where the baselines fail
  keep       the baseline passes and the naive upgrade breaks -- the subjects
             where a search has something to find

Usage
    python scripts/screen_candidates.py owner/repo [owner/repo ...]
    python scripts/screen_candidates.py --repos candidates.txt
    python scripts/screen_candidates.py --pythons 3.10,3.9 owner/repo

Results are appended to screening/screening.jsonl (one line per repository),
so a long screen can be interrupted and resumed; already-screened repositories
are skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aura.environment import find_interpreter  # noqa: E402
from aura.reqfile import ReqFile  # noqa: E402

# Below this, a passing suite is not evidence: one green test cannot
# tell an upgrade that broke the project from one that did not.
MIN_PASSING_TESTS = 3

TEST_REQ_PATTERNS = (
    "requirements_test*.txt", "requirements-test*.txt", "test-requirements*.txt",
    "test_requirements*.txt", "requirements_dev*.txt", "requirements-dev*.txt",
    "dev-requirements*.txt", "requirements/test*.txt", "requirements/dev*.txt",
)

# Tooling that runs *around* a test suite rather than inside it. Old pins of
# these routinely fail to build on a modern interpreter while contributing
# nothing to whether the tests pass, so they are left out of the screen.
RUNNER_ONLY = {
    "nose", "nose2", "coverage", "pytest-cov", "codecov", "coveralls", "tox",
    "flake8", "pylint", "black", "isort", "mypy", "pre-commit", "twine",
    "wheel", "setuptools", "pip", "bumpversion", "ipython", "jupyter",
}


def sh(cmd, cwd=None, timeout=900):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        # A missing interpreter is a fact about this candidate's environment,
        # not a reason to abandon the whole screening run.
        return 127, f"could not run {cmd[0]}: {exc}"


# pip's own summary lines, in the order they are worth reading. "error:
# subprocess-exited-with-error" is pip saying a build failed and nothing more,
# so it is the last resort rather than the first match -- a screening record
# that says only that cannot be acted on.
_PIP_CAUSES = (
    r"No matching distribution found for \S+",
    r"Could not find a version that satisfies[^\n]+",
    r"ERROR: Cannot install[^\n]+",
    r"ResolutionImpossible[^\n]*",
    r"Requires-Python[^\n]+",
    r"error: (?:command|Microsoft|legacy-install)[^\n]+",
    r"\w*Error: [^\n]+",
    r"error: [^\n]+",
)


def install_error(output: str) -> str:
    """Why pip failed, in one line a reader can act on."""
    for pattern in _PIP_CAUSES:
        found = re.findall(pattern, output)
        if found:
            return found[-1].strip()[:300]
    tail = [l.strip() for l in output.splitlines() if l.strip()]
    return " | ".join(tail[-3:])[:300]


def _has_test_functions(path: Path) -> bool:
    try:
        return bool(re.search(r"^\s*(?:async\s+)?def test_", path.read_text(errors="replace"), re.M))
    except OSError:
        return False


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


def find_test_targets(repo: Path) -> list[str]:
    """What to hand pytest, named the way this project happens to name it.

    A test directory is the usual answer, but only when pytest will collect
    something from it: projects exist whose tests live in ``tests/`` under names
    like ``project_creator.py``, which pytest ignores when given the directory
    and collects happily when given the file. Falling back to naming the files
    keeps those projects as candidates instead of recording "no tests ran" for
    a suite that runs perfectly well.
    """
    for name in ("tests", "test"):
        directory = repo / name
        if not directory.is_dir():
            continue
        collectible = [p for p in directory.rglob("*.py")
                       if p.name.startswith("test_") or p.name.endswith("_test.py")]
        if collectible:
            return [name]
        named = sorted(p for p in directory.rglob("*.py")
                       if _has_test_functions(p))
        if named:
            return [str(p.relative_to(repo)) for p in named]
    found = [p for p in repo.rglob("*.py")
             if ".git" not in p.parts
             and (p.name.startswith("test_") or p.name.endswith("_test.py")
                  or p.name == "tests.py")]
    return sorted(str(p.relative_to(repo)) for p in found)


def count_tests(repo: Path, targets: list[str]) -> int:
    n = 0
    for t in targets:
        path = repo / t
        files = [path] if path.is_file() else list(path.rglob("*.py"))
        for f in files:
            try:
                n += len(re.findall(r"^\s*(?:async\s+)?def test_",
                                    f.read_text(errors="replace"), re.M))
            except OSError:
                pass
    return n


def requirement_lines(repo: Path) -> tuple[list[str], list[str], list[str]]:
    """(app requirement lines, test requirement lines, test files used)."""
    app = ReqFile.load(repo / "requirements.txt")
    app_lines = [l.raw.split("#")[0].strip() for l in app.requirements()]
    test_lines, used = [], []
    for pattern in TEST_REQ_PATTERNS:
        for f in sorted(repo.glob(pattern)):
            used.append(str(f.relative_to(repo)))
            for l in ReqFile.load(f).requirements():
                if l.name and l.name not in RUNNER_ONLY:
                    test_lines.append(l.raw.split("#")[0].strip())
    return [l for l in app_lines if l], test_lines, used


def unpin(lines: list[str]) -> list[str]:
    rf = ReqFile.from_text("\n".join(lines) + "\n")
    rf.drop_all_constraints()
    return [l.render().split("#")[0].strip() for l in rf.requirements()]


def trial(python: str, lines: list[str], repo: Path, targets: list[str],
          work: Path) -> dict:
    env = work / "env"
    shutil.rmtree(env, ignore_errors=True)
    code, out = sh([python, "-m", "venv", str(env)])
    py = str(env / "bin" / "python")
    if code or not Path(py).exists():
        # venv has been seen to exit 0 and still leave no interpreter (a full
        # disk, a broken ensurepip). Check for the file rather than trust the
        # exit code, or the next command dies with a bare FileNotFoundError.
        return {"stage": "venv", "ok": False,
                "detail": (out[-400:] or f"{py} was not created")}
    sh([py, "-m", "pip", "install", "-q", "--upgrade", "pip"])

    req = work / "reqs.txt"
    body = list(lines)
    if not any(re.match(r"^pytest\b", l, re.I) for l in body):
        body.append("pytest")          # the runner, not part of the subject
    req.write_text("\n".join(body) + "\n")
    started = time.time()
    code, out = sh([py, "-m", "pip", "install", "-q", "-r", str(req)],
                   timeout=1800)
    if code:
        return {"stage": "install", "ok": False, "detail": install_error(out),
                "seconds": round(time.time() - started, 1)}

    code, out = sh([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets],
                   cwd=repo, timeout=900)
    tail = summary_line(out)
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
    first_error = next((l.strip() for l in out.splitlines()
                        if re.match(r"^E\s", l)), "")
    # A baseline does not have to be green, only runnable: the judge rejects a
    # candidate for tests that *newly* fail, so a suite with pre-existing
    # failures still constrains an upgrade through the tests that pass. What
    # cannot be used is a suite that never ran.
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
    return {"stage": "tests", "ok": passed >= MIN_PASSING_TESTS,
            "failing": failed,
            "summary": tail[:160], "passed": passed,
            "detail": first_error[:300],
            "seconds": round(time.time() - started, 1)}


def screen(slug: str, pythons: list[str], work_root: Path) -> dict:
    name = slug.replace("/", "_")
    repo = work_root / "repos" / name
    record = {"repo": slug, "screened": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                        time.gmtime())}
    if not repo.exists():
        code, out = sh(["git", "clone", "-q", "--depth", "1",
                        f"https://github.com/{slug}", str(repo)], timeout=600)
        if code:
            return {**record, "verdict": "reject", "reason": "clone failed"}
    _, head = sh(["git", "rev-parse", "HEAD"], cwd=repo)
    record["commit"] = head.strip()

    if not (repo / "requirements.txt").exists():
        return {**record, "verdict": "reject", "reason": "no requirements.txt"}
    targets = find_test_targets(repo)
    if not targets:
        return {**record, "verdict": "reject", "reason": "no tests found"}
    record.update(test_targets=targets, test_functions=count_tests(repo, targets),
                  needs_services=any((repo / f).exists() for f in
                                     ("docker-compose.yml", "docker-compose.yaml")))

    app, test, test_files = requirement_lines(repo)
    record.update(requirements=app, test_requirement_files=test_files,
                  test_requirements=test)

    work = work_root / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    for version in pythons:
        python = find_interpreter(version)
        if python is None:
            continue
        base = trial(python, app + test, repo, targets, work)
        if base["ok"]:
            record.update(python_version=version, baseline=base)
            break
        record.setdefault("baseline_attempts", {})[version] = base
    else:
        return {**record, "verdict": "reject",
                "reason": "no working baseline on " + ", ".join(pythons)}

    upgraded = trial(find_interpreter(record["python_version"]),
                     unpin(app + test), repo, targets, work)
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
                             f"{upgraded['detail'][:120]}")
    shutil.rmtree(work / "env", ignore_errors=True)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repos", nargs="*", help="owner/name")
    ap.add_argument("--repos", dest="repo_file", help="file with one owner/name per line")
    ap.add_argument("--pythons", default="3.13,3.12,3.11,3.10,3.9,3.8",
                    help="interpreters to try for the baseline, in order")
    ap.add_argument("--out", default="screening/screening.jsonl")
    ap.add_argument("--force", action="store_true", help="re-screen repos already done")
    args = ap.parse_args()

    slugs = list(args.repos)
    if args.repo_file:
        slugs += [l.strip() for l in Path(args.repo_file).read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
    if not slugs:
        ap.error("give at least one owner/name")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and not args.force:
        done = {json.loads(l)["repo"] for l in out.read_text().splitlines() if l.strip()}

    pythons = [p.strip() for p in args.pythons.split(",") if p.strip()]
    marks = {"keep": "KEEP  ", "weak": "weak  ", "reject": "reject"}
    for i, slug in enumerate(slugs, 1):
        if slug in done:
            print(f"[{i}/{len(slugs)}] {slug}: already screened (use --force)")
            continue
        print(f"[{i}/{len(slugs)}] {slug} ...", flush=True)
        started = time.time()
        try:
            rec = screen(slug, pythons, out.parent)
        except Exception as exc:
            # One unscreenable repository must not cost the whole batch: the
            # run is long, and the rest of the list is still worth screening.
            rec = {"repo": slug, "verdict": "reject",
                   "reason": f"screening crashed: {type(exc).__name__}: {exc}",
                   "screened": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        rec["screen_seconds"] = round(time.time() - started, 1)
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        extra = ""
        if rec.get("python_version"):
            extra = (f" py{rec['python_version']}, {rec['baseline'].get('passed', 0)} "
                     f"tests pass at baseline")
        print(f"    {marks[rec['verdict']]} {rec['reason'][:110]}{extra}")

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    keep = [r for r in rows if r["verdict"] == "keep"]
    print(f"\n{len(rows)} screened: {len(keep)} keep, "
          f"{sum(r['verdict'] == 'weak' for r in rows)} weak, "
          f"{sum(r['verdict'] == 'reject' for r in rows)} reject  ->  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
