#!/usr/bin/env python
"""The dependabot arm: real Dependabot, run on GitHub.

For each subject, a private throwaway copy is made on your GitHub account and
GitHub does everything a maintainer's setup would:

    setup     push the subject, at its pinned commit, to a new private repo
              (dependabot-check-<subject>) with Dependabot enabled for pip, CI
              that runs the subject's tests on its Python version, and a step
              that merges a Dependabot PR once those tests pass. When a merge
              makes another PR conflict, Dependabot rebases it and CI runs again.
    status    show where it is: PRs opened, merged, still running CI.
    collect   once it has settled, test the final main branch once more on
              GitHub, then write the result as the dependabot arm's run:
                results/<subject>/<subject>__dependabot.log       readable
                results/<subject>/<subject>__dependabot.jsonl     for the report
                results/<subject>/<subject>__dependabot.requirements.txt
                results/<subject>/<subject>__dependabot.github.json  raw PR data
    delete    remove the throwaway repo.

Nothing is pushed anywhere except the new repo. Needs the GitHub CLI:
    brew install gh && gh auth login && gh auth refresh -s workflow,delete_repo

Usage:
    python scripts/real_dependabot.py setup   mean-review-collector
    python scripts/real_dependabot.py status  mean-review-collector
    python scripts/real_dependabot.py collect mean-review-collector
    python scripts/real_dependabot.py delete  mean-review-collector
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
# How long collect will wait for the CI that its own merges trigger. A wall
# clock, not a number of iterations: see the settling loop in collect().
SETTLE_WAIT_S = 60
# Longer than Dependabot takes to rebase what a merge left behind.
SETTLE_QUIET_S = 900
# Enough for a long merge cascade with a quiet period after each merge.
SETTLE_BUDGET_S = 3 * 3600
sys.path.insert(0, str(ROOT))

# The bot this module drives. scripts/real_renovate.py reuses everything here
# and overrides these, write_bot_config, after_push and _prs.
ARM = "dependabot"
REPO_PREFIX = "dependabot-check"
VISIBILITY = "--private"
# Take the tests already failing from the AURA arms' own baseline when it
# exists, rather than rerunning the suite locally in Docker first.
PREFER_ARM_BASELINE = False

DEPENDABOT_YML = """\
version: 2
updates:
  - package-ecosystem: pip
    directory: "{directory}"
    schedule:
      interval: daily
    # All at once, not the default five: every proposal gets its PR.
    open-pull-requests-limit: 50
{allow}"""

# Tests on every PR and on main, and nothing else. Merging is left to
# collect(), which applies the same rule -- merged if and only if the tests
# passed -- in rounds until nothing changes.
#
# Merging from inside CI raced the scan: the merge pushes to main, and a
# Dependabot job still opening pull requests for a large project is cancelled
# when that happens. find-that-postcode pins 57 packages, 19 of its 23 direct
# dependencies are behind, and it received one pull request before its scan
# was cancelled.
TESTS_YML = """\
name: tests
on:
  pull_request:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: read
jobs:
  tests:
    runs-on: {runner}
{job_env}    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "{python}"
      - run: python -m pip install --upgrade pip
      # The same ladder aura.environment climbs. pip 24.1 skips any release
      # whose metadata is not valid PEP 440 and then reports it as missing, so
      # a baseline that needs an older installer would fail here and every
      # Dependabot pull request with it -- scoring this arm on a repository
      # that never worked rather than on what Dependabot proposed.
      # Then, as the arms do, the build tools of the pins' own era for a
      # source distribution a current setuptools can no longer build.
      - run: |
{install}{editable}      # --deselect: tests that fail before anything is bumped. CI judges a
      # pull request absolutely, the arms judge against the baseline; without
      # this the project's own broken test fails every pull request.
      - run: python -m pytest {target}{deselect}
{workdir}"""


def sh(cmd, cwd=None, check=True, capture=False, input=None):
    return subprocess.run(cmd, cwd=cwd, check=check, text=True,
                          capture_output=capture, input=input)


def subject(name: str) -> dict:
    for s in json.loads((ROOT / "subjects.json").read_text()):
        if s["name"] == name:
            return s
    sys.exit(f"{name} is not in subjects.json")


def gh_user() -> str:
    if shutil.which("gh") is None:
        sys.exit("GitHub CLI not installed: brew install gh && gh auth login")
    out = sh(["gh", "api", "user", "--jq", ".login"], capture=True, check=False)
    if out.returncode != 0:
        sys.exit("GitHub CLI is not logged in: gh auth login")
    return out.stdout.strip()


def repo_name(name: str) -> str:
    return f"{gh_user()}/{REPO_PREFIX}-{name}"


def install_commands(req: str, extra: str, era: tuple[str, ...],
                     setup: list[str]) -> list[str]:
    """The install the arms perform, as shell lines: pip's ladder, then the
    subject's own environment_setup (a test runner, a test-only tool)."""
    plain = f"python -m pip install -r {req}{extra}"
    ladder = f"{plain} || {{ python -m pip install -q pip==24.0 && {plain}; }}"
    if era:
        ladder += (f" || {{ python -m pip install -q {' '.join(era)} && "
                   f"python -m pip install --no-build-isolation -r {req}{extra}; }}")
    return [ladder] + [cmd.replace("{python}", "python") for cmd in setup]


def arm_baseline_failures(name: str) -> list[str]:
    """Tests already failing at the baseline, as the AURA arms recorded it."""
    import ast
    for arm in ("aura_linear_greedy", "aura_full_qwen14b"):
        path = RESULTS_DIR / name / f"{name}__{arm}.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            e = json.loads(line)
            if e.get("event") == "validation" and e.get("phase") == "baseline":
                failing = e.get("failing")
                return sorted(ast.literal_eval(failing) if isinstance(failing, str) else failing or [])
    return []


def baseline_failures(work: Path, python: str, target: str, packaged: bool,
                      install_lines: list[str],
                      environment: dict | None = None) -> list[str]:
    """Tests that already fail before anything is bumped.

    CI judges a pull request absolutely -- one red test and the PR cannot merge
    -- while the arms are judged against the baseline, where a test already
    failing is recorded and tolerated. A project shipping one broken test would
    therefore score zero here for a reason that has nothing to do with
    Dependabot: lung-cancer-detection has a test that writes to a directory
    absent from a fresh checkout, and all ten of its pull requests failed on it.

    Run once, in the platform CI uses, so those tests can be excluded there and
    the two sides ask the same question.
    """
    docker = None
    for candidate in ("docker", str(Path.home() / ".docker/bin/docker"),
                      "/Applications/Docker.app/Contents/Resources/bin/docker"):
        if shutil.which(candidate) or Path(candidate).exists():
            docker = candidate
            break
    if docker is None:
        return []
    # -rEf: failures *and* errors. pytest reports a test that could not run at
    # all under ERROR, and such a test fails a pull request exactly as a failing
    # one does -- nba2k25-plp has three failing and sixty-four erroring, and
    # excluding only the three left every pull request red.
    # The same install CI runs. A plainer one here failed where CI would not
    # (flask-huey-example's MarkupSafe 1.0 needs the era's setuptools) and then
    # reported nothing as already failing, or missed a test tool the subject
    # installs (wgadmin's flask-testing) and deselected a whole test file.
    script = ("( " + " ; ".join(install_lines) + " ) >/dev/null 2>&1; "
              + ("pip install -q -e . --no-deps >/dev/null 2>&1; " if packaged else "")
              + f"python -m pytest -q --tb=no -rEf {target} 2>&1")
    out = subprocess.run(
        [docker, "run", "--rm", "--platform", "linux/amd64",
         "-v", f"{work.resolve()}:/repo", "-w", "/repo",
         *[a for k, v in (environment or {}).items() for a in ("-e", f"{k}={v}")],
         # bookworm, as in screen_docker: the floating tag is now trixie,
         # whose glibc refuses to load an old torch wheel, and a test file
         # that fails to import is recorded here as "already failing" and
         # then skipped in CI -- taking most of the oracle with it.
         f"python:{python}-bookworm", "bash", "-lc", script],
        capture_output=True, text=True, timeout=2400)
    failures = re.findall(r"^(?:FAILED|ERROR) (\S+)",
                          (out.stdout or "") + (out.stderr or ""), re.M)
    return sorted({f.split(" - ")[0] for f in failures})


def baseline_record(name: str) -> Path:
    """Where the file this repository was created with is kept."""
    path = RESULTS_DIR / name / f"{name}__{ARM}.baseline.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def baseline_requirements(slug: str) -> list[str]:
    """The reconstructed baseline for this subject, if it needed one.

    Every arm starts from the same environment. For a project whose own
    requirements no longer install, that environment is the reconstruction at
    its commit date -- so Dependabot has to start there too, or its CI is red
    before any bump lands and the arm reports a zero that says nothing about
    Dependabot.
    """
    log = ROOT / "screening" / "screening.jsonl"
    if not log.exists():
        return []
    latest: dict = {}
    for line in log.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["repo"]] = row
    return list((latest.get(slug) or {}).get("reconstructed_requirements") or [])


def write_bot_config(work: Path, directory: str, req: str, compiled: bool) -> None:
    """The bot's own configuration, written into the new repository."""
    # Scope. A project that compiles its requirements (requirements.in,
    # pyproject.toml) is one Dependabot updates by its *inputs*: it bumps
    # the input and regenerates the lock. Forcing it to consider every
    # locked package instead exceeds its own job budget -- measured at 55
    # minutes on this project's 59 pins, cancelled by GitHub before a
    # second pull request existed. Where there is no input file, the lock
    # is the manifest and transitive pins have to be offered explicitly.
    allow = "" if compiled else (
        "    allow:\n"
        "      # No input file: the pinned list is the manifest, so the\n"
        "      # transitive pins are the project's own decisions too.\n"
        '      - dependency-type: "all"\n')
    (work / ".github" / "dependabot.yml").write_text(
        DEPENDABOT_YML.format(directory=directory, allow=allow))


def after_push(repo: str) -> None:
    """Anything the bot needs on the repository once it exists."""


def setup(name: str, dry_run: bool = False) -> None:
    s = subject(name)
    repo = repo_name(name)
    req = s.get("requirements_file", "requirements.txt")
    directory = "/" + str(Path(req).parent).lstrip(".").strip("/")
    target = (s.get("validation") or {}).get("pytest_target", "")
    python = s.get("python_version", "3.11")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / name
        print(f"cloning {s['url']} at {s.get('commit', 'HEAD')[:10]} ...")
        sh(["git", "clone", "-q", s["url"], str(work)])
        if s.get("commit"):
            sh(["git", "checkout", "-q", s["commit"]], cwd=work)
        sh(["git", "checkout", "-q", "-B", "main"], cwd=work)

        # The project's own CI would run alongside and muddy the picture.
        shutil.rmtree(work / ".github", ignore_errors=True)
        # Dependabot reads these to decide which Python the project needs and
        # refuses anything below 3.10, so a file saying "3.9.2" stops it before
        # it proposes anything. They say nothing about dependencies, and the CI
        # workflow below pins the interpreter the tests actually run on.
        for pinned in (".python-version", "runtime.txt"):
            (work / pinned).unlink(missing_ok=True)
        if req.endswith(".txt") and (work / "Pipfile").exists():
            # A second manifest, like Poetry's tables: Dependabot updates the
            # Pipfile instead of the file the arms measure, and its
            # python_version line ("3.7" for pyconxriddle) makes Dependabot
            # refuse the repository outright -- it supports only 3.10 and up.
            for name in ("Pipfile", "Pipfile.lock"):
                (work / name).unlink(missing_ok=True)
            print(f"  dropped Pipfile and Pipfile.lock: Dependabot now updates {req}")
        if _drop_poetry_dependencies(work):
            print("  dropped the Poetry dependency tables and poetry.lock: "
                  f"Dependabot now updates {req}, the file the arms measure")

        slug = "/".join(s["url"].rstrip("/").removesuffix(".git").split("/")[-2:])
        # The arms' frozen baseline when there is one, so Dependabot starts
        # from exactly their environment; screening's reconstruction otherwise.
        frozen = next((p for p in (RESULTS_DIR / name / f"{name}.baseline.txt",
                                   ROOT / "baselines" / f"{name}.txt") if p.exists()), None)
        rebuilt = ([l for l in frozen.read_text().splitlines() if l.strip()]
                   if frozen else baseline_requirements(slug))
        if rebuilt:
            (work / req).write_text("\n".join(rebuilt) + "\n")
            # Kept so collect() measures against what this repository actually
            # started from. Reading the subject's own commit instead counts a
            # merged bump of a package the declared file never listed as no
            # change at all.
            baseline_record(name).write_text("\n".join(rebuilt) + "\n")
            print(f"using the reconstructed baseline ({len(rebuilt)} pins): the "
                  "project's own requirements no longer install, and every arm "
                  "starts from the same environment")
        else:
            # The file as the arms read it: parsing drops lines that name no
            # installable release (a Debian pip-freeze's pkg-resources==0.0.0),
            # and CI would otherwise fail on every pull request because of one.
            from aura.reqfile import ReqFile
            declared = work / req
            if declared.exists():
                as_read = ReqFile.load(declared).to_text()
                if as_read != declared.read_text():
                    declared.write_text(as_read)
                    print("  dropped lines no release answers to "
                          "(pkg-resources==0.0.0), as the arms do")
        gh_dir = work / ".github"
        (gh_dir / "workflows").mkdir(parents=True)
        compiled = any((work / n).exists() for n in
                       ("requirements.in", "pyproject.toml", "setup.py", "setup.cfg"))
        write_bot_config(work, directory, req, compiled)
        pins = (work / req).read_text()
        needs_pytest = not re.search(r"^\s*pytest\b", pins, re.M | re.I)
        # A project that is itself a package has tests that import it, so CI
        # has to install it like one -- with --no-deps, leaving the dependency
        # set Dependabot is bumping untouched. Without this every PR fails on
        # ModuleNotFoundError and the arm reports a zero that says nothing.
        # Taken from the registry, where screening recorded what it observed
        # when it actually ran the install. Inferring it from a filename says
        # yes to a pyproject.toml that only configures black -- pytorch-ocr's
        # does -- and the editable install then fails in CI, failing every
        # pull request and scoring this arm zero for our own mistake.
        packaged = s.get("is_installable")
        if packaged is None:
            packaged = (work / "setup.py").exists()
            pyproject = work / "pyproject.toml"
            if not packaged and pyproject.exists():
                packaged = bool(re.search(
                    r"^\s*\[(project|build-system|tool\.(poetry|flit|pdm|hatch))",
                    pyproject.read_text(errors="replace"), re.M))
            cfg = work / "setup.cfg"
            if not packaged and cfg.exists():
                packaged = "[metadata]" in cfg.read_text(errors="replace")
        editable = ("      - run: python -m pip install -e . --no-deps\n"
                    if packaged else "")
        print("checking which tests already fail before any bump ...")
        environment = s.get("environment") or {}
        era = ()
        try:
            from aura.baseline import era_build_tools
            from aura.index import PackageIndex
            from aura.reqfile import ReqFile
            era = era_build_tools(PackageIndex(), ReqFile.load(work / req).pins(),
                                  on_date=s.get("commit_date", ""),
                                  python_version=python)
        except Exception as exc:
            print(f"  (no era build tools: {exc})")
        install_lines = install_commands(
            req, " pytest" if needs_pytest else "", era,
            s.get("environment_setup") or [])
        test_dir = (s.get("validation") or {}).get("test_dir", "")
        already_failing = (arm_baseline_failures(name)
                           if PREFER_ARM_BASELINE else [])
        if already_failing:
            print(f"  using the arms' baseline: {len(already_failing)} already failing")
        else:
            already_failing = baseline_failures(work, python, target, packaged,
                                                install_lines, environment)
        if not already_failing:
            # The local check cannot run where the pins have no build for this
            # machine (old TensorFlow or torch on Apple silicon) and returns
            # nothing; the arms' own baseline, recorded on the cluster, says
            # which tests already fail.
            already_failing = arm_baseline_failures(name)
            if already_failing:
                print(f"  using the arms' baseline: {len(already_failing)} already failing")
        if already_failing:
            print(f"  {len(already_failing)} already failing; CI will skip them, "
                  "as the arms tolerate them")
            for test in already_failing:
                print(f"    {test}")
            # A whole file, not a test: pytest reports a module it could not
            # import that way, and deselecting it drops every test inside.
            # pytorch-ocr's 44-test suite came down to two such files, both of
            # them collection errors from an installer problem of ours.
            whole_files = [t for t in already_failing if "::" not in t]
            if whole_files:
                print(f"\n  *** {len(whole_files)} of these is a whole test FILE: "
                      f"{', '.join(whole_files)}\n      Every test in it will be "
                      f"skipped for every pull request. A file fails to collect "
                      f"when it cannot be imported, which is usually the "
                      f"environment rather than the project -- check before "
                      f"trusting this arm's number.\n")
        deselect = "".join(f" --deselect {test}" for test in already_failing)
        (gh_dir / "workflows" / "tests.yml").write_text(TESTS_YML.format(
            # The subject's placeholder variables, as the arms get them: a
            # baseline that needs a region or a secret key fails in CI without
            # them, and every pull request with it.
            job_env=("    env:\n" + "".join(f"      {k}: {json.dumps(str(v))}\n"
                                         for k, v in environment.items())
                     if environment else ""),
            # GitHub's current runner image carries no Python older than 3.8;
            # the previous Ubuntu LTS still has 3.7.
            runner="ubuntu-22.04" if _older_than_38(python) else "ubuntu-latest",
            python=python, target=target, editable=editable, deselect=deselect,
            workdir=(f"        working-directory: {test_dir}\n" if test_dir else ""),
            install="".join(f"          {line}\n" for line in install_lines)))
        for cmd in (["git", "add", "-A"],
                    ["git", "-c", "user.name=aura", "-c", "user.email=aura@localhost",
                     "commit", "-q", "-m",
                     f"{ARM.capitalize()} check: CI and auto-merge"
                     + (" (requirements reconstructed at the commit date, the "
                        "baseline every arm starts from)" if rebuilt else "")]):
            sh(cmd, cwd=work)
        # The history is the project's own; drop the link to it.
        sh(["git", "remote", "remove", "origin"], cwd=work)

        if dry_run:
            # Nothing leaves this machine: the tree is kept for inspection.
            out = ROOT / "results" / name / f"{name}__{ARM}.dryrun"
            shutil.rmtree(out, ignore_errors=True)
            shutil.copytree(work, out, ignore=shutil.ignore_patterns(".git"))
            print(f"\ndry run: nothing pushed. The repository would contain {out}")
            return

        print(f"creating private repo {repo} ...")
        # Created and pushed in two steps, not by "gh repo create --push".
        # As one step a transient failure on the push -- GitHub answered a 500
        # once and refused the SSH key once, minutes apart -- leaves the repo
        # created and empty while this process exits and deletes the tree it
        # spent minutes preparing, so the retry has to start from the clone.
        # GitHub limits how fast an account may create repositories ("You have
        # created too many repositories, too quickly"); creating fifty in a row
        # hit it at the tenth. Nothing is created when it refuses, so wait and
        # try again.
        for attempt in range(1, 37):
            made = sh(["gh", "repo", "create", repo, VISIBILITY],
                      capture=True, check=False)
            if made.returncode == 0:
                break
            if "too quickly" not in (made.stderr + made.stdout):
                raise SystemExit(f"could not create {repo}: {made.stderr.strip()}")
            print(f"  GitHub: too many repositories too quickly; waiting 5 min "
                  f"(attempt {attempt}/36) ...", flush=True)
            time.sleep(300)
        else:
            raise SystemExit(f"could not create {repo}: still rate-limited after 3 hours")
        sh(["git", "remote", "add", "origin",
            f"git@github.com:{repo}.git"], cwd=work)
        for attempt in range(1, 5):
            done = sh(["git", "push", "-u", "origin", "HEAD:main"],
                      cwd=work, check=False)
            if done.returncode == 0:
                break
            print(f"  push failed (attempt {attempt}/4), retrying ...")
            time.sleep(5 * attempt)
        else:
            raise SystemExit(
                f"could not push to {repo}. The repository exists and is empty; "
                f"delete it with 'gh repo delete {repo} --yes' and run setup again.")
        # Needed by the workflow's fallback when two PRs race for the base.
        sh(["gh", "api", "-X", "PATCH", f"repos/{repo}",
            "-F", "allow_auto_merge=true"], check=False)
        after_push(repo)

    print(f"""
Done. {ARM.capitalize()} starts on its own within minutes of the push; watch
  https://github.com/{repo}/pulls
  https://github.com/{repo}/actions
It opens one PR per dependency, the tests run on each, and passing ones merge.
PRs that conflict after a merge are rebased by Dependabot and tested again.
Check progress with
  python scripts/{Path(sys.argv[0]).name} status {name}
and once nothing has changed for a few hours:
  python scripts/{Path(sys.argv[0]).name} collect {name}""")


_POETRY_DEPENDENCY_TABLE = re.compile(
    r"^\[tool\.poetry\.(dependencies|dev-dependencies|group\.[^\]]+\.dependencies)\]\s*$")


def _drop_poetry_dependencies(work: Path) -> bool:
    """Leave the requirements file as the only list of dependencies.

    A project that declares its packages in both a requirements file and
    Poetry's tables gets its Dependabot pull requests on the Poetry side, which
    CI -- installing the requirements file, as the arms do -- never reads.
    htb-api's 13 such merges all "passed" against unchanged pins, Flask and
    Werkzeug among them. The packaging metadata stays, so the project still
    installs; only the python constraint is kept of the dependency tables.
    """
    pyproject = work / "pyproject.toml"
    if not pyproject.exists():
        return False
    kept, dropping, changed = [], False, False
    for line in pyproject.read_text().splitlines():
        header = line.strip().startswith("[")
        if header:
            dropping = bool(_POETRY_DEPENDENCY_TABLE.match(line.strip()))
            if dropping:
                changed = True
                if line.strip() == "[tool.poetry.dependencies]":
                    kept.append(line)
                continue
        if dropping:
            if re.match(r"^\s*python\s*=", line) and kept[-1:] == ["[tool.poetry.dependencies]"]:
                kept.append(line)
            continue
        kept.append(line)
    if not changed:
        return False
    pyproject.write_text("\n".join(kept) + "\n")
    (work / "poetry.lock").unlink(missing_ok=True)
    return True


def _older_than_38(python: str) -> bool:
    major, minor = (int(p) for p in python.split(".")[:2])
    return (major, minor) < (3, 8)


def _pins(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        # Extras allowed: "sqlalchemy[asyncio]==2.0.15" is a pin of sqlalchemy.
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)", line)
        if m:
            out[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return out


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _prs(repo: str) -> list[dict]:
    """Dependabot's PRs, with their CI outcome."""
    raw = json.loads(sh(["gh", "pr", "list", "-R", repo, "--state", "all",
                         "--limit", "300", "--json",
                         "number,title,state,createdAt,updatedAt,mergedAt,url,"
                         "author,statusCheckRollup"], capture=True).stdout)
    out = []
    for pr in raw:
        # Dependabot follows the repository's own commit convention, so the
        # title may be "Bump x from 1 to 2" or "chore(deps): bump x from 1 to
        # 2". Matching only the first form silently records no PRs at all.
        m = re.search(r"\bbump (\S+) from (\S+) to (\S+)", pr["title"], re.I)
        if not m:
            continue
        checks = [c for c in pr.get("statusCheckRollup") or []
                  if (c.get("name") or c.get("context")) == "tests"]
        # The latest run of the tests job is the one that decided the PR.
        check = checks[-1] if checks else {}
        out.append({
            "number": pr["number"], "url": pr["url"],
            "package": _canon(m.group(1)), "from": m.group(2), "to": m.group(3),
            "state": pr["state"].lower(), "created": pr["createdAt"],
            "updated": pr["updatedAt"], "merged": pr.get("mergedAt"),
            "ci_status": (check.get("status") or "").lower(),
            "ci": (check.get("conclusion") or "").lower(),
            "ci_url": check.get("detailsUrl", ""),
        })
    return sorted(out, key=lambda p: p["package"])


def _failure_reason(repo: str, ci_url: str) -> str:
    """The line of a failed CI run that says why, best effort."""
    m = re.search(r"/runs/(\d+)", ci_url or "")
    if not m:
        return ""
    out = sh(["gh", "run", "view", m.group(1), "-R", repo, "--log-failed"],
             capture=True, check=False).stdout
    lines = [re.sub(r"^\S+\t\S+\t\S+ ", "", l).strip() for l in out.splitlines()]
    for pattern in (r"^ERROR: (Cannot install|Could not find|ResolutionImpossible)",
                    r"^ERROR:", r"^E\s+\w*Error", r"\d+ failed", r"error:"):
        hits = [l for l in lines if re.search(pattern, l)]
        if hits:
            return hits[0][:200]
    return ""


def _hours_since(stamp: str) -> float:
    when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - when).total_seconds() / 3600


def sweep(repo: str, prs: list[dict]) -> int:
    """Merge every open PR whose tests passed, as a maintainer would.

    The workflow merges a PR when its own CI goes green, but two PRs going
    green together race for the same base: the loser is rejected with "Base
    branch was modified", and since it does not conflict, Dependabot never
    rebases it. GitHub's own auto-merge would queue it, but that is unavailable
    for private repositories on a free plan, so the PR stays open with a green
    tick and the arm reports it as rejected by CI -- which is false.

    The rule is the same one the workflow applies, so this adds no judgement of
    its own: merged if and only if the tests passed.
    """
    merged, tried = 0, set()
    for pr in prs:
        # A grouped pull request appears once per package it bumps.
        if pr["state"] != "open" or pr["ci"] != "success" or pr["number"] in tried:
            continue
        tried.add(pr["number"])
        out = sh(["gh", "pr", "merge", str(pr["number"]), "-R", repo, "--merge"],
                 capture=True, check=False)
        if out.returncode == 0:
            print(f"  merged #{pr['number']} {pr['package']} "
                  f"(its tests passed; the workflow lost a race for the base)")
            merged += 1
    return merged


def status(name: str) -> None:
    repo = repo_name(name)
    prs = _prs(repo)
    if not prs:
        print(f"{repo}: no Dependabot PRs yet (it usually starts within minutes)")
        return
    running = [p for p in prs if p["state"] == "open" and p["ci_status"] != "completed"]
    for p in prs:
        ci = p["ci"] or p["ci_status"] or "no CI yet"
        print(f"  #{p['number']:<4} {p['package']:<22} {p['from']:>10} -> {p['to']:<10}"
              f" CI {ci:<12} {p['state']}")
    quiet = min(_hours_since(p["updated"]) for p in prs)
    print(f"\n{len(prs)} PRs, {sum(p['state'] == 'merged' for p in prs)} merged, "
          f"{len(running)} still running CI; last change {quiet:.1f} h ago")


def _test_main(repo: str) -> tuple[str, str]:
    """Run the tests workflow once on main. Returns (conclusion, url)."""
    before = {r["databaseId"] for r in json.loads(sh(
        ["gh", "run", "list", "-R", repo, "--workflow", "tests.yml",
         "--json", "databaseId", "--limit", "50"], capture=True).stdout)}
    sh(["gh", "workflow", "run", "tests.yml", "-R", repo, "--ref", "main"])
    for _ in range(60):
        time.sleep(5)
        runs = json.loads(sh(["gh", "run", "list", "-R", repo, "--workflow",
                              "tests.yml", "--event", "workflow_dispatch",
                              "--json", "databaseId,url", "--limit", "5"],
                             capture=True).stdout)
        new = [r for r in runs if r["databaseId"] not in before]
        if new:
            run = new[0]
            print(f"testing the final main on GitHub: {run['url']}")
            sh(["gh", "run", "watch", str(run["databaseId"]), "-R", repo],
               check=False, capture=True)
            done = json.loads(sh(["gh", "run", "view", str(run["databaseId"]), "-R",
                                  repo, "--json", "conclusion"], capture=True).stdout)
            return (done.get("conclusion") or "unknown").lower(), run["url"]
    return "not started", ""


def _file_at(repo: str, path: str, ref: str) -> str:
    raw = sh(["gh", "api", f"repos/{repo}/contents/{path}?ref={ref}",
              "--jq", ".content"], capture=True)
    return base64.b64decode(raw.stdout).decode()


def collect(name: str, settle_hours: float, force: bool) -> None:
    from aura.index import PackageIndex, version_age_report
    from aura.telemetry import Event, Outcome, RunConfig, RunLog
    from evaluation.runner import final_requirements_path, log_path_for

    s = subject(name)
    repo = repo_name(name)
    req = s.get("requirements_file", "requirements.txt")
    prs = _prs(repo)
    if not prs:
        sys.exit(f"{repo} has no Dependabot PRs yet; check status first")

    running = [p for p in prs if p["state"] == "open" and p["ci_status"] != "completed"]
    if running and not force:
        # Nothing to wait for on the caller's behalf: say so and stop, rather
        # than holding the terminal while GitHub works.
        sys.exit(f"not settled: {len(running)} PR(s) still running CI. "
                 "Wait, or --force.")

    # Settling is a loop, not a moment: merging a PR makes Dependabot rebase the
    # rest, their CI runs again, and some of those then pass. Sweep, wait for
    # the CI that causes, sweep again, until a sweep merges nothing.
    #
    # Bounded by wall clock, not by a count of iterations. As a count, waiting
    # for CI spent the same budget as merging, so a repository with many pull
    # requests exhausted it while CI was still churning and stopped with green
    # pull requests unmerged: age-of-empires-api was collected at 18 of its 36
    # packages that way, and reported as Dependabot's result.
    deadline = time.time() + SETTLE_BUDGET_S
    converged = False
    while time.time() < deadline:
        running = [p for p in prs if p["state"] == "open"
                   and p["ci_status"] != "completed"]
        if running:
            print(f"  waiting for CI on {len(running)} PR(s) ...", flush=True)
            time.sleep(SETTLE_WAIT_S)
            prs = _prs(repo)
            continue
        if not sweep(repo, prs):
            # Nothing to merge *now* is not the same as settled. After each
            # merge Dependabot rebases the remaining pull requests, minutes
            # later and on its own schedule, and some of those then pass: with
            # no quiet period the loop left pacman and hackton-intgest with four
            # green, mergeable pull requests each and marked them settled.
            # Settled means nothing open has changed for SETTLE_QUIET_S.
            # Merged ones included: right after this loop's own merge, the open
            # pull requests still carry the timestamps of a day ago -- Dependabot
            # has not rebased them yet -- and counting only those called htb-api
            # settled a minute after merging 2 of its 14 green PRs.
            recent = [p for p in prs
                      if _hours_since(p["updated"]) * 3600 < SETTLE_QUIET_S]
            if not recent:
                converged = True
                break
            print(f"  nothing to merge, but {len(recent)} PR(s) changed in the last "
                  f"{SETTLE_QUIET_S // 60} min; waiting for Dependabot ...", flush=True)
            time.sleep(SETTLE_WAIT_S)
            prs = _prs(repo)
            continue
        time.sleep(SETTLE_WAIT_S // 2)
        prs = _prs(repo)

    running = [p for p in prs if p["state"] == "open" and p["ci_status"] != "completed"]
    quiet = min(_hours_since(p["updated"]) for p in prs)
    # A loop that converged waited out the quiet period after its own merges;
    # holding it to hours of quiet as well would reject every collection that
    # merged anything.
    if not force and not converged and (running or quiet < settle_hours):
        sys.exit(f"not settled: {len(running)} PR(s) still running CI, last change "
                 f"{quiet:.1f} h ago (need {settle_hours} h). Wait, or --force.")
    if not converged:
        # --force means "collect anyway", not "pretend this finished". A figure
        # taken here is a floor on what Dependabot reaches, and saying so is the
        # difference between a measurement and an artefact.
        print(f"\n  *** NOT SETTLED after {SETTLE_BUDGET_S // 60} min: "
              f"{len(running)} PR(s) still running CI. The numbers below are a "
              f"LOWER BOUND on what Dependabot reaches -- run collect again.\n",
              flush=True)

    # What this repository began with: the reconstructed baseline when the
    # subject needed one, else the project's file at its pinned commit.
    record = baseline_record(name)
    frozen = RESULTS_DIR / name / f"{name}.baseline.txt"
    # A repository set up before the subject's first cluster run has no
    # record, and its history starts afresh, so the project's own commit is not
    # in it: fall back to the frozen baseline, which the arms all start from.
    original_text = (record.read_text() if record.exists()
                     else frozen.read_text() if frozen.exists()
                     else _file_at(repo, req, s["commit"]))
    final_text = _file_at(repo, req, "main")
    original, final = _pins(original_text), _pins(final_text)
    main_ci, main_url = _test_main(repo)

    for p in prs:
        if p["state"] != "merged" and p["ci"] == "failure":
            p["reason"] = _failure_reason(repo, p["ci_url"])

    log_path = log_path_for(RESULTS_DIR, name, ARM)
    rc = RunConfig(repo=name, arm=ARM, tiers=(),
                   python_version=s.get("python_version", ""),
                   notes=f"github-actions:{repo}")
    log = RunLog(log_path, rc)
    log.emit(Event.NOTE, note=f"github_{ARM}", repo=repo,
             prs=len({p["number"] for p in prs}),
             merged=len({p["number"] for p in prs if p["state"] == "merged"}))
    for p in prs:
        merged = p["state"] == "merged"
        log.emit(Event.PROPOSAL, package=p["package"], from_version=p["from"],
                 to_version=p["to"], source=ARM, accepted=merged,
                 error_class="" if merged else (
                     "ci_failed" if p["ci"] == "failure" else f"open ({p['ci'] or 'no CI'})"),
                 pr=p["number"], pr_url=p["url"], pr_state=p["state"], ci=p["ci"],
                 reason=p.get("reason", ""), merged_at=p["merged"])
    order = [p["package"] for p in sorted((p for p in prs if p["merged"]),
                                          key=lambda p: p["merged"])]
    log.emit(Event.NOTE, note="github_merge_order", order=order)
    log.emit(Event.NOTE, note="github_main_tested", conclusion=main_ci, url=main_url)

    updated = {pkg: [original[pkg], v] for pkg, v in final.items()
               if pkg in original and original[pkg] != v}
    unchanged = sorted(pkg for pkg in original if pkg not in updated)
    if main_ci not in ("success", "not started"):
        outcome, reason = Outcome.FAILED, f"the final main fails its tests on GitHub ({main_ci})"
    elif updated:
        outcome, reason = Outcome.MODERNIZED, ""
    else:
        outcome, reason = Outcome.KEPT_BASELINE, f"no {ARM.capitalize()} PR merged"
    index = PackageIndex()
    log.finish(outcome, updated=updated, unchanged=unchanged, reason=reason,
               version_age={"before": version_age_report(index, original),
                            "after": version_age_report(index, final)})

    # The same result file every other arm writes, so this arm appears in the
    # status grid and the summaries rather than reading as "not collected".
    from dataclasses import asdict

    from evaluation.runner import ArmResult

    log_path.with_suffix(".result.json").write_text(json.dumps(asdict(ArmResult(
        subject=name, arm=ARM, outcome=outcome.value,
        updated=updated, reason=reason,
        log_path=str(log_path), counters=dict(log.counters),
        version_age_before=version_age_report(index, original),
        version_age_after=version_age_report(index, final),
        baseline_stage="github")), indent=2) + "\n")

    final_requirements_path(log_path).write_text(final_text)
    log_path.with_name(log_path.stem + ".github.json").write_text(json.dumps(
        {"repo": repo, "main_tests": main_ci, "main_tests_url": main_url,
         # Whether the merge cascade ran to a standstill. A collection that
         # stopped early is a lower bound, and the file has to say so: the
         # number is otherwise indistinguishable from a settled one.
         "settled": converged,
         "open_prs_at_collection": len(running),
         "prs": prs}, indent=2) + "\n")
    print(log_path.with_suffix(".log").read_text())


def delete(name: str) -> None:
    repo = repo_name(name)
    print(f"deleting {repo} (GitHub will ask you to confirm) ...")
    sh(["gh", "repo", "delete", repo])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["setup", "status", "collect", "delete"])
    ap.add_argument("subject")
    ap.add_argument("--settle-hours", type=float, default=3.0,
                    help="collect: hours without any PR change before it counts "
                         "as settled (default 3)")
    ap.add_argument("--force", action="store_true",
                    help="collect even if it has not settled")
    ap.add_argument("--dry-run", action="store_true",
                    help="setup: build the repository locally, push nothing")
    args = ap.parse_args()
    if args.action == "collect":
        collect(args.subject, args.settle_hours, args.force)
    elif args.action == "setup":
        setup(args.subject, dry_run=args.dry_run)
    else:
        {"setup": setup, "status": status, "delete": delete}[args.action](args.subject)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
