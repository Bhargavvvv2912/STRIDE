#!/usr/bin/env python
"""The renovate arm: real Renovate, run on GitHub.

The same experiment as scripts/real_dependabot.py, with Renovate as the bot.
Everything else is shared, so the two arms differ only in the bot: the same
private throwaway copy of the subject (renovate-check-<subject>), the same CI
running the subject's tests with the same already-failing tests skipped, and
the same rule for merging -- a pull request is merged if and only if its tests
passed, in rounds, until nothing changes.

Renovate is given its recommended configuration, which groups packages that
are released together (group:monorepos, group:recommended) into one pull
request: its own answer to coupled upgrades, and the reason it is worth a
separate arm. It runs self-hosted from a workflow in the repository, at setup
and after every merge, so it needs a token whose pull requests trigger
CI (one opened with the workflow's own GITHUB_TOKEN does not). Export it in the
shell that runs setup; it is stored as the repository secret RENOVATE_TOKEN and
never written to disk:
    read -s RENOVATE_TOKEN && export RENOVATE_TOKEN

Usage:
    python scripts/real_renovate.py setup   tensorgrad --dry-run
    python scripts/real_renovate.py setup   tensorgrad
    python scripts/real_renovate.py status  tensorgrad
    python scripts/real_renovate.py collect tensorgrad
    python scripts/real_renovate.py delete  tensorgrad
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_dependabot as rd  # noqa: E402

rd.ARM = "renovate"
rd.REPO_PREFIX = "renovate-check"
# Public while the run lasts: Actions minutes are free on public repositories
# and fifty subjects would exhaust a private quota. The subjects are public
# projects already; make the repositories private again after collect.
rd.VISIBILITY = "--public"


def baseline_failures(work, python, target, packaged, install_lines,
                      environment=None) -> list[str]:
    """The tests Dependabot's CI skipped for this subject, read from its repository.

    Both bots then face exactly the same CI, and nothing is rerun locally (the
    Dependabot arm's check runs the suite in Docker, minutes to an hour per
    subject). Without a Dependabot repository, setup falls back to the AURA
    arms' own baseline.
    """
    name = work.name
    try:
        yml = rd._file_at(f"{rd.gh_user()}/dependabot-check-{name}",
                          ".github/workflows/tests.yml", "main")
    except Exception:
        return []
    return sorted(set(re.findall(r"--deselect (\S+)", yml)))

# Renovate as a maintainer would switch it on, plus what the comparison needs.
# config:recommended already includes group:monorepos and group:recommended,
# the grouping of packages released together, and keeps Renovate's other
# defaults -- a separate pull request for a minor and a major update, for one.
RENOVATE_JSON = {
    "$schema": "https://docs.renovatebot.com/renovate-schema.json",
    "extends": ["config:recommended"],
    # The file the arms measure, and nothing else: a pyproject.toml or a
    # setup.py beside it would otherwise get pull requests CI never reads, as
    # Poetry's tables did for Dependabot (see _drop_poetry_dependencies).
    "enabledManagers": ["pip_requirements"],
    # config:recommended skips files under test directories; a subject whose
    # requirements live there would get no pull requests at all.
    "ignorePaths": [],
    # Every proposal at once, as Dependabot's limit of 50 gives it.
    "prConcurrentLimit": 0,
    "prHourlyLimit": 0,
    "branchConcurrentLimit": 0,
    # Merging is left to collect(), with the same rule for both bots.
    "automerge": False,
}

RENOVATE_YML = """\
name: renovate
# No schedule: on a private repository every run spends Actions minutes, and
# a half-hourly run across fifty repositories spends the month's quota in a
# day. Renovate runs once at setup, after every merge -- so pull requests left
# behind are rebased promptly, as Dependabot does on its own -- and when
# collect() starts it again.
on:
  push:
    branches: [main]
  workflow_dispatch:
concurrency:
  group: renovate
  cancel-in-progress: false
jobs:
  renovate:
    runs-on: ubuntu-latest
    steps:
      - uses: renovatebot/github-action@v46.3.7
        with:
          token: ${{ secrets.RENOVATE_TOKEN }}
        env:
          RENOVATE_REPOSITORIES: ${{ github.repository }}
          RENOVATE_ONBOARDING: "false"
          RENOVATE_REQUIRE_CONFIG: "required"
          LOG_LEVEL: info
"""


def write_bot_config(work: Path, directory: str, req: str, compiled: bool) -> None:
    config = dict(RENOVATE_JSON)
    # pip_requirements reads requirements*.txt by default; a subject's file
    # can be named anything (reqs/base.txt), so name it exactly.
    config["pip_requirements"] = {
        "managerFilePatterns": [f"/^{re.escape(req.lstrip('./'))}$/"]}
    # And only that file: the pattern above is added to the default
    # requirements*.txt ones, not put in their place, so on its own Renovate
    # also updated tensorgrad's vendored tensorly/doc/requirements_doc.txt --
    # pull requests CI never reads, counted as Renovate's.
    config["includePaths"] = [req.lstrip("./")]
    (work / "renovate.json").write_text(json.dumps(config, indent=2) + "\n")
    (work / ".github" / "workflows" / "renovate.yml").write_text(RENOVATE_YML)
    # Forced: a subject's .gitignore can exclude it -- tensorgrad's has
    # "**.json" -- and Renovate then finds no config and does nothing.
    rd.sh(["git", "add", "-f", "renovate.json", ".github"], cwd=work)


def after_push(repo: str) -> None:
    rd.sh(["gh", "secret", "set", "RENOVATE_TOKEN", "-R", repo],
          capture=True, input=os.environ["RENOVATE_TOKEN"])
    print("stored RENOVATE_TOKEN as a repository secret; starting Renovate ...")
    rd.sh(["gh", "workflow", "run", "renovate.yml", "-R", repo, "--ref", "main"],
          check=False)


# Diffs by (pull request, head commit): a pull request's packages change only
# when Renovate pushes to it, and collect() polls every minute.
_DIFFS: dict[tuple[int, str], list[tuple[str, str, str]]] = {}


def _bumps(repo: str, number: int, head: str, req: str) -> list[tuple[str, str, str]]:
    """(package, from, to) for every pin the pull request changes.

    Read from the diff, not the title: a major update's title gives only the
    major ("Update dependency torch to v2"), and a grouped one names the group
    ("Update pytorch monorepo") rather than its packages.
    """
    key = (number, head)
    if key not in _DIFFS:
        diff = rd.sh(["gh", "pr", "diff", str(number), "-R", repo],
                     capture=True, check=False).stdout
        # The measured file's part of the diff only (see includePaths).
        parts = re.split(r"^diff --git ", diff, flags=re.M)
        diff = "".join(p for p in parts if p.startswith(f"a/{req} "))
        old = rd._pins("\n".join(l[1:] for l in diff.splitlines()
                                 if l.startswith("-") and not l.startswith("---")))
        new = rd._pins("\n".join(l[1:] for l in diff.splitlines()
                                 if l.startswith("+") and not l.startswith("+++")))
        _DIFFS[key] = sorted((pkg, old.get(pkg, ""), v) for pkg, v in new.items()
                             if old.get(pkg) != v)
    return _DIFFS[key]


def _prs(repo: str) -> list[dict]:
    """Renovate's PRs with their CI outcome, one entry per package bumped."""
    name = repo.split("/", 1)[1].removeprefix(rd.REPO_PREFIX + "-")
    req = rd.subject(name).get("requirements_file", "requirements.txt").lstrip("./")
    raw = json.loads(rd.sh(["gh", "pr", "list", "-R", repo, "--state", "all",
                            "--limit", "300", "--json",
                            "number,title,state,createdAt,updatedAt,mergedAt,url,"
                            "headRefName,headRefOid,statusCheckRollup"],
                           capture=True).stdout)
    out = []
    for pr in raw:
        if not pr["headRefName"].startswith("renovate/"):
            continue
        checks = [c for c in pr.get("statusCheckRollup") or []
                  if (c.get("name") or c.get("context")) == "tests"]
        check = checks[-1] if checks else {}
        bumps = _bumps(repo, pr["number"], pr["headRefOid"], req)
        for package, old, new in bumps:
            out.append({
                "number": pr["number"], "url": pr["url"], "title": pr["title"],
                "group": len(bumps) > 1,
                "package": rd._canon(package), "from": old, "to": new,
                "state": pr["state"].lower(), "created": pr["createdAt"],
                "updated": pr["updatedAt"], "merged": pr.get("mergedAt"),
                "ci_status": (check.get("status") or "").lower(),
                "ci": (check.get("conclusion") or "").lower(),
                "ci_url": check.get("detailsUrl", ""),
            })
    return sorted(out, key=lambda p: (p["package"], p["number"]))


rd.write_bot_config = write_bot_config
rd.after_push = after_push
rd.baseline_failures = baseline_failures
rd._prs = _prs

if __name__ == "__main__":
    if (len(sys.argv) > 1 and sys.argv[1] == "setup" and "--dry-run" not in sys.argv
            and not os.environ.get("RENOVATE_TOKEN")):
        sys.exit("RENOVATE_TOKEN is not set in this shell. Run\n"
                 "    read -s RENOVATE_TOKEN && export RENOVATE_TOKEN\n"
                 "and paste the token, then run setup again.")
    raise SystemExit(rd.main())
