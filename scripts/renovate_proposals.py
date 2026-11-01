#!/usr/bin/env python
"""What Renovate proposes for a subject, from the real Renovate CLI, offline.

Runs the Renovate image in local mode (--platform=local, a dry run: no GitHub,
no pull requests) on the subject's baseline requirements with the renovate
arm's configuration, and writes every branch it would open:
    results/<subject>/<subject>__renovate.proposals.json
The replay (CI and merging) is done by our own harness from these.

    python scripts/renovate_proposals.py tensorgrad [more subjects ...]
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "ghcr.io/renovatebot/renovate:44"


def renovate_config(req: str) -> dict:
    """The renovate arm's renovate.json, read from real_renovate.py's source
    (importing it needs the GitHub CLI)."""
    src = (ROOT / "scripts" / "real_renovate.py").read_text()
    ns: dict = {}
    exec(re.search(r"RENOVATE_JSON = \{.*?\n\}\n", src, re.S).group(0), ns)
    config = dict(ns["RENOVATE_JSON"])
    path = req.lstrip("./")
    config["pip_requirements"] = {"managerFilePatterns": [f"/^{re.escape(path)}$/"]}
    config["includePaths"] = [path]
    return config


def project_file(subject: dict, req: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
        subprocess.run(["git", "-c", "credential.helper=", "clone", "-q", "--filter=blob:none",
                        "--no-checkout", subject["url"], tmp], check=True, env=env)
        return subprocess.run(["git", "show", f"{subject['commit']}:{req}"], cwd=tmp,
                              check=True, capture_output=True, text=True).stdout


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def proposals(name: str) -> list[dict]:
    subject = next(s for s in json.loads((ROOT / "subjects.json").read_text())
                   if s["name"] == name)
    req = subject.get("requirements_file", "requirements.txt").lstrip("./")
    res = ROOT / "results" / name
    baseline = next((p for p in (res / f"{name}__renovate.baseline.txt",
                                 res / f"{name}__dependabot.baseline.txt",
                                 res / f"{name}.baseline.txt") if p.exists()), None)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / req).parent.mkdir(parents=True, exist_ok=True)
        if baseline:
            shutil.copy(baseline, work / req)
        else:
            # No reconstruction: the bots started from the project's own file
            # at its pinned commit, as the arms read it (see real_dependabot
            # setup). Cloned anonymously: a public project, no account needed.
            text = project_file(subject, req)
            sys.path.insert(0, str(ROOT))
            from aura.reqfile import ReqFile
            (work / req).write_text(text)
            (work / req).write_text(ReqFile.load(work / req).to_text())
        (work / "renovate.json").write_text(json.dumps(renovate_config(req), indent=2))
        git = ["git", "-c", "user.email=aura@localhost", "-c", "user.name=aura"]
        for cmd in (["git", "init", "-q"], ["git", "add", "-A"], git + ["commit", "-qm", "baseline"]):
            subprocess.run(cmd, cwd=work, check=True)
        out = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{work}:/usr/src/app", "-w", "/usr/src/app",
             "-e", "LOG_LEVEL=debug", "-e", "LOG_FORMAT=json", IMAGE,
             "renovate", "--platform=local", "--dry-run=full"],
            capture_output=True, text=True, timeout=1800)
    branches: dict[str, dict] = {}
    for line in out.stdout.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("msg") != "packageFiles with updates":
            continue
        for pf in entry["config"].get("pip_requirements", []):
            for dep in pf["deps"]:
                for u in dep.get("updates", []):
                    b = branches.setdefault(u["branchName"], {"branch": u["branchName"],
                                                               "updates": []})
                    b["updates"].append({"package": canon(dep["depName"]),
                                         "from": dep.get("currentValue", "").lstrip("="),
                                         "to": u["newVersion"],
                                         "type": u.get("updateType", "")})
    if not branches and "packageFiles with updates" not in out.stdout:
        raise SystemExit(f"{name}: Renovate produced no plan:\n{out.stdout[-2000:]}{out.stderr[-2000:]}")
    result = sorted(branches.values(), key=lambda b: b["branch"])
    (res / f"{name}__renovate.proposals.json").write_text(json.dumps(
        {"subject": name, "renovate_image": IMAGE, "branches": result}, indent=2) + "\n")
    return result


if __name__ == "__main__":
    for n in sys.argv[1:]:
        print(f"{n}: {len(proposals(n))} branch(es)", flush=True)
