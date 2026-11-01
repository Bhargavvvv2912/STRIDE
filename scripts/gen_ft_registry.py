#!/usr/bin/env python3
"""Fast candidate search for the fine-tuning test set (GitHub API only, no clones).

Searches requirements.txt files that pin a package of a target family at an
old major/minor (code search on the pin itself), then triages each repository
with about three API calls: its requirements.txt, its file tree and its HEAD
commit. Kept when it pins two or more coupled packages, at most 40 pins, has
three or more test files, and does not pin Flask (Flask dominates the training
data, so the test set avoids it). Families are chosen to be rare or absent in
the fine-tuning data.

Selection is structural only; whether the LLM is needed is decided later by
running greedy, never by any LLM result.

    python scripts/gen_ft_registry.py --out screening/walk_registry_ft.json --target 2000
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# family -> pins to search for (old versions, so there is a major move to make)
QUERIES = {
    "transformers": ["transformers==4.1", "transformers==4.2", "transformers==4.3", "transformers==3.", "tokenizers==0.1"],
    "tensorflow": ["tensorflow==2.3", "tensorflow==2.4", "tensorflow==2.5", "tensorflow==2.6", "tensorflow==2.7",
                   "tensorflow==2.8", "tensorflow==2.9", "keras==2.4", "keras==2.6", "tensorflow-estimator=="],
    "pydantic": ["pydantic==1.8", "pydantic==1.9", "pydantic==1.10", "fastapi==0.6", "fastapi==0.7", "fastapi==0.8"],
    "sqlalchemy": ["SQLAlchemy==1.3", "SQLAlchemy==1.4", "sqlalchemy==1.3", "sqlalchemy==1.4", "alembic==1.4",
                   "alembic==1.6", "databases==0."],
    "celery": ["celery==4.", "celery==5.0", "celery==5.1", "celery==5.2", "kombu==4.", "kombu==5.2"],
    "numpy": ["numpy==1.19", "numpy==1.20", "numpy==1.21", "pandas==1.1", "pandas==1.2", "pandas==1.3",
              "scipy==1.5", "scipy==1.6", "scipy==1.7", "scikit-learn==0.2", "scikit-learn==1.0"],
    "torch": ["pytorch-lightning==1.", "torchvision==0.8", "torchvision==0.9", "torchvision==0.10",
              "torchaudio==0.8", "torchtext==0.9"],
    "aiohttp": ["aiohttp==3.6", "aiohttp==3.7", "aiohttp==3.8", "aiohttp-jinja2=="],
    "django": ["djangorestframework==3.1", "channels==2.", "channels==3.", "django-filter==2."],
    "langchain": ["langchain==0.0", "langchain-core==0.1", "openai==0."],
    "aws": ["aiobotocore==1.", "aiobotocore==2.", "s3fs==2021", "s3fs==2022"],
    "marshmallow": ["marshmallow==3.1", "marshmallow-sqlalchemy==0.2"],
    "httpx": ["httpx==0.1", "httpx==0.2", "httpcore=="],
}
# Second round: finer version prefixes surface repos the 1,000-hit cap of the
# broad queries cut off.
QUERIES_EXTRA = {
    "celery": ["celery==4.1", "celery==4.2", "celery==4.3", "celery==4.4", "kombu==4.6", "celery==5.3"],
    "transformers": ["transformers==4.4", "transformers==4.5", "transformers==4.6", "transformers==4.8",
                     "transformers==4.10", "transformers==4.12", "transformers==4.15", "transformers==4.18",
                     "transformers==4.20", "transformers==4.24", "transformers==4.26", "transformers==4.28"],
    "tensorflow": ["tensorflow==2.10", "tensorflow==2.11", "tensorflow==2.12", "tensorflow==2.2",
                   "tensorflow-cpu==2.", "keras==2.8", "keras==2.10"],
    "numpy": ["numpy==1.16", "numpy==1.17", "numpy==1.18", "numpy==1.22", "pandas==0.25", "pandas==1.0",
              "pandas==1.4", "scipy==1.4", "scipy==1.8", "scikit-learn==0.24", "scikit-learn==0.23"],
    "pydantic": ["pydantic==1.5", "pydantic==1.6", "pydantic==1.7", "fastapi==0.5", "fastapi==0.9",
                 "fastapi==0.10", "starlette==0.1"],
    "sqlalchemy": ["SQLAlchemy==1.2", "sqlalchemy==1.2", "Flask-SQLAlchemy==2", "alembic==1.7", "alembic==1.8",
                   "sqlmodel==0.0"],
    "torch": ["torchvision==0.11", "torchvision==0.12", "torchvision==0.13", "torchmetrics==0.",
              "torch==1.10", "torch==1.12", "torchaudio==0.10"],
    "aiohttp": ["aiohttp==3.5", "aiohttp==3.4", "yarl==1.", "multidict==4."],
    "langchain": ["langchain==0.1", "openai==1.", "langchain-openai==0.0", "langchain-community==0.0"],
    "django": ["channels==4.", "channels-redis==", "djangorestframework==3.9", "djangorestframework==3.11"],
    "aws": ["botocore==1.2", "boto3==1.1", "s3fs==0."],
    "httpx": ["httpx==0.23", "httpx==0.24", "respx=="],
}
COUPLED = {
    "transformers", "tokenizers", "huggingface-hub", "torch", "torchvision", "torchaudio", "torchtext",
    "pytorch-lightning", "lightning", "torchmetrics", "tensorflow", "tensorflow-estimator", "keras",
    "tensorflow-addons", "tensorflow-probability", "tensorboard", "pydantic", "pydantic-core",
    "fastapi", "starlette", "sqlalchemy", "alembic", "databases", "sqlmodel", "celery", "kombu",
    "billiard", "amqp", "vine", "numpy", "pandas", "scipy", "scikit-learn", "numba", "llvmlite",
    "aiohttp", "aiohttp-jinja2", "yarl", "multidict", "django", "djangorestframework", "channels",
    "daphne", "django-filter", "langchain", "langchain-core", "langchain-community", "openai",
    "boto3", "botocore", "aiobotocore", "s3fs", "fsspec", "marshmallow", "marshmallow-sqlalchemy",
    "httpx", "httpcore", "h11", "anyio", "matplotlib", "protobuf", "grpcio",
}
TESTFILE = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py)$")
PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==\s*([^\s;#]+)", re.M)
canon = lambda n: re.sub(r"[-_.]+", "-", n).lower()


def gh(args, jq=None):
    cmd = ["gh", "api"] + args + (["--jq", jq] if jq else [])
    for _ in range(4):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
        msg = (r.stdout + r.stderr).lower()
        if "rate limit" in msg or "secondary" in msg:
            time.sleep(65)
            continue
        return None
    return None


def search(q):
    repos = []
    for page in range(1, 11):
        out = gh(["-X", "GET", "search/code", "-f", f"q=\"{q}\" filename:requirements.txt",
                  "-f", "per_page=100", "-f", f"page={page}"])
        if not out:
            break
        items = json.loads(out).get("items", [])
        repos += [(i["repository"]["full_name"], i["path"]) for i in items]
        if len(items) < 100:
            break
        time.sleep(7)          # code search: 10 requests per minute
    return repos


def python_for(pins, date):
    v = lambda s: tuple(int(x) for x in re.findall(r"\d+", s)[:3])
    np_, t, tf = pins.get("numpy"), pins.get("torch"), pins.get("tensorflow")
    if (np_ and v(np_) < (1, 17, 3)) or (t and v(t) < (1, 4)) or (tf and v(tf) < (2, 2)):
        return "3.7"
    if (t and v(t) < (1, 8)) or (tf and v(tf) < (2, 5)) or (date and date < "2021-01-01"):
        return "3.8"
    return "3.9"


def triage(repo, path, family):
    raw = gh([f"repos/{repo}/contents/{path}"], ".content")
    if not raw:
        return None, "no file"
    req = base64.b64decode(raw.strip()).decode("utf-8", "replace")
    pins = {canon(m.group(1)): m.group(3) for m in PIN.finditer(req)}
    if "flask" in pins or re.search(r"^\s*flask\b", req, re.I | re.M):
        return None, "flask"
    if not 3 <= len(pins) <= 40:
        return None, f"{len(pins)} pins"
    coupled = [p for p in pins if p in COUPLED]
    if len(coupled) < 2:
        return None, "no coupled pair"
    tree = gh([f"repos/{repo}/git/trees/HEAD?recursive=1"], '[.tree[] | select(.type=="blob") | .path]')
    if not tree:
        return None, "no tree"
    tests = [p for p in json.loads(tree) if TESTFILE.search(p) and "site-packages" not in p
             and not p.startswith(("venv/", ".venv/", "env/"))]
    if len(tests) < 3:
        return None, f"{len(tests)} test files"
    head = gh([f"repos/{repo}/commits/HEAD"], "[.sha, .commit.committer.date] | @tsv")
    if not head:
        return None, "no commit"
    sha, date = head.strip().split("\t")
    if date[:10] > "2025-06-30":
        return None, "too recent"
    name = "w-" + re.sub(r"[^a-z0-9]+", "-", repo.lower()).strip("-")[:60]
    return {"name": name, "url": f"https://github.com/{repo}", "commit": sha, "commit_date": date[:10],
            "python_version": python_for(pins, date[:10]), "coupled": coupled[:8],
            "targets": tests[:25], "reqs": path, "pins": len(pins), "family": family}, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "screening/walk_registry_ft.json"))
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--extra", action="store_true",
                    help="run the second, more specific query round (QUERIES_EXTRA)")
    args = ap.parse_args()
    queries = {k: list(v) for k, v in (QUERIES_EXTRA if args.extra else QUERIES).items()}
    out = Path(args.out)
    reg = json.loads(out.read_text()) if out.exists() else []
    seen = {r["url"].split("github.com/")[1].lower() for r in reg}
    for f in ROOT.glob("screening/walk_registry*.json"):
        if f != out:
            try:
                seen |= {r["url"].split("github.com/")[1].lower() for r in json.loads(f.read_text())}
            except Exception:
                pass
    for s in json.loads((ROOT / "subjects.json").read_text()):
        seen.add(s["url"].rstrip("/").split("github.com/")[1].lower())
    log = out.with_suffix(".log")
    while len(reg) < args.target:
        progressed = False
        for family, qs in queries.items():
            if not qs:
                continue
            q = qs.pop(0)
            progressed = True
            hits = search(q)
            added = 0
            for repo, path in hits:
                if repo.lower() in seen or len(reg) >= args.target:
                    continue
                seen.add(repo.lower())
                try:
                    row, _ = triage(repo, path, family)
                except Exception:
                    row = None
                if row:
                    reg.append(row)
                    added += 1
                    out.write_text(json.dumps(reg, indent=1))
            with log.open("a") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')} {family:<12} {q:<28} hits {len(hits):>4}  added {added:>3}  total {len(reg)}\n")
        if not progressed:
            break
    with log.open("a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} === done {len(reg)}\n")


if __name__ == "__main__":
    main()
