#!/usr/bin/env python3
"""Build the coupled-upgrade benchmark: real package groups, real pip conflicts.

A case is (group, Python, date, target). Its starting state is each group
package at the newest release on that date (for that Python) such that pip
resolves them together. The task is to upgrade the target. A case is kept only
when it is the LLM tier's situation:

  * moving the target alone to its newest release FAILS pip's resolver (that
    failure is the conflict log the prompt shows), and
  * some joint upgrade exists: pip resolves "target > start" with the other
    group packages free to move up (that resolution is the reference answer).

The prompt is AURA's own compact prompt (aura.planner.build_compact_prompt)
built from that conflict log, so a model sees exactly what AURA would send it.
Nothing here runs a project's tests: the benchmark measures whether a proposal
resolves, which is the LLM tier's job; the real-subject runs measure the rest.

    python scripts/bench/build_cases.py --out data/bench/cases.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from aura.environment import find_interpreter  # noqa: E402
from aura.index import PackageIndex  # noqa: E402
from aura.planner import build_compact_prompt  # noqa: E402
from packaging.version import Version  # noqa: E402

# family -> (group, seen in fine-tuning data?)
GROUPS = {
    "flask": (["flask", "werkzeug", "jinja2", "itsdangerous", "click", "markupsafe"], True),
    "torch": (["torch", "torchvision", "torchaudio"], True),
    "django": (["django", "djangorestframework", "django-filter"], True),
    "aws": (["boto3", "botocore", "s3transfer"], True),
    "transformers": (["transformers", "tokenizers", "huggingface-hub"], False),
    "tensorflow": (["tensorflow", "keras", "tensorboard"], False),
    "celery": (["celery", "kombu", "billiard", "vine"], False),
    "fastapi": (["fastapi", "pydantic", "starlette"], False),
    "sqlalchemy": (["sqlalchemy", "alembic", "sqlalchemy-utils"], False),
    "aiohttp": (["aiohttp", "aiohttp-jinja2", "yarl", "multidict"], False),
    "httpx": (["httpx", "httpcore", "h11"], False),
    "langchain": (["langchain", "langchain-core", "langchain-community"], False),
    "numpy": (["numpy", "pandas", "scipy", "scikit-learn"], False),
    "aiobotocore": (["aiobotocore", "botocore", "s3fs", "fsspec"], False),
    "marshmallow": (["marshmallow", "marshmallow-sqlalchemy", "sqlalchemy"], False),
    "graphql": (["graphene", "graphql-core", "graphql-relay"], False),
    # Breadth: more independent families, each with documented coupling (exact
    # pins, upper bounds or lockstep releases). None is a main-study family but
    # moto, which shares boto3/botocore with "aws".
    "pyarrow": (["pandas", "pyarrow", "numpy"], False),
    "scrapy": (["scrapy", "twisted", "parsel", "w3lib"], False),
    "requests": (["requests", "urllib3", "chardet", "idna"], False),
    "matplotlib": (["matplotlib", "numpy", "contourpy", "pillow"], False),
    "jupyter": (["jupyter-client", "jupyter-core", "ipykernel", "tornado"], False),
    "grpc": (["grpcio", "grpcio-tools", "protobuf"], False),
    "opentelemetry": (["opentelemetry-api", "opentelemetry-sdk", "opentelemetry-exporter-otlp-proto-http"], False),
    "typer": (["typer", "click", "rich"], False),
    "google-cloud": (["google-cloud-storage", "google-api-core", "google-auth", "protobuf"], False),
    "pytest": (["pytest", "pytest-asyncio", "pluggy"], False),
    "sphinx": (["sphinx", "sphinx-rtd-theme", "docutils"], False),
    "jax": (["jax", "jaxlib", "numpy"], False),
    "dask": (["dask", "distributed"], False),
    "spacy": (["spacy", "thinc", "pydantic"], False),
    "openai": (["openai", "httpx", "pydantic"], False),
    "moto": (["moto", "boto3", "botocore"], True),
    "pylint": (["pylint", "astroid"], False),
    "flake8": (["flake8", "pycodestyle", "pyflakes", "mccabe"], False),
    "mkdocs": (["mkdocs", "mkdocs-material", "markdown"], False),
    "numba": (["numba", "llvmlite", "numpy"], False),
    "scikit-image": (["scikit-image", "numpy", "scipy", "tifffile"], False),
}
PYTHONS = ["3.9", "3.10", "3.11"]
DATES = [f"{y}-{m:02d}-15" for y in range(2020, 2025) for m in (2, 6, 10)]


def resolver_python(version: str, cache: Path) -> str:
    """A venv on that interpreter with a current pip (for --report)."""
    venv = cache / f"pip-{version}"
    py = venv / "bin" / "python"
    if not py.exists():
        base = find_interpreter(version)
        if not base:
            sys.exit(f"no Python {version}; install it with `uv python install {version}`")
        subprocess.run([base, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip>=24"], check=True)
    return str(py)


def pip_dry(py: str, reqs: list[str]) -> tuple[bool, str, dict]:
    """pip's real resolver on a fresh, empty target: (ok, log, resolved pins)."""
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp, "requirements.txt")
        req.write_text("\n".join(reqs) + "\n")
        report = Path(tmp, "report.json")
        r = subprocess.run([py, "-m", "pip", "install", "--dry-run", "--ignore-installed",
                            "--disable-pip-version-check", "--report", str(report),
                            "--target", str(Path(tmp, "t")), "-r", str(req)],
                           capture_output=True, text=True, timeout=900)
        resolved = {}
        if r.returncode == 0 and report.exists():
            for item in json.loads(report.read_text()).get("install", []):
                md = item.get("metadata", {})
                resolved[canon(md.get("name", ""))] = md.get("version")
        return r.returncode == 0, r.stdout + r.stderr, resolved


canon = lambda n: re.sub(r"[-_.]+", "-", n).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bench/cases.jsonl")
    ap.add_argument("--per-family", type=int, default=40, help="cases kept per family at most")
    ap.add_argument("--easy-cap", type=int, default=10,
                    help="at most this many greedy-solvable cases per family; the rest are hard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", nargs="*", default=sorted(GROUPS))
    ap.add_argument("--pythons", nargs="*", default=PYTHONS)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    counts, kind_counts = {}, {}
    for l in (out.open() if out.exists() else []):
        c = json.loads(l)
        counts[c["family"]] = counts.get(c["family"], 0) + 1
        kind_counts[(c["family"], c.get("kind"))] = kind_counts.get((c["family"], c.get("kind")), 0) + 1
    idx = PackageIndex()
    rng = random.Random(args.seed)
    cache = out.parent / ".resolvers"
    interps = {py: resolver_python(py, cache) for py in args.pythons}

    why = {}
    note = lambda fam, k: why.setdefault(fam, {}).__setitem__(k, why[fam].get(k, 0) + 1)
    combos = [(fam, py, date) for fam in args.families for py in args.pythons for date in DATES]
    rng.shuffle(combos)
    with out.open("a") as fh:
        for fam, py, date in combos:
            if counts.get(fam, 0) >= args.per_family:
                continue
            group, seen = GROUPS[fam]
            start = {}
            for p in group:
                v = idx.newest_at(p, date, python_version=py)
                if v:
                    start[canon(p)] = v
            if len(start) < 2:
                note(fam, "fewer than 2 packages released by that date")
                continue
            ok, slog, _ = pip_dry(interps[py], [f"{p}=={v}" for p, v in start.items()])
            if not ok:
                note(fam, "starting set does not resolve")
                why.setdefault(fam, {}).setdefault("_example", slog[-300:])
                continue
            newest = {}
            for p in start:
                h = idx.history(p)
                c = h.compatible(h.between(start[p], h.versions[-1]), py)[0] if h and h.versions else []
                newest[p] = max(c, key=Version) if c else start[p]
            # Variants: no cap ("easy" when all-newest resolves), and for each other
            # package with a newer major, a project-side cap below that major --
            # the "pydantic<2" line real requirement files carry. The cap is part
            # of the case: "newest" honours it, and so must every answer.
            variants = [(None, None)]
            for q in start:
                nxt = Version(start[q]).major + 1
                if Version(newest[q]).major >= nxt:
                    variants.append((q, f"<{nxt}"))
            for target in start:
                for cap_pkg, cap in variants:
                    if cap_pkg == target or newest[target] == start[target]:
                        continue
                    cid = f"{fam}|{py}|{date}|{target}|{cap_pkg}{cap or ''}"
                    if cid in done:
                        continue
                    allowed = dict(newest)
                    if cap_pkg:
                        h = idx.history(cap_pkg)
                        c = [v for v in h.compatible(h.between(start[cap_pkg], h.versions[-1]), py)[0]
                             if Version(v).major < int(cap[1:])]
                        allowed[cap_pkg] = max(c, key=Version) if c else start[cap_pkg]
                    capline = [f"{cap_pkg}{cap}"] if cap_pkg else []
                    alone = [f"{p}=={allowed[p] if p == target else v}" for p, v in start.items()]
                    ok_alone, log, _ = pip_dry(interps[py], alone)
                    if ok_alone:
                        note(fam, "solo upgrade resolves (not coupled)")
                        continue
                    ok_newest, _, _ = pip_dry(interps[py], [f"{p}=={v}" for p, v in allowed.items()])
                    kind = "easy" if ok_newest else "hard"
                    quota = args.easy_cap if kind == "easy" else args.per_family - args.easy_cap
                    if kind_counts.get((fam, kind), 0) >= quota:
                        continue
                    free = ([f"{target}>{start[target]}"] + capline
                            + [f"{p}>={v}" for p, v in start.items() if p != target])
                    ok_ref, _, ref = pip_dry(interps[py], free)
                    if not ok_ref:
                        note(fam, "no joint upgrade exists")
                        continue
                    prompt = build_compact_prompt(target=target, error_log=log, current=start,
                                                  available=allowed, history=[])
                    case = {"id": cid, "family": fam, "kind": kind, "seen_in_training": seen,
                            "python": py, "date": date, "target": target, "cap": capline,
                            "start": start, "newest": allowed,
                            "reference": {p: ref.get(p) for p in start}, "prompt": prompt}
                    fh.write(json.dumps(case) + "\n")
                    fh.flush()
                    done.add(cid)
                    counts[fam] = counts.get(fam, 0) + 1
                    kind_counts[(fam, kind)] = kind_counts.get((fam, kind), 0) + 1
                    print(f"{len(done):4} {kind:4} {cid}", flush=True)
                    if counts.get(fam, 0) >= args.per_family:
                        break
                if counts.get(fam, 0) >= args.per_family:
                    break
    print("cases per family:", dict(sorted(counts.items())))
    for fam in args.families:
        print(f"  {fam}: rejected {json.dumps(why.get(fam, {}))[:600]}")
    print("by kind:", {f"{f}/{k}": n for (f, k), n in sorted(kind_counts.items())})


if __name__ == "__main__":
    main()
