#!/usr/bin/env python
"""Find repositories worth screening as subjects.

Searches GitHub for projects that *could* satisfy the sample protocol, so that
screen_candidates.py -- which installs and tests each one, and costs minutes per
repository -- is only spent on plausible ones. Nothing here decides inclusion:
that is screening's job, and it is decided by running the tests.

What it looks for, and why:

  a requirements.txt at the root         pinned or ranged: a range is turned
                                          into the versions the project had on
                                          its commit date before anything is
                                          measured
  a test suite in the repository          the oracle must come from the project,
                                          not from us
  no docker-compose                      a project that ships one expects
                                          services to be running; whether tests
                                          need a database is decided by running
                                          them, not by reading the requirements
  no GPU-scale dependencies               torch and friends make every probe a
                                          multi-GB download
  old pins, not archived                  a set already at its newest releases
                                          has no headroom to measure

Needs the GitHub CLI, logged in (brew install gh && gh auth login).

Usage:
    python scripts/find_candidates.py                 the default seed queries
    python scripts/find_candidates.py --query "Django==2" --query "celery==4"
    python scripts/find_candidates.py --out candidates.txt

Then:
    python scripts/screen_candidates.py --repos candidates.txt
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.subjects import ERAS  # noqa: E402

# Old pins of widely used libraries: each one is a project that has not been
# updated in years, which is the population this work is about.
# A frozen closure is what survives: it reproduces (so the baseline still
# installs) and it couples (so there is something for a search to find). Small
# partially-pinned files mostly come back "weak" -- the naive upgrade already
# passes -- or dead.
MIN_PINS, MAX_PINS = 15, 40     # set from the command line
MAX_BYTES = 0                   # --max-bytes: only requirement files this short
PUSHED_AFTER = "2019-01-01"     # set from --era; a stratum is a date window
PUSHED_BEFORE = "2100-01-01"

# Each of these pins a release whose successor removed something a test suite
# notices: Flask 2 dropped flask._compat, Jinja2 3 dropped escape, Werkzeug 2.1
# dropped several helpers, SQLAlchemy 2.0 changed the query API, pytest 8
# dropped nose-style setup. A project pinned below one of them and exercising
# it is where a search has something to find.
SEED_QUERIES = [
    "Flask==1.1 Werkzeug==",
    "Flask==1.0 Jinja2==",
    "SQLAlchemy==1.3 pytest==",
    "Werkzeug==0.16",
    "itsdangerous==1.1.0 Flask==",
    "Flask-SQLAlchemy==2.4",
    "Django==2.2 pytest==",
    "marshmallow==2 flask==",
]

# Only what cannot be installed here at all: CUDA-only builds, and frameworks
# that need a compiler toolchain of their own.
#
# torch and tensorflow are deliberately *not* here. They are large, and every
# probe installs them again -- but torch/torchvision is the coupling this work
# is about (each torchvision release pins an exact torch), and a sample with no
# such subject cannot show a co-update that a per-dependency tool structurally
# cannot make. The wheel cache pays for the download once, and the agent sizes
# its parallel probes from the wheel's own footprint, so a heavy subject probes
# narrowly rather than filling the disk.
HEAVY = re.compile(r"^(tensorflow-gpu|jax\[cuda|jaxlib|cupy|mxnet|"
                   r"paddlepaddle|onnxruntime-gpu|detectron2)\b", re.I)

# Large but usable: worth saying so, because these subjects take hours rather
# than minutes and their probes are what the batch policy has to narrow.
BULKY = re.compile(r"^(torch|torchvision|torchaudio|tensorflow|transformers|"
                   r"spacy|scipy|opencv-python)\b", re.I)
# Declaring a client library is not the same as needing the service: a Django
# project listing psycopg2-binary usually runs its tests on SQLite, and
# psycopg2-binary installs from a wheel without PostgreSQL present. Only the
# ones that cannot even be installed without a server are pre-filtered; whether
# the *tests* need one is answered by running them, which screening does.
SERVICE = re.compile(r"^(mysqlclient|psycopg2)(?!-binary)\b", re.I)
SERVICE_FILES = ("docker-compose.yml", "docker-compose.yaml", "compose.yaml")


def gh(args: list[str], retries: int = 4) -> str:
    """Run a gh command, waiting out the search API's rate limit.

    Code search allows ten requests a minute, and a sweep makes far more than
    that. Treating a rate-limited call as "no results" silently shrinks the
    candidate pool, which looks exactly like a stratum having no projects in
    it -- the one conclusion this search exists to avoid drawing by accident.
    """
    if shutil.which("gh") is None:
        sys.exit("GitHub CLI not installed: brew install gh && gh auth login")
    for attempt in range(retries):
        out = subprocess.run(["gh", *args], capture_output=True, text=True)
        if out.returncode == 0:
            return out.stdout
        if "rate limit" not in (out.stderr or "").lower():
            return ""
        wait = 20 * (attempt + 1)
        print(f"  (rate limited; waiting {wait}s)", flush=True)
        time.sleep(wait)
    return ""


def search(query: str, limit: int) -> list[str]:
    """Repositories with a root requirements.txt matching this query."""
    # `filename:` also matches web-requirements.txt and the like, so the root
    # file is selected here rather than trusted from the query.
    if MAX_BYTES:
        # `gh search code` drops the size qualifier; the REST endpoint honours
        # it. Code search ranks long files first, so without it a search for
        # short requirement files returns almost nothing but long ones.
        found = []
        for page in range(1, (limit + 99) // 100 + 1):
            raw = gh(["api", "-X", "GET", "search/code",
                      "-f", f"q={query} filename:requirements.txt size:<{MAX_BYTES}",
                      "-f", f"per_page={min(limit, 100)}", "-f", f"page={page}"])
            if not raw:
                break
            items = json.loads(raw).get("items", [])
            found += [h["repository"]["full_name"] for h in items
                      if h["path"] == "requirements.txt"]
            if len(items) < 100:
                break
        return found
    raw = gh(["search", "code", f"{query} filename:requirements.txt",
              "--limit", str(limit), "--json", "repository,path"])
    if not raw:
        return []
    hits = json.loads(raw)
    return [h["repository"]["nameWithOwner"] for h in hits
            if h["path"] == "requirements.txt"]


def repo_info(slug: str) -> dict | None:
    raw = gh(["api", f"repos/{slug}"])
    return json.loads(raw) if raw else None


def file_at(slug: str, path: str) -> str | None:
    raw = gh(["api", f"repos/{slug}/contents/{path}", "--jq", ".content"])
    if not raw.strip():
        return None
    try:
        return base64.b64decode(raw).decode("utf-8", "replace")
    except Exception:
        return None


def has_tests(slug: str) -> bool:
    raw = gh(["api", f"repos/{slug}/contents/", "--jq", ".[].name"])
    names = set(raw.split())
    if {"tests", "test"} & names:
        return True
    # A single test module at the root counts too.
    return any(n.startswith("test_") and n.endswith(".py") for n in names)


def assess(slug: str) -> dict:
    """Why this repository is or is not worth screening."""
    info = repo_info(slug)
    if not info:
        return {"repo": slug, "verdict": "skip", "why": "not readable"}
    if info.get("archived"):
        return {"repo": slug, "verdict": "skip", "why": "archived"}
    if info.get("size", 0) > 200_000:            # KB
        return {"repo": slug, "verdict": "skip", "why": "very large checkout"}
    pushed = (info.get("pushed_at") or "")[:10]
    if pushed and not (PUSHED_AFTER <= pushed <= PUSHED_BEFORE):
        # Outside the stratum being filled. The sample is stratified by age, so
        # a repository from the wrong era is not a candidate for this search
        # however good it looks.
        return {"repo": slug, "verdict": "skip",
                "why": f"last pushed {pushed}, outside {PUSHED_AFTER}..{PUSHED_BEFORE}"}

    text = file_at(slug, "requirements.txt")
    if text is None:
        return {"repo": slug, "verdict": "skip", "why": "no root requirements.txt"}
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.strip().startswith("#")]
    # Ranges count. A ranged file says what was allowed rather than what was
    # installed, and the baseline reconstruction turns that into the versions
    # the project had on its commit date -- so such a project is measurable,
    # and excluding it would leave out most modern ones.
    pinned = [l for l in lines if re.match(r"^[A-Za-z0-9]", l)]
    if not (MIN_PINS <= len(pinned) <= MAX_PINS):
        return {"repo": slug, "verdict": "skip",
                "why": f"{len(pinned)} pinned requirement(s), want "
                       f"{MIN_PINS}-{MAX_PINS}"}
    if any(l.startswith(("-e", "git+", "http")) or " @ " in l for l in lines):
        return {"repo": slug, "verdict": "skip", "why": "installs from a URL or VCS"}
    heavy = [l for l in pinned if HEAVY.match(l)]
    if heavy:
        return {"repo": slug, "verdict": "skip",
                "why": f"heavy dependency: {heavy[0].split('==')[0]}"}
    service = [l for l in pinned if SERVICE.match(l)]
    if service:
        return {"repo": slug, "verdict": "skip",
                "why": f"needs a service: {service[0].split('==')[0]}"}
    names = set(gh(["api", f"repos/{slug}/contents/", "--jq", ".[].name"]).split())
    if names & set(SERVICE_FILES):
        return {"repo": slug, "verdict": "skip", "why": "ships docker-compose"}
    if not has_tests(slug):
        return {"repo": slug, "verdict": "skip", "why": "no test directory"}

    bulky = sorted({l.split("==")[0].split(">")[0].split("[")[0].strip().lower()
                    for l in pinned if BULKY.match(l)})
    return {"repo": slug, "verdict": "screen", "pinned": len(pinned),
            "stars": info.get("stargazers_count", 0),
            "pushed": (info.get("pushed_at") or "")[:10], "bulky": bulky,
            "why": f"{len(pinned)} pinned deps, tests present"
                   + (f"; heavy: {', '.join(bulky)}" if bulky else "")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", action="append", default=[],
                    help="search term (repeatable); defaults to the seed set")
    ap.add_argument("--limit", type=int, default=100, help="hits per query")
    ap.add_argument("--assess", type=int, default=40,
                    help="how many distinct repositories to assess")
    ap.add_argument("--out", default="screening/candidates.txt")
    ap.add_argument("--era", default="",
                    help="only repositories last pushed inside this stratum "
                         "(" + ", ".join(e[0] for e in ERAS) + "); the sample is "
                         "stratified by age, so a search fills one stratum. "
                         "Several, comma-separated, span adjacent strata")
    ap.add_argument("--min-pins", type=int, default=15)
    ap.add_argument("--max-pins", type=int, default=40,
                    help="a long freeze is a slow subject: every probe installs "
                         "all of it")
    ap.add_argument("--max-bytes", type=int, default=0,
                    help="search only requirement files smaller than this; "
                         "code search otherwise ranks long files first")
    args = ap.parse_args()
    global MIN_PINS, MAX_PINS, PUSHED_AFTER, PUSHED_BEFORE, MAX_BYTES
    MIN_PINS, MAX_PINS = args.min_pins, args.max_pins
    MAX_BYTES = args.max_bytes
    if args.era:
        window = {name: (start, end) for name, start, end, _ in ERAS}
        eras = [e.strip() for e in args.era.split(",") if e.strip()]
        unknown = [e for e in eras if e not in window]
        if unknown:
            sys.exit(f"unknown era {unknown[0]!r}; expected one of "
                     + ", ".join(window))
        PUSHED_AFTER = min(window[e][0] for e in eras)
        PUSHED_BEFORE = max(window[e][1] for e in eras)
        print(f"era {args.era}: repositories last pushed "
              f"{PUSHED_AFTER} .. {PUSHED_BEFORE}\n")

    queries = args.query or SEED_QUERIES
    slugs: list[str] = []
    for index, query in enumerate(queries):
        if index:
            time.sleep(7)          # ten code searches a minute is the ceiling
        found = search(query, args.limit)
        print(f"{query:<24} {len(found)} repo(s) with a root requirements.txt")
        slugs += found

    known = set()
    if (ROOT / "subjects.json").exists():
        known = {s["url"].rstrip("/").split("github.com/")[-1]
                 for s in json.loads((ROOT / "subjects.json").read_text())}
    ordered = [s for s in dict.fromkeys(slugs) if s not in known][:args.assess]
    print(f"\nassessing {len(ordered)} distinct repositories ...\n")

    keep = []
    for slug in ordered:
        row = assess(slug)
        if row["verdict"] == "screen":
            keep.append(row)
            print(f"  SCREEN  {slug:<45} {row['pinned']:>3} pins  "
                  f"{row['stars']:>5}*  last push {row['pushed']}"
                  + (f"  [{', '.join(row['bulky'])}]" if row.get("bulky") else ""))
        else:
            print(f"  skip    {slug:<45} {row['why']}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{r['repo']}\n" for r in keep))
    print(f"\n{len(keep)} candidate(s) -> {out}")
    print(f"next:\n  python scripts/screen_candidates.py --repos {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
