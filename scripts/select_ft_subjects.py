#!/usr/bin/env python3
"""Apply docs/FT_TEST_SELECTION.md: choose the fine-tuning test subjects.

    # stage A: eligible candidates from the walk-check screen; add them to
    # subjects.json for a greedy-only run
    python scripts/select_ft_subjects.py eligible --add

    # stage B: after the greedy runs, rank and pick the 10
    python scripts/select_ft_subjects.py rank
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packaging.version import InvalidVersion, Version  # noqa: E402

# Every screen run so far: the fine-tuning search, and the earlier ones.
REGISTRIES = sorted((ROOT / "screening").glob("walk_registry*.json"))
WALKS = [ROOT / "results/walk_ft", ROOT / "results/walk_v2"]
TEST_TOOLS = {"pytest", "pytest-asyncio", "pytest-cov", "pytest-mock", "pytest-xdist", "pytest-aiohttp",
              "pytest-httpx", "pytest-django", "coverage", "pluggy", "mock", "respx", "freezegun"}
# Family of a candidate without a search label, from what it pins.
FAMILY_OF = [("transformers", {"transformers", "tokenizers", "huggingface-hub"}),
             ("tensorflow", {"tensorflow", "keras", "tensorflow-estimator"}),
             ("torch", {"torch", "torchvision", "torchaudio", "pytorch-lightning", "lightning", "torchtext"}),
             ("django", {"django", "djangorestframework", "channels"}),
             ("celery", {"celery", "kombu"}), ("langchain", {"langchain", "langchain-core", "openai"}),
             ("pydantic", {"pydantic", "fastapi", "starlette"}), ("aiohttp", {"aiohttp", "aiohttp-jinja2"}),
             ("sqlalchemy", {"sqlalchemy", "alembic", "databases"}), ("httpx", {"httpx", "httpcore"}),
             ("marshmallow", {"marshmallow"}), ("aws", {"boto3", "botocore", "aiobotocore", "s3fs"}),
             ("numpy", {"numpy", "pandas", "scipy", "scikit-learn", "numba"})]
ALLOWED = {"celery", "numpy", "pydantic", "aiohttp", "langchain", "sqlalchemy", "httpx",
           "marshmallow", "transformers", "tensorflow", "aws"}
PER_FAMILY, TAKE = 2, 10
canon = lambda n: re.sub(r"[-_.]+", "-", n).lower()


def V(v):
    try:
        return Version(str(v))
    except (InvalidVersion, TypeError):
        return Version("0")


def subject_name(walk_name: str) -> str:
    return walk_name.removeprefix("w-")[:40]


def joint_set(w: dict) -> dict:
    if w.get("target"):
        return w["target"]
    if w.get("joint_ok") and w.get("newest"):
        return w["newest"]
    return {}


def family(r: dict) -> str | None:
    if r.get("family"):
        return r["family"]
    names = {canon(c) for c in r.get("coupled", [])}
    return next((fam for fam, keys in FAMILY_OF if names & keys), None)


def eligible():
    reg = {}
    for f in REGISTRIES:
        try:
            for r in json.loads(f.read_text()):
                reg.setdefault(r["name"], r)
        except (ValueError, KeyError):
            pass
    # Main-study subjects are in the fine-tuning data: never test on them.
    trained = {s["url"].rstrip("/").lower() for s in json.loads((ROOT / "subjects.json").read_text())
               if s.get("domain") != "ft-test"}
    out, reasons = [], {}
    seen = set()
    for f in sorted(p for d in WALKS for p in d.glob("w-*.json")):
        name = f.stem
        r = reg.get(name)
        if not r or name in seen:
            continue
        seen.add(name)
        r = {**r, "family": family(r)}
        if r["url"].rstrip("/").lower() in trained:
            reasons["main-study subject (trained on)"] = reasons.get("main-study subject (trained on)", 0) + 1
            continue
        w = json.loads(f.read_text())
        if w.get("error") or not w.get("baseline"):
            reasons["baseline fails"] = reasons.get("baseline fails", 0) + 1
            continue
        J, walk = joint_set(w), w.get("walk") or {}
        ahead = sorted(c for c in J if V(J[c]) > V(walk.get(c, "0")))
        if not [c for c in ahead if canon(c) not in TEST_TOOLS]:
            reasons["no newer joint set"] = reasons.get("no newer joint set", 0) + 1
            continue
        if r.get("family") not in ALLOWED:
            reasons["family seen in training"] = reasons.get("family seen in training", 0) + 1
            continue
        out.append({"walk": w, "reg": r, "joint": J, "ahead": ahead})
    return out, reasons


_INDEX = None


def index():
    global _INDEX
    if _INDEX is None:
        from aura.index import PackageIndex
        _INDEX = PackageIndex()
    return _INDEX


def installable(r: dict) -> bool:
    """As the walk check decides it: a setup.py or pyproject.toml at the root."""
    import subprocess
    slug = r["url"].rstrip("/").split("github.com/")[1]
    out = subprocess.run(["gh", "api", f"repos/{slug}/contents?ref={r['commit']}", "--jq", "[.[].name]"],
                         capture_output=True, text=True).stdout
    try:
        names = set(json.loads(out or "[]"))
    except ValueError:
        names = set()
    return bool(names & {"setup.py", "pyproject.toml"})


def add_subjects(rows) -> list[str]:
    subs = json.loads((ROOT / "subjects.json").read_text())
    have = {s["name"] for s in subs}
    added = []
    for row in rows:
        r, w = row["reg"], row["walk"]
        name = subject_name(r["name"])
        if name in have:
            continue
        targets = [t for t in r["targets"] if t not in (w.get("ignored_tests") or [])]
        # The screen's recorded repairs, so AURA runs in the environment it accepted:
        # each missing module at the newest release on the project's commit date.
        # The screen's environment, exactly: pytest 7.1.3 when the project pins
        # none, the project itself installed when it has a setup.py/pyproject.toml.
        setup = ['{python} -c "import pytest" 2>/dev/null || {python} -m pip install "pytest==7.1.3"']
        for rep in w.get("repairs") or []:
            pkg = rep.rsplit(" ", 1)[-1]
            ver = index().newest_at(pkg, r["commit_date"], python_version=w.get("python") or r["python_version"])
            setup.append(f'{{python}} -m pip install "{pkg}=={ver}"' if ver else f"{{python}} -m pip install {pkg}")
        subs.append({
            "name": name, "url": r["url"], "commit": r["commit"], "commit_date": r["commit_date"],
            "domain": "ft-test", "is_ml_research": False,
            "python_version": w.get("python") or r["python_version"],
            "requirements_file": r.get("reqs", "requirements.txt"),
            "is_installable": installable(r),
            "validation": {"type": "pytest", "pytest_target": " ".join(targets)},
            "validation_provenance": "project",
            "validation_notes": (
                f"Fine-tuning test candidate ({r.get('family')} family), eligible under "
                f"docs/FT_TEST_SELECTION.md: baseline passes in the walk check "
                f"(passed/failed {w.get('baseline')}); a newer passing joint set exists "
                f"({row['joint']}) beyond the walk ({w.get('walk')}). Chosen by structure, "
                "before any LLM run."),
            "screened_date": "2026-10-09",
            "environment_setup": setup,
        })
        added.append(name)
    (ROOT / "subjects.json").write_text(json.dumps(subs, indent=2, ensure_ascii=False) + "\n")
    return added


def frozen_requirements(subject: str):
    """(exact pins, declared specifiers) from the subject's frozen baseline.

    A line may pin and cap at once ("numpy==1.24.3, <2.0.0"); the cap is the
    project's own decision and bounds what any upgrade may reach.
    """
    from packaging.requirements import Requirement
    base = ROOT / "results" / subject / f"{subject}.baseline.txt"
    if not base.exists():
        return None, None
    pins, specs = {}, {}
    for line in base.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            q = Requirement(line)
        except Exception:
            continue
        n = canon(q.name)
        specs[n] = q.specifier
        exact = [sp.version for sp in q.specifier if sp.operator == "=="]
        if exact:
            pins[n] = exact[0]
    return pins, specs


def frozen_pins(subject: str) -> dict | None:
    return frozen_requirements(subject)[0]


def coupled_by_metadata(subject: str, py: str) -> bool | None:
    """The main study's definition, on the subject's frozen baseline."""
    from aura.index import PackageIndex
    from packaging.requirements import Requirement
    pins = frozen_pins(subject)
    if pins is None:
        return None
    env = {"python_version": py, "python_full_version": py + ".0", "sys_platform": "linux",
           "platform_system": "Linux", "os_name": "posix", "platform_machine": "x86_64",
           "implementation_name": "cpython", "platform_python_implementation": "CPython", "extra": ""}
    idx = PackageIndex()
    for p, v in pins.items():
        h = idx.history(p)
        if not h or not h.versions:
            continue
        cands = h.compatible(h.between(v, h.versions[-1]), py)[0]
        if not cands:
            continue
        for rq in idx.requires_dist(p, max(cands, key=V)) or []:
            try:
                q = Requirement(rq)
            except Exception:
                continue
            if q.marker and not q.marker.evaluate(env):
                continue
            n = canon(q.name)
            if n in pins and n != p and q.specifier and not q.specifier.contains(pins[n], prereleases=True):
                return True
    return False


def rank():
    rows, _ = eligible()
    ranked, notes = [], []
    for row in rows:
        r, J = row["reg"], row["joint"]
        name = subject_name(r["name"])
        res = ROOT / "results" / name / f"{name}__aura_linear_greedy.result.json"
        if not res.exists():
            notes.append(f"{name}: greedy not run yet")
            continue
        g = json.loads(res.read_text())
        if g.get("outcome") == "failed":
            notes.append(f"{name}: greedy run failed ({g.get('reason')})")
            continue
        coupled = coupled_by_metadata(name, row["walk"].get("python") or r["python_version"])
        if not coupled:
            notes.append(f"{name}: not coupled by metadata" if coupled is False else f"{name}: no frozen baseline")
            continue
        upd = {canon(k): v for k, v in (g.get("updated") or {}).items()}
        pins, specs = frozen_requirements(name)
        pins, specs = pins or {}, specs or {}
        # The joint set counts only where the project's own requirements allow it:
        # a version above a declared cap ("numpy<2") is not an upgrade anyone may make.
        jc = {canon(k): v for k, v in J.items()}
        capped = sorted(c for c in jc if c in specs and specs[c]
                        and not specs[c].contains(V(jc[c]).base_version if V(jc[c]) else jc[c], prereleases=True)
                        and any(sp.operator in ("<", "<=", "~=", "!=") for sp in specs[c]))
        jc = {c: v for c, v in jc.items() if c not in capped}
        # Where greedy left each coupled package: its new version if it moved it,
        # else the frozen baseline's.
        final = lambda c: upd[c][1] if c in upd else pins.get(c, "0")
        behind = sorted(c for c in jc if V(final(c)) < V(jc[c]))
        if not behind:
            notes.append(f"{name}: greedy reaches the joint set")
            continue
        majors = sum(1 for c in behind if V(jc[c]).major > V(final(c)).major)
        stuck = not any(c in jc for c in upd)
        ranked.append({"subject": name, "family": r.get("family"), "behind": behind,
                       "n_behind": len(behind), "major_gaps": majors, "greedy_fully_stuck": stuck,
                       "joint": jc, "greedy": {c: final(c) for c in behind}})
    ranked.sort(key=lambda x: (-x["n_behind"], -x["major_gaps"], x["subject"]))
    chosen, per = [], {}
    for x in ranked:
        if len(chosen) == TAKE:
            break
        if per.get(x["family"], 0) >= PER_FAMILY:
            continue
        per[x["family"]] = per.get(x["family"], 0) + 1
        chosen.append(x)
    out = ROOT / "screening/ft_test_selection.json"
    out.write_text(json.dumps({"chosen": chosen, "ranked": ranked, "notes": notes}, indent=1))
    print(f"qualifying {len(ranked)}; chosen {len(chosen)} (written to {out.relative_to(ROOT)})")
    for x in chosen:
        print(f"  {x['subject']:<42} {x['family']:<12} behind {x['n_behind']} (major {x['major_gaps']})"
              f"{'  greedy fully stuck' if x['greedy_fully_stuck'] else ''}")
    for n in notes:
        print("  -", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["eligible", "rank"])
    ap.add_argument("--add", action="store_true", help="eligible: add them to subjects.json")
    args = ap.parse_args()
    if args.stage == "eligible":
        rows, reasons = eligible()
        screened = len({p.stem for d in WALKS for p in d.glob("w-*.json")})
        print(f"screened {screened}; eligible {len(rows)}; excluded {reasons}")
        fams = {}
        for row in rows:
            fams[row["reg"].get("family")] = fams.get(row["reg"].get("family"), 0) + 1
        print("eligible by family:", dict(sorted(fams.items(), key=lambda kv: -kv[1])))
        if args.add:
            added = add_subjects(rows)
            print(f"added {len(added)} to subjects.json:", " ".join(added))
    else:
        rank()


if __name__ == "__main__":
    main()
