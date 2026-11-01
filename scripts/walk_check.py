"""Walkability check: can the linear scan reach what a joint move reaches?

Runs with whatever interpreter invokes it (the subject's Python version), with
the repository at $WALK_SRC and scratch files under $WALK_TMP.

Bounded linear, joint and greedy probes on coupled packages. A LOCKED result
is provisional, not proof of exhaustive unreachability. Tests must pass cleanly
inside a network namespace. See docs/WALK_CHECK_V2.md for limits and deployment.

    python walk.py REQS COUPLED(comma) TARGETS(comma)
Prints one JSON line: {"baseline":..., "joint":..., "walk":..., "locked":bool}
"""
import json, os, re, subprocess, sys, urllib.request, shutil, shlex
from walk_support import ProbeLogger, clean_result, reaches_target, runnable_baseline, acceptable, conflict_names
from pathlib import Path

reqs_path, coupled, targets = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
SRC = os.environ.get("WALK_SRC", "/src")
TMP = os.environ.get("WALK_TMP", "/tmp/walk")
os.makedirs(TMP, exist_ok=True)
PIP = [sys.executable, "-m", "pip"]
# Keep relative -r includes and editable paths rooted in an untouched source copy.
REQ_ROOT = Path(TMP) / "requirements-source"
shutil.rmtree(REQ_ROOT, ignore_errors=True)
shutil.copytree(SRC, REQ_ROOT, ignore=shutil.ignore_patterns(".git", ".venv*", "venv*", "env", ".env", "__pycache__", "node_modules"), symlinks=True, ignore_dangling_symlinks=True)
REQ_FILE = REQ_ROOT / Path(reqs_path).resolve().relative_to(Path(SRC).resolve())
lines = REQ_FILE.read_text().splitlines()
LOG = ProbeLogger(Path(os.environ.get("WALK_LOG_DIR", str(Path(TMP) / "logs"))), REQ_ROOT)
command = LOG.run
# Fail closed if a network namespace cannot be created. No online test fallback.
OFFLINE = shlex.split(os.environ.get("WALK_OFFLINE_PREFIX", "unshare --user --map-root-user --net --"))
OFFLINE_OK = bool(OFFLINE) and command(OFFLINE + [sys.executable, "-c", "pass"], stage="offline-preflight").returncode == 0
if not OFFLINE_OK:
    if os.environ.get("WALK_REQUIRE_OFFLINE") == "1":
        print(json.dumps({"schema_version": 2, "error": "offline isolation unavailable", "logs": str(LOG.root)}))
        sys.exit(1)
    OFFLINE = []  # no namespace on this node: network tests were deselected at triage; recorded below
canon = lambda n: re.sub(r"[-_.]+", "-", n).lower()
pins = {}
for l in lines:
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==\s*([^\s;#]+)", l)
    if m:
        pins[canon(m.group(1))] = m.group(3)
coupled = [c for c in (canon(c) for c in coupled) if c in pins]
installable = any(Path(SRC, f).exists() for f in ("setup.py", "pyproject.toml"))


def V(v):
    from packaging.version import Version
    try:
        return Version(v)
    except Exception:
        return None


def versions(pkg, current):
    data = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=60))
    cur = V(current)
    best = {}
    from packaging.specifiers import SpecifierSet
    here = ".".join(map(str, sys.version_info[:3]))
    for v, files in data["releases"].items():
        pv = V(v)
        if not pv or pv.is_prerelease or pv.is_devrelease or not files or all(f.get("yanked") for f in files):
            continue
        rp = next((f.get("requires_python") for f in files if f.get("requires_python")), None)
        try:
            if rp and here not in SpecifierSet(rp):
                continue  # not installable on the subject's interpreter
        except Exception:
            pass
        if cur and pv <= cur:
            continue
        key = (pv.major, pv.minor)
        if key not in best or pv > V(best[key]):
            best[key] = v
    return sorted(best.values(), key=V, reverse=True)


def write(state):
    out = []
    for l in lines:
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==", l)
        if m and canon(m.group(1)) in state:
            out.append(re.sub(r"(==\s*)[^\s;#]+", lambda match: match.group(1) + state[canon(m.group(1))], l, count=1))
        else:
            out.append(l)
    REQ_FILE.write_text("\n".join(out) + "\n")


def counts(out):
    p = re.search(r"(\d+) passed", out)
    f = re.search(r"(\d+) failed", out)
    e = re.search(r"(\d+) errors?", out)
    return (int(p.group(1)) if p else 0, (int(f.group(1)) if f else 0) + (int(e.group(1)) if e else 0))


BASE = None


def run(state, force=False):
    global LAST_ERR, LAST_OUT, LAST_COUNTS, LAST_STAGE
    LAST_ERR = LAST_OUT = ""
    LAST_COUNTS = None
    LAST_STAGE = "resolve"
    LOG.state = dict(state)
    write(state)
    if not force:
        dry = command(PIP + ["install", "-q", "--dry-run", "--ignore-installed", "-r", str(REQ_FILE)], capture_output=True, text=True)
        if dry.returncode:
            LAST_ERR = (dry.stdout + dry.stderr)[-600:]
            return None
    LAST_STAGE = "install"
    shutil.rmtree(f"{TMP}/v", ignore_errors=True)
    command([sys.executable, "-m", "venv", f"{TMP}/v"], check=True)
    pip = [f"{TMP}/v/bin/pip", "install", "-q"]
    extra = [x for x in __import__("os").environ.get("EXTRA", "").split(",") if x]
    runner = [] if "pytest" in pins else ["pytest==7.1.3"]
    if extra:
        # Repaired modules first: a requirement's own build may import them.
        command(pip + ["setuptools<70", "wheel", *extra], capture_output=True)
    r = command(pip + ["-r", str(REQ_FILE), *runner, "setuptools<70", *extra], capture_output=True, text=True)
    if r.returncode:
        # The build tools of the pins' era, without isolation -- AURA's own retry.
        command(pip + ["setuptools<58", "wheel"], capture_output=True)
        r = command(pip + ["--no-build-isolation", "-r", str(REQ_FILE), *runner, *extra],
                           capture_output=True, text=True)
        if r.returncode:
            LAST_ERR = (r.stdout + r.stderr)[-600:]
            return None
    shutil.rmtree(f"{TMP}/w", ignore_errors=True)
    shutil.copytree(SRC, f"{TMP}/w", ignore=shutil.ignore_patterns(".git", ".venv*", "venv*", "env", ".env", "__pycache__", "node_modules"), symlinks=True, ignore_dangling_symlinks=True)
    if installable:
        LAST_STAGE = "project-install"
        local = command(pip + ["--no-deps", f"{TMP}/w"], capture_output=True)
        if local.returncode:
            LAST_ERR = local.stdout + local.stderr
            return None
    command([f"{TMP}/v/bin/pip", "freeze", "--all"], stage="freeze")
    LAST_STAGE = "tests"
    home = Path(TMP) / "test-home"
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir()
    test_env = {**os.environ, "HOME": str(home), "XDG_CACHE_HOME": str(home / ".cache")}
    t = command(OFFLINE + [f"{TMP}/v/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets,
                           *[f"--ignore={f}" for f in IGNORED]],
                       cwd=f"{TMP}/w", env=test_env, stage="tests", capture_output=True, text=True, timeout=1800)
    LAST_OUT = t.stdout + t.stderr
    LAST_COUNTS = counts(LAST_OUT)
    return (*LAST_COUNTS, t.returncode)


def ok(res):
    return acceptable(res, BASE)


def resolvable(st):
    LOG.state = dict(st)
    write(st)
    return command(PIP + ["install", "-q", "--dry-run", "--ignore-installed", "-r", str(REQ_FILE)], capture_output=True).returncode == 0


LAST_OUT = ""
LAST_ERR = ""
LAST_COUNTS = None
LAST_STAGE = "baseline"
IGNORED = []
BASE = run({}, force=True)
# A test file that cannot even be collected aborts the whole suite. Exclude the
# files pytest reports as collection errors, once, and record them: the
# subject would list the remaining files as its targets.
if not runnable_baseline(BASE):
    bad = sorted(set(re.findall(r"^ERROR (\S+\.py)", LAST_OUT, re.M)))
    if bad and len(bad) < 10:
        IGNORED = bad
        BASE = run({}, force=True)
REPAIRS = []
# Recorded repairs: modules missing from the requirements (a test helper, or a
# package a requirement's build imports), installed as the project would have
# had them. Up to three, one after another; never the project's own modules.
MAP = {"yaml": "pyyaml", "dateutil": "python-dateutil", "bs4": "beautifulsoup4", "dotenv": "python-dotenv",
       "jwt": "pyjwt", "factory": "factory-boy", "attr": "attrs", "pytest_mock": "pytest-mock",
       "pytest_asyncio": "pytest-asyncio", "multipart": "python-multipart", "sklearn": "scikit-learn",
       "mock": "mock", "requests": "requests", "toml": "toml", "PIL": "pillow",
       "cv2": "opencv-python-headless", "rest_framework": "djangorestframework", "jose": "python-jose",
       "Crypto": "pycryptodome", "magic": "python-magic", "pytest_httpx": "pytest-httpx",
       "pytest_django": "pytest-django", "pytest_aiohttp": "pytest-aiohttp", "faker": "Faker",
       "model_bakery": "model-bakery", "asgi_lifespan": "asgi-lifespan", "pydantic_settings": "pydantic-settings",
       "aioresponses": "aioresponses", "fakeredis": "fakeredis", "freezegun": "freezegun", "respx": "respx",
       "moto": "moto", "httpx": "httpx", "dateparser": "dateparser", "psycopg2": "psycopg2-binary",
       "MySQLdb": "mysqlclient", "google": None, "tests": None, "test": None, "app": None, "src": None}


def own_module(mod):
    return (Path(SRC, mod).exists() or Path(SRC, mod + ".py").exists()
            or bool(list(Path(SRC).glob(f"*/{mod}"))) or bool(list(Path(SRC).glob(f"*/{mod}.py"))))


for _ in range(3):
    if runnable_baseline(BASE):
        break
    m = re.search(r"No module named '([A-Za-z0-9_]+)", (LAST_OUT or "") + (LAST_ERR or ""))
    if not m or own_module(m.group(1)):
        break
    mod = m.group(1)
    pkg = MAP.get(mod, mod.replace("_", "-"))
    if pkg is None or any(pkg in r for r in REPAIRS):
        break
    os.environ["EXTRA"] = ",".join(x for x in [os.environ.get("EXTRA", ""), pkg] if x)
    REPAIRS.append(f"installed test-only module {pkg}")
    BASE = run({}, force=True)
    if not runnable_baseline(BASE) and not IGNORED:
        bad = sorted(set(re.findall(r"^ERROR (\S+\.py)", LAST_OUT or "", re.M)))
        if bad and len(bad) < 10:
            IGNORED = bad
            BASE = run({}, force=True)
if not runnable_baseline(BASE):
    print(json.dumps({"schema_version": 2, "error": "baseline does not run", "baseline": LAST_COUNTS,
                      "stage": LAST_STAGE, "pytest_exit": BASE[2] if BASE else None,
                      "why": (LAST_ERR or LAST_OUT)[-600:], "ignored_tests": IGNORED,
                      "python": ".".join(map(str, sys.version_info[:2])), "logs": str(LOG.root)}))
    sys.exit(1)
cands = {c: versions(c, pins[c]) for c in coupled}
(LOG.root / "candidates.json").write_text(json.dumps(cands, indent=2))
newest = {c: (cands[c][0] if cands[c] else pins[c]) for c in coupled}
joint = run(newest)
state = {c: pins[c] for c in coupled}
for _ in range(3):
    changed = False
    for c in coupled:
        installs = 0
        for v in cands[c]:
            if V(v) <= V(state[c]):
                break
            if not resolvable({**state, c: v}):
                continue
            installs += 1
            if installs > 20:
                break
            if ok(run({**state, c: v})):
                state[c] = v
                changed = True
                break
    if not changed:
        break
# Joint moves from where the walk stalled: every pair of coupled packages, both
# moved above the walk, newest combinations first; the resolver's dry run prunes
# impossible pairs before anything installs. A passing pair beyond the walk is a
# move only a joint proposal can make.
import itertools
target, behind, tried = None, [], 0
if ok(joint) and any(V(state[c]) < V(newest[c]) for c in coupled):
    target = newest
    behind = [c for c in coupled if V(state[c]) < V(newest[c])]
def resolver_pick(free, strict=True, fixed=None):
    """pip's choice for the packages in `free` (above, or for strict=False at
    least, their current versions); `fixed` are pinned exactly; the rest stay."""
    fixed = fixed or {}
    LOG.state = {"free": sorted(free), "fixed": fixed, "walk": dict(state)}
    out = []
    for l in lines:
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==", l)
        name = canon(m.group(1)) if m else None
        cur = state.get(name, pins.get(name)) if name else None
        if m and name in fixed:
            out.append(f"{m.group(1)}{m.group(2) or ''}=={fixed[name]}")
        elif m and name in free:
            out.append(f"{m.group(1)}{m.group(2) or ''}{'>' if strict else '>='}{cur}")
        elif m:
            out.append(f"{m.group(1)}{m.group(2) or ''}=={cur}")
        else:
            out.append(l)
    REQ_FILE.write_text("\n".join(out) + "\n")
    r = command(PIP + ["install", "-q", "--dry-run", "--ignore-installed", "--report", f"{TMP}/rep.json",
                        "-r", str(REQ_FILE)], capture_output=True, text=True)
    if r.returncode:
        return None
    picked = {canon(i["metadata"]["name"]): i["metadata"]["version"] for i in json.load(open(f"{TMP}/rep.json"))["install"]}
    return {c: picked.get(c, state.get(c, pins.get(c))) for c in free}


# 1) Enumerated pairs: for each version of a (newest first), the newest partner
#    version of b the resolver accepts; up to 4 installs per ordered pair.
for a, b in itertools.permutations(coupled, 2):
    if target:
        break
    va = [v for v in cands[a] if V(v) > V(state[a])]
    vb = [v for v in cands[b] if V(v) > V(state[b])]
    runs, dry = 0, 0
    for x in va:
        partner = None
        for y in vb[:12]:
            dry += 1
            if resolvable({**state, a: x, b: y}):
                partner = y
                break
        if dry > 150:
            break
        if partner is None:
            continue
        st = {**state, a: x, b: partner}
        runs += 1
        tried += 1
        if ok(run(st)):
            target, behind = st, [a, b]
            break
        if runs >= 4:
            break
# 2) Fallback: let the resolver pick a group above the walk.
for g in list(itertools.combinations(coupled, 2)) + ([tuple(coupled)] if len(coupled) > 2 else []):
    if target:
        break
    pick = resolver_pick(set(g))
    if not pick or all(V(pick[c]) <= V(state[c]) for c in g):
        continue
    st = {**state, **pick}
    tried += 1
    if ok(run(st)):
        target, behind = st, [c for c in g if V(pick[c]) > V(state[c])]
# Greedy, as AURA's _tier_greedy: target one coupled package at its newest
# installable version; while the resolver refuses, add every pinned package its
# conflict report names, each at its newest installable version; then test.
NEWEST_CACHE = {}


def newest_of(name):
    if name in cands:
        return cands[name][0] if cands[name] else None
    if name not in NEWEST_CACHE:
        try:
            vs = versions(name, pins.get(name, "0"))
        except Exception:
            vs = []
        NEWEST_CACHE[name] = vs[0] if vs else None
    return NEWEST_CACHE[name]


def greedy_from(c):
    group = {c: cands[c][0]}
    for _ in range(8):
        st = {**pins, **state, **group}
        LOG.state = {"greedy": dict(group)}
        write(st)
        # Not -q: pip prints the "conflict is caused by" lines only without it.
        dry = command(PIP + ["install", "--dry-run", "--ignore-installed", "-r", str(REQ_FILE)],
                      capture_output=True, text=True)
        if dry.returncode == 0:
            return st if ok(run(st)) else None
        named = conflict_names(dry.stdout + dry.stderr, set(pins)) - set(group)
        added = False
        for n in named:
            v = newest_of(n)
            if v and V(v) > V(st.get(n, "0")):
                group[n] = v
                added = True
        if not added:
            return None
    return None


greedy_ok = []
if target:
    for c in coupled:
        if not cands[c] or V(cands[c][0]) <= V(state[c]):
            continue
        st = greedy_from(c)
        if st:
            greedy_ok.append(st)
            if reaches_target(st, target, V):
                break
greedy_dominates = any(reaches_target(st, target, V) for st in greedy_ok) if target else False
# A passing but incomparable greedy move could unlock a later path. Be conservative.
locked = bool(target) and not greedy_ok and not LOG.interrupted
verdict = ("joint_advantage_candidate" if locked else "greedy_reaches_target" if greedy_dominates
           else "inconclusive_interrupted_probe" if LOG.interrupted
           else "inconclusive_greedy_path" if greedy_ok else "inconclusive_within_budget")
print(json.dumps({"schema_version": 2, "verdict": verdict, "logs": str(LOG.root),
                  "offline": OFFLINE_OK, "repairs": REPAIRS, "ignored_tests": IGNORED, "python": ".".join(map(str, sys.version_info[:2])), "search_exhaustive": False, "baseline_pytest_exit": BASE[2],
                  "extra": os.environ.get("EXTRA", ""), "baseline": BASE[:2], "coupled": coupled, "newest": newest, "joint_ok": ok(joint), "walk": state,
                  "target": target, "behind": behind, "pair_runs": tried, "greedy_reaches": greedy_ok, "greedy_dominates_target": greedy_dominates, "locked": locked}))
