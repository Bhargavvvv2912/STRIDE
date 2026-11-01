import json, re, sys, base64, subprocess
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ".")
from aura.index import PackageIndex
from packaging.requirements import Requirement
from packaging.version import Version
from packaging.utils import canonicalize_name as cn
S = sys.argv[1]
regs = json.load(open(f"{S}/passed_regs.json"))
idx = PackageIndex()
TEST = {"pytest","pluggy","pytest-cov","coverage","pytest-asyncio","pytest-django","flake8","pycodestyle","pyflakes","mccabe","black","autopep8","sphinx","sphinx-rtd-theme","pygments","pytest-mock","mypy","pylint","isort","tox"}
def env(py): return {"python_version": py, "python_full_version": py+".0", "sys_platform":"linux","platform_system":"Linux","os_name":"posix","platform_machine":"x86_64","implementation_name":"cpython","platform_python_implementation":"CPython","extra":""}
def one(r):
    slug = "/".join(r["url"].rstrip("/").split("/")[-2:])
    try:
        raw = subprocess.run(["gh","api",f"repos/{slug}/contents/{r.get('reqs','requirements.txt')}?ref={r['commit']}","--jq",".content"],capture_output=True,text=True,timeout=60).stdout
        txt = base64.b64decode(raw).decode()
    except Exception as e: return r["name"], None, str(e)
    py = r["python_version"]; pins = {}
    for l in txt.splitlines():
        l=l.split("#")[0].strip()
        if not l or l.startswith("-"): continue
        try: q = Requirement(l)
        except Exception: continue
        if q.marker and not q.marker.evaluate(env(py)): continue
        v = idx.newest_at(q.name, r["commit_date"], python_version=py, specifier=str(q.specifier) or None)
        if v: pins[cn(q.name)] = v
    edges = []
    for p, v in pins.items():
        h = idx.history(p)
        if not h or not h.versions: continue
        c = h.compatible(h.between(v, h.versions[-1]), py)[0]
        if not c: continue
        new = max(c, key=Version)
        for rq in idx.requires_dist(p, new) or []:
            try: q = Requirement(rq)
            except Exception: continue
            if q.marker and not q.marker.evaluate(env(py)): continue
            n = cn(q.name)
            if n in pins and n != p and q.specifier and not q.specifier.contains(pins[n], prereleases=True):
                edges.append(f"{p}->{n}{q.specifier}")
    return r["name"], pins, edges
with ThreadPoolExecutor(8) as ex: out = list(ex.map(one, regs))
json.dump(out, open(f"{S}/cand_labels.json","w"), indent=1)
for n,p,e in out:
    if p is None: continue
    if not e: print("NONCOUPLED", n, len(p), sorted(p)[:8])
print(sum(1 for _,p,e in out if p is not None and not e), "non-coupled of", sum(1 for _,p,_ in out if p is not None))
