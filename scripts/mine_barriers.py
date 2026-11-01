import json, csv, re, sys, os
from collections import defaultdict
R = "/Users/bhargavks/Documents/AUR/AURA/results"
ARM = "aura_full_qwen14b"
projects = [r["project"] for r in csv.DictReader(open("/Users/bhargavks/Documents/AUR/AURA/docs/paper/data/rq2_latest.csv"))]
def major(v):
    m = re.match(r"(\d+)(?:\.(\d+))?", v or ""); 
    if not m: return None
    a = int(m.group(1)); return (a, int(m.group(2) or 0)) if a == 0 else (a,)
API = re.compile(r"(AttributeError|ImportError|ModuleNotFoundError|TypeError|has no attribute|cannot import|unexpected keyword|DeprecationWarning|removed)")
rows = []
for p in projects:
    res = f"{R}/{p}/{p}__{ARM}.result.json"; log = f"{R}/{p}/{p}__{ARM}.jsonl"
    if not os.path.exists(res): continue
    d = json.load(open(res)); after = d.get("version_age_after", {}).get("per_package", {})
    final = {k: v[1] for k, v in d.get("updated", {}).items()}
    hits = defaultdict(list)
    for l in open(log):
        e = json.loads(l)
        if e.get("event") != "attempt" or not e.get("install_ok") or e.get("accepted"): continue
        if not e.get("new_failures"): continue
        pk = e.get("package"); v = (e.get("plan") or {}).get(pk)
        if not v: continue
        hits[pk].append((v, len(e["new_failures"]), bool(API.search(e.get("error_excerpt") or "")), e.get("tier")))
    for pk, hs in hits.items():
        behind = after.get(pk)
        if not behind: continue  # reached latest
        cur = final.get(pk) or "?"
        cross = [h for h in hs if cur == "?" or major(h[0]) != major(cur)]
        api = any(h[2] for h in hs)
        rows.append(dict(project=p, package=pk, final=cur, behind=behind, tried=sorted({h[0] for h in hs})[-3:],
                         major_cross=bool(cross), api_signal=api, tests_broken=max(h[1] for h in hs)))
rows.sort(key=lambda r: (-r["major_cross"], -r["api_signal"], -r["behind"]))
json.dump(rows, open(sys.argv[1], "w"), indent=1)
print(len(rows), "barriers in", len({r['project'] for r in rows}), "projects;",
      sum(r["major_cross"] for r in rows), "cross a major version")
for r in rows[:40]:
    print(f"{r['project'][:28]:28} {r['package'][:18]:18} {r['final']:>10} behind={r['behind']:<4} major={int(r['major_cross'])} api={int(r['api_signal'])} broke={r['tests_broken']} tried={r['tried']}")
