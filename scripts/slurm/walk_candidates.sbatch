#!/bin/bash
#
# Screen candidates for the LLM-helps stratum on the cluster, CPU only: for
# each one, establish its baseline, simulate the linear scan on its coupled
# packages (scripts/walk_check.py), and search for a joint move beyond where
# that walk stalls. LOCKED means such a move exists: a candidate the LLM tier
# can win and greedy cannot. No GPU and no LLM.
#
#   REGISTRY=screening/walk_registry_1.json sbatch --array=0-19%4 scripts/slurm/walk_candidates.sbatch
#
# Each task takes every Nth entry. V2 verdicts land in results/walk_v2/<name>.json;
# audit with scripts/audit_walk.py --results results/walk_v2 --registry <registry>
#
#SBATCH --job-name=aura-walk
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/walk_%A_%a.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the AURA directory}"
source scripts/slurm/cluster_env.sh
export REGISTRY="${REGISTRY:-screening/walk_registry.json}"
RESULT_DIR="${WALK_RESULT_DIR:-results/walk_v2}"
mkdir -p "$RESULT_DIR"
RESULT_DIR=$(cd "$RESULT_DIR" && pwd)
TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
INDEX=${SLURM_ARRAY_TASK_ID:-0}
LOCAL="${TMPDIR:-/tmp}/aura-walk-${SLURM_ARRAY_JOB_ID:-$$}-${INDEX}"
mkdir -p "$LOCAL" results/walk
trap 'rm -rf "${LOCAL:?}"' EXIT
export PIP_CACHE_DIR="$PIP_CACHE"

.devenv/bin/python -c "
import json, os
rows = json.load(open(os.environ['REGISTRY']))
for r in rows[$INDEX::$TASKS]:
    print(r['name'], r['url'], r['commit'], r['python_version'], ','.join(r['coupled']), ','.join(r['targets']), r.get('reqs', 'requirements.txt'), r.get('commit_date', ''))
" | while read -r NAME URL COMMIT PY COUPLED TARGETS REQS DATE; do
  OUT="$RESULT_DIR/$NAME.json"
  [ -s "$OUT" ] && { echo "skip $NAME (done)"; continue; }
  echo "===== $NAME ($PY)  $(date +%H:%M)"
  W="$LOCAL/$NAME"; rm -rf "$W"; mkdir -p "$W"
  LOG_DIR="$RESULT_DIR/logs/$NAME/${SLURM_JOB_ID:-local}"
  mkdir -p "$LOG_DIR"
  { git clone -q "$URL" "$W/src" && git -C "$W/src" checkout -q "$COMMIT"; } > "$LOG_DIR/clone.log" 2>&1 || { echo '{"schema_version":2,"error":"clone failed"}' > "$OUT"; continue; }
  # Python ladder: a repo whose pins exist only for newer Pythons, or whose
  # code uses newer syntax, fails its baseline on the guessed interpreter.
  # Retry those (and only those) on 3.10 and 3.11; the result records which ran.
  LADDER="$PY"
  case "$PY" in 3.7|3.8|3.9) LADDER="$PY 3.10 3.11";; 3.10) LADDER="3.10 3.11";; esac
  REQS0="$REQS"
  for PY in $LADDER; do
    REQS="$REQS0"; rm -rf "$W/tool" "$W/tmp" "$W/src/.aura_dated_requirements.txt" "$W/src/.aura_frozen_requirements.txt"
    INTERP=$(uv python find "$PY" 2>/dev/null || { uv python install "$PY" >/dev/null 2>&1 && uv python find "$PY"; })
    [ -n "$INTERP" ] || { echo '{"error":"no interpreter"}' > "$OUT.tmp"; continue; }
    { "$INTERP" -m venv "$W/tool" && "$W/tool/bin/pip" install -q packaging "pip<24.1"; } > "$LOG_DIR/bootstrap.log" 2>&1 || { echo '{"schema_version":2,"error":"tool bootstrap failed"}' > "$OUT.tmp"; continue; }
    # Unpinned transitive dependencies drift; pin them as they were on the
    # project's commit date (the declared == pins are kept as written).
    if [ -n "$DATE" ] && uv pip compile "$W/src/$REQS" --exclude-newer "${DATE}T23:59:59Z" \
         --python-version "$PY" --quiet -o "$W/src/.aura_dated_requirements.txt" > "$LOG_DIR/compile.log" 2>&1; then
      # The environment the subject would be frozen to: each declared line pinned
      # at its commit-date version, transitive packages left to resolve (as in
      # baselines/). A fully pinned closure can pass where the subject would not.
      python3 - "$W/src/$REQS" "$W/src/.aura_dated_requirements.txt" > "$W/src/.aura_frozen_requirements.txt" <<'PYEOF'
import re, sys
canon = lambda n: re.sub(r"[-_.]+", "-", n).lower()
dated = {}
for l in open(sys.argv[2]):
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s;#]+)", l)
    if m:
        dated[canon(m.group(1))] = m.group(3)
for l in open(sys.argv[1]):
    s = l.strip()
    if not s or s.startswith("#") or s.startswith("-"):
        continue
    m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?", s)
    if not m:
        continue
    n = canon(m.group(1))
    print(f"{m.group(1)}{m.group(2) or ''}=={dated[n]}" if n in dated else s)
PYEOF
      [ -s "$W/src/.aura_frozen_requirements.txt" ] && REQS=.aura_frozen_requirements.txt
    fi
    WALK_SRC="$W/src" WALK_TMP="$W/tmp" WALK_LOG_DIR="$LOG_DIR" timeout --kill-after=30s 5400 "$W/tool/bin/python" scripts/walk_check.py \
        "$W/src/$REQS" "$COUPLED" "$TARGETS" > "$LOG_DIR/checker.stdout.log" 2> "$LOG_DIR/checker.stderr.log"
    RC=$?
    LINE=$(grep '^{' "$LOG_DIR/checker.stdout.log" | tail -1)
    if [ -n "$LINE" ]; then echo "$LINE" > "$OUT.tmp"; else
      python3 -c "import json,sys;print(json.dumps({'schema_version':2,'error':'walk timeout' if int(sys.argv[2]) in (124,137) else 'walk crashed','exit_code':int(sys.argv[2]),'tail':open(sys.argv[1]).read()[-800:]}))" "$LOG_DIR/checker.stderr.log" "$RC" > "$OUT.tmp"; fi
    grep -q '"error": "baseline does not run"' "$OUT.tmp" 2>/dev/null || break
    grep -qE "No matching distribution|Could not find a version|Requires-Python|unsupported operand type\\(s\\) for \\||SyntaxError|subprocess-exited-with-error|Failed to build" "$OUT.tmp" || break
    echo "  baseline failed on $PY with an interpreter-shaped error; trying the next Python"
  done
  mv "$OUT.tmp" "$OUT"
  # Keep the per-probe logs only where they matter (home quota).
  grep -q '"locked": true\|"error"' "$OUT" || rm -rf "$LOG_DIR"
  echo "  -> $(head -c 300 "$OUT")"
  rm -rf "$W"
done
echo "done $(date +%H:%M)"
