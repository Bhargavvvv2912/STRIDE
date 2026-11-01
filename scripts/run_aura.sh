#!/usr/bin/env bash
#
# Run AURA's evaluation arms on one subject or the whole registry.
#
#   ./scripts/run_aura.sh                                  every included subject, core arms
#   ./scripts/run_aura.sh mean-review-collector                      one subject, core arms
#   ./scripts/run_aura.sh mean-review-collector aura_full_qwen14b one subject, one arm
#   ./scripts/run_aura.sh all aura_full,aura_linear        every subject, chosen arms
#   FRESH=1 ./scripts/run_aura.sh mean-review-collector essential     delete that subject's results
#                                                          first and run every arm again
#
# Safe to interrupt with Ctrl-C and re-run: completed arms are skipped, so a
# campaign can be spread over days. Nothing is lost on a kill -9 either -- the
# event log is flushed after every event.

set -uo pipefail
cd "$(dirname "$0")/.."

SUBJECT="${1:-all}"
ARMS="${2:-core}"
REGISTRY="${REGISTRY:-subjects.json}"
RESULTS="${RESULTS:-results}"
PIP_CACHE="${PIP_CACHE:-$HOME/.aura-wheels}"
TIMEOUT="${TIMEOUT:-21600}"          # 6h per arm, matching the paper's budget
MAX_PASSES="${MAX_PASSES:-20}"        # ceiling on re-scan passes
FRESH="${FRESH:-0}"                  # 1 = start this subject from scratch
WORKSPACES="${WORKSPACES:-.workspaces}"  # per-arm checkouts and venvs (large)
REPO_CACHE="${REPO_CACHE:-.repo_cache}"  # clones of the subjects
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_HOST
PY="./.devenv/bin/python"

red()  { printf '\033[31m%s\033[0m\n' "$*"; }
grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
ylw()  { printf '\033[33m%s\033[0m\n' "$*"; }
die()  { red "FAILED: $*"; exit 1; }

echo "=============================================================="
echo " AURA evaluation"
echo "=============================================================="
echo "  subject(s) : $SUBJECT"
echo "  arms       : $ARMS"
echo "  registry   : $REGISTRY"
echo "  results    : $RESULTS"
echo "  timeout    : ${TIMEOUT}s per arm"
echo "  max passes : $MAX_PASSES"
echo "  fresh start: $([ "$FRESH" = 1 ] && echo yes || echo "no (finished arms are skipped)")"
echo

# ---------------------------------------------------------------- preflight
# Checked up front because each of these fails hours in otherwise, and two of
# them fail *silently* in ways that invalidate results rather than stopping.

[ -x "$PY" ] || die "no interpreter at $PY (create the dev venv first)"
[ -f "$REGISTRY" ] || die "no registry at $REGISTRY"

# The protocol audits the *sample*. A screening registry is a list of
# candidates chosen by family -- all ML when the family is torch -- so it is
# not held to the sample's mix; screen_candidates.sbatch sets SCREENING=1.
SKIP_AUDIT=()
if [ "${SCREENING:-0}" = "1" ]; then
  SKIP_AUDIT=(--skip-audit)    # evaluation.cli run audits again on its own
  ylw "  [--] screening registry: sample-protocol audit skipped"
else
  $PY -m evaluation.cli audit --registry "$REGISTRY" > /tmp/aura_audit.txt 2>&1 \
    || { cat /tmp/aura_audit.txt; die "the sample protocol has unresolved violations"; }
  grn "  [ok] registry audits clean"
fi

# Every subject being run must have its declared interpreter installed. Running one on a
# different Python measures the interpreter, not the dependency change.
MISSING=$($PY - "$REGISTRY" "$SUBJECT" <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
from aura.environment import find_interpreter
missing = []
only = None if sys.argv[2] == "all" else set(sys.argv[2].split(","))
for s in json.load(open(sys.argv[1])):
    if not s.get("included", True) or (only and s["name"] not in only):
        continue
    v = s.get("python_version", "3.11")
    if find_interpreter(v) is None:
        missing.append(f"{s['name']} needs Python {v}")
print("\n".join(missing))
PYEOF
)
if [ -n "$MISSING" ]; then
  red "  [!!] missing interpreters:"
  echo "$MISSING" | sed 's/^/       /'
  die "install them with pyenv (e.g. pyenv install 3.10.18)"
fi
grn "  [ok] every subject's interpreter is installed"

# An LLM arm without a reachable model wastes the whole run.
if echo "$ARMS" | grep -qE "llm|aura_full|essential|core|all"; then
  if curl -s -m 3 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1; then
    grn "  [ok] ollama is running"
  else
    ylw "  [--] ollama is not running; starting it"
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    for _ in $(seq 1 20); do
      sleep 1
      curl -s -m 2 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 && break
    done
    curl -s -m 3 http://$OLLAMA_HOST/api/tags >/dev/null 2>&1 \
      && grn "  [ok] ollama started" \
      || ylw "  [--] ollama still unreachable; LLM-tier arms will fail"
  fi
  # Every LLM arm runs once per model, so each one must already be pulled.
  NEEDED=$($PY -c "from evaluation.arms import MODELS; print(' '.join(s.split(':',1)[1] for s in MODELS.values()))")
  HAVE=$(curl -s -m 3 http://$OLLAMA_HOST/api/tags 2>/dev/null)
  for model in $NEEDED; do
    if echo "$HAVE" | grep -q "\"$model\""; then
      grn "  [ok] model $model is available"
    else
      die "model $model is not pulled (run: ollama pull $model)"
    fi
  done
fi

mkdir -p "$PIP_CACHE"
grn "  [ok] wheel cache at $PIP_CACHE"
echo

# ------------------------------------------------------------------- run
RESUME=(--resume)
if [ "$FRESH" = 1 ]; then
  [ "$SUBJECT" = "all" ] && die "FRESH=1 needs a subject name; it will not wipe every subject's results"
  case "$ARMS" in
    essential|core|all|*,*)
      # Every arm of this subject, except the dependabot arm: that one is
      # collected from GitHub and this harness cannot recreate it. Arms of one
      # subject can also run as separate jobs, so this must not be used while
      # any of them is running.
      find "$RESULTS/$SUBJECT" -maxdepth 1 -type f ! -name '*__dependabot.*' \
        -delete 2>/dev/null || true
      rm -rf "${WORKSPACES:?}/$SUBJECT"
      ylw "  [--] removed previous results for $SUBJECT (kept the dependabot arm)"
      ;;
    *)
      # One arm: remove only its files, so parallel jobs on other arms of the
      # same subject are untouched.
      rm -f "$RESULTS/$SUBJECT/${SUBJECT}__${ARMS}."*
      rm -rf "${WORKSPACES:?}/$SUBJECT/$ARMS"
      ylw "  [--] removed previous results for $SUBJECT/$ARMS"
      ;;
  esac
  RESUME=()
fi

SUBJECT_ARG=()
[ "$SUBJECT" != "all" ] && SUBJECT_ARG=(--subjects "$SUBJECT")

STAMP=$(date +%Y%m%dT%H%M%S)
CONSOLE="$RESULTS/run_${STAMP}.out"
mkdir -p "$RESULTS"
echo "Console output is also being written to $CONSOLE"
echo "Interrupt with Ctrl-C at any time; re-run this script to continue."
echo

START=$(date +%s)
$PY -m evaluation.cli run \
    --registry "$REGISTRY" \
    --arms "$ARMS" \
    ${SUBJECT_ARG[@]+"${SUBJECT_ARG[@]}"} \
    --results "$RESULTS" \
    --workspaces "$WORKSPACES" \
    --repo-cache "$REPO_CACHE" \
    --pip-cache "$PIP_CACHE" \
    --timeout "$TIMEOUT" \
    --max-passes "$MAX_PASSES" \
    ${RESUME[@]+"${RESUME[@]}"} \
    ${SKIP_AUDIT[@]+"${SKIP_AUDIT[@]}"} 2>&1 | tee "$CONSOLE"
STATUS=${PIPESTATUS[0]}
ELAPSED=$(( $(date +%s) - START ))

echo
echo "=============================================================="
printf ' finished in %dh %dm %ds (exit %s)\n' \
       $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)) "$STATUS"
echo "=============================================================="
$PY -m evaluation.cli status --results "$RESULTS" --registry "$REGISTRY" --arms "$ARMS"
echo
echo "Next:"
echo "  $PY -m evaluation.cli status --results $RESULTS --registry $REGISTRY"
echo "  $PY -m evaluation.cli report --results $RESULTS --out docs/METRICS.md"
exit "$STATUS"
