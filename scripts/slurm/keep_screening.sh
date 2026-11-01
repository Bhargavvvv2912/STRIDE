#!/bin/bash
# Keep a walk-check screen going past the 12-hour limit: whenever no screening
# task is queued or running and the registry still has unscreened repos, submit
# the array again (finished repos are skipped). Stops when everything is
# screened, or when a whole round screened nothing new (something is wrong).
#
#   nohup bash scripts/slurm/keep_screening.sh screening/walk_registry_ft_all.json results/walk_ft 3 \
#     > ~/keep_screening.log 2>&1 &
#
# Args: registry, result dir, number of parallel tasks (default 3).
set -u
cd "$(dirname "$0")/../.."
REG="$1"; OUT="$2"; TASKS="${3:-3}"

remaining() {
  python3 - "$REG" "$OUT" <<'EOF'
import json, os, sys
reg, out = sys.argv[1], sys.argv[2]
names = [r["name"] for r in json.load(open(reg))]
print(sum(1 for n in names if not os.path.exists(os.path.join(out, n + ".json"))))
EOF
}

last=-1
while true; do
  left=$(remaining)
  if [ "$left" -eq 0 ]; then echo "$(date) all screened"; break; fi
  if ! squeue -u "$USER" -h -n aura-walk | grep -q .; then
    if [ "$left" -eq "$last" ]; then
      echo "$(date) a full round screened nothing new ($left left); stopping -- check slurm_logs/walk_*.out"
      break
    fi
    echo "$(date) $left left; submitting $TASKS tasks"
    git pull -q || true          # picks up a longer registry if one was pushed
    REGISTRY="$REG" WALK_RESULT_DIR="$OUT" sbatch --array=0-$((TASKS - 1)) scripts/slurm/walk_candidates.sbatch
    last=$left
  fi
  sleep 600
done
