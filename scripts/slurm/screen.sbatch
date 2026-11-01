#!/bin/bash
#
# Screen candidate subjects on the cluster -- the machine the experiments run
# on, which is the only machine whose verdict means anything.
#
# Screening on a laptop rejects subjects that are perfectly good: an Apple
# Silicon Mac has no wheels for the old pins this work is about (numpy 1.20 for
# Python 3.8, say), so pip compiles them from source and the baseline "fails" for
# a reason that has nothing to do with the project. On Linux x86_64 the same
# pins install from manylinux wheels.
#
# No GPU: screening installs packages and runs tests, and never calls a model.
#
#   REPOS=screening/candidates.txt sbatch ... scripts/slurm/screen.sbatch
#   REPOS="owner/one owner/two"    sbatch ... scripts/slurm/screen.sbatch
#
#SBATCH --job-name=aura-screen
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/screen_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the AURA directory}"
source scripts/slurm/cluster_env.sh

PYTHONS="${PYTHONS:-3.13,3.12,3.11,3.10,3.9,3.8}"
REPOS="${REPOS:-screening/candidates.txt}"

echo "pythons : $PYTHONS"
echo "repos   : $REPOS"

# Each version a candidate might need, installed once and reused.
for version in ${PYTHONS//,/ }; do uv python install "$version"; done

ARGS=()
# Re-screen by default: a verdict recorded on another machine (a laptop with no
# wheels for these pins) is not a verdict about the project, and skipping those
# repositories would keep the wrong answer. FORCE=0 to honour earlier records.
[ "${FORCE:-1}" = "1" ] && ARGS+=(--force)
if [ -f "$REPOS" ]; then
  ARGS+=(--repos "$REPOS")
else
  for repo in $REPOS; do ARGS+=("$repo"); done
fi

# Its own file: screening also runs on the laptop, in containers matching this
# platform, and both appending to one tracked file makes every git pull here a
# merge conflict over records neither side edited.
.devenv/bin/python scripts/screen_candidates.py \
  --out screening/screening_cluster.jsonl \
  --pythons "$PYTHONS" "${ARGS[@]}"
STATUS=$?

echo
echo "verdicts are appended to screening/screening_cluster.jsonl"
echo "keepers:"
.devenv/bin/python - <<'PYEOF'
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("screening/screening_cluster.jsonl").read_text().splitlines() if l.strip()]
latest = {r["repo"]: r for r in rows}
for r in latest.values():
    if r["verdict"] == "keep":
        print(f"  {r['repo']:<45} py{r.get('python_version')}  "
              f"{r.get('baseline', {}).get('passed', 0)} tests pass at baseline")
PYEOF
exit $STATUS
