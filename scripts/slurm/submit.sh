#!/usr/bin/env bash
#
# Submit AURA runs as a Slurm job array, one (subject, arm) per task, at most 4
# at once -- the cluster's per-user limit. Arms of the same subject are
# independent (separate workspaces, separate result files), so running them side
# by side is the same experiment finished sooner.
#
#   ALL=1 ./scripts/slurm/submit.sh                   every included subject
#   SUBJECTS=mean-review-collector ./scripts/slurm/submit.sh
#   ARMS=aura_full_qwen14b ./scripts/slurm/submit.sh  one arm only
#   ARMS=oneshot ./scripts/slurm/submit.sh          the one-shot LLM arms
#   ARMS=essential+oneshot ./scripts/slurm/submit.sh  both, for a new subject
#   FRESH=1 ./scripts/slurm/submit.sh                 discard earlier results
#   ACCOUNT=... PARTITION=... ./scripts/slurm/submit.sh   if the cluster needs them

set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/slurm/cluster_env.sh
mkdir -p slurm_logs

REGISTRY="${REGISTRY:-subjects.json}"
# An empty SUBJECTS means every subject. Twice that was a typo or a fresh shell,
# not a decision, and queued a hundred runs; it has to be said out loud now.
if [ -z "${SUBJECTS:-}" ] && [ "${ALL:-}" != "1" ]; then
  echo "SUBJECTS is not set. Name the subjects (export SUBJECTS=name1,name2),"
  echo "or pass ALL=1 to submit every included subject."
  exit 1
fi
# The job reads this file when it starts, not when it is submitted, so it has to
# be unique per submission: three submit.sh calls inside one second used to share
# one file, and every array ran whichever arms the last call wrote.
LIST="slurm_logs/tasks_$(date +%Y%m%dT%H%M%S)_$$.txt"
# One line per task: "<subject> <arm>".
SUBJECTS="${SUBJECTS:-}" ARMS="${ARMS:-essential}" .devenv/bin/python -c "
import json, os, sys
sys.path.insert(0, '.')
from evaluation.arms import ALL_MODELS, CORE_ARMS, ESSENTIAL_ARMS, ONESHOT_ARMS, full_suite

wanted = [s.strip() for s in os.environ['SUBJECTS'].split(',') if s.strip()]
subjects = [s['name'] for s in json.load(open('$REGISTRY'))
            if s.get('included', True) and (not wanted or s['name'] in wanted)]
suite = {a.name: a for a in full_suite() if a.upgrade_source != 'github'}
names = os.environ['ARMS']
if names == 'essential':
    arms = [a for a in ESSENTIAL_ARMS if a in suite]
elif names == 'core':
    arms = [a for a in CORE_ARMS if a in suite]
elif names == 'oneshot':
    arms = [a for a in ONESHOT_ARMS if a in suite]
elif names == 'essential+oneshot':
    arms = [a for a in (*ESSENTIAL_ARMS, *ONESHOT_ARMS) if a in suite]
elif names == 'all':
    arms = list(suite)
else:
    arms = [a.strip() for a in names.split(',') if a.strip()]
    unknown = [a for a in arms if a not in suite]
    if unknown:
        sys.exit('unknown arm(s): ' + ', '.join(unknown))
for subject in subjects:
    for arm in arms:
        print(subject, arm)
" > "$LIST"
N=$(wc -l < "$LIST" | tr -d ' ')
[ "$N" -gt 0 ] || { echo "no runs to submit"; exit 1; }

EXTRA=()
[ -n "${ACCOUNT:-}" ] && EXTRA+=(--account="$ACCOUNT")
[ -n "${PARTITION:-}" ] && EXTRA+=(--partition="$PARTITION")

echo "submitting $N run(s), at most 4 at a time:"
sed 's/^/  /' "$LIST"
TASK_LIST="$LIST" sbatch ${EXTRA[@]+"${EXTRA[@]}"} --export=ALL \
  --array="0-$((N - 1))%4" scripts/slurm/aura_array.sbatch
echo
echo "watch:   squeue -u \$USER"
echo "logs:    slurm_logs/   (per task)   and results/<subject>/   (per arm)"
