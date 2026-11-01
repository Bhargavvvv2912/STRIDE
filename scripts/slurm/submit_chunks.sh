#!/usr/bin/env bash
#
# Submit one arm for many subjects in small batches, waiting for room in the
# queue before each: one array of 91 tasks was refused outright
# (AssocGrpSubmitJobsLimit, the cap on jobs a user may have queued).
#
#   ARMS=renovate_replay ./scripts/slurm/submit_chunks.sh scripts/bot_subjects.txt
#   CHUNK=6 MAX_QUEUED=12 ARMS=... ./scripts/slurm/submit_chunks.sh list.txt
#
# The list has one subject per line. A subject whose result for the arm already
# exists is left out, so the loop can be stopped and started again.

set -uo pipefail
cd "$(dirname "$0")/../.."

LIST_FILE=${1:?usage: ARMS=<arm> $0 <subjects-file>}
ARMS=${ARMS:?set ARMS}
CHUNK=${CHUNK:-8}            # subjects per submission
MAX_QUEUED=${MAX_QUEUED:-16} # wait until fewer than this many of your jobs are queued

todo=()
for s in $(grep -v '^\s*$' "$LIST_FILE"); do
  [ -f "results/$s/${s}__${ARMS}.result.json" ] || todo+=("$s")
done
echo "${#todo[@]} subject(s) to run for $ARMS, $CHUNK per submission"

for ((i = 0; i < ${#todo[@]}; i += CHUNK)); do
  batch=$(IFS=,; echo "${todo[*]:i:CHUNK}")
  while [ "$(squeue -u "$USER" -h -r | wc -l)" -ge "$MAX_QUEUED" ]; do
    sleep 60
  done
  echo; echo "== batch $((i / CHUNK + 1)): $batch"
  until SUBJECTS="$batch" ARMS="$ARMS" ./scripts/slurm/submit.sh; do
    echo "  refused; retrying in 5 min"; sleep 300
  done
done
echo; echo "all submitted; watch with: squeue -u \$USER"
