#!/bin/bash
# Queue the full STRIDE grid for one or more subjects, one job after another
# (one GPU slot): models x agents x arms. A failed job does not stop the queue.
#
#   scripts/slurm/submit_grid.sh concept-transformer
#   scripts/slurm/submit_grid.sh koandeaire neural-backed-decision-trees
#
# Override the grid with environment variables, e.g.
#   SKIP=10 scripts/slurm/submit_grid.sh concept-transformer   # first 10 already queued
#   MODELS="qwen3-coder:30b" AGENTS="codex" ARMS="stride" scripts/slurm/submit_grid.sh concept-transformer
#   AFTER=4730 scripts/slurm/submit_grid.sh koandeaire     # start after job 4730
#
# The cluster caps how many jobs one user may have queued (about 10). When a
# submission is refused for that, the script waits and retries, so a long grid
# can be left to feed itself in the background:
#   nohup scripts/slurm/submit_grid.sh concept-transformer > slurm_logs/grid.log 2>&1 &
#
# Prints one line per job and appends it to slurm_logs/grid_jobs.tsv.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ $# -ge 1 ] || { sed -n 2,13p "$0"; exit 1; }

MODELS=${MODELS:-"qwen3-coder:30b qwen3-coder-next:latest"}
AGENTS=${AGENTS:-"mini-swe-agent codex opencode"}
ARMS=${ARMS:-"stride pin_latest all_latest"}
ROUNDS=${ROUNDS:-3}
PREV=${AFTER:-}
SKIP=${SKIP:-0}
N=0
mkdir -p slurm_logs

for SUBJECT in "$@"; do
  for MODEL in $MODELS; do
    for AGENT in $AGENTS; do
      for ARM in $ARMS; do
        N=$((N + 1))
        [ "$N" -le "$SKIP" ] && continue
        EXTRA=(--agent "$AGENT")
        case $ARM in
          stride) EXTRA+=(--rounds "$ROUNDS") ;;
          *)      EXTRA+=(--no-triage) ;;     # the published, agent-only setting
        esac
        DEP=()
        [ -n "$PREV" ] && DEP=(--dependency="afterany:$PREV")
        while true; do
          if OUT=$(STRIDE_MODEL="$MODEL" sbatch --parsable "${DEP[@]}" \
                   --job-name="${AGENT:0:4}-${ARM:0:5}" \
                   scripts/slurm/stride.sbatch "$SUBJECT" "$ARM" "${EXTRA[@]}" 2>&1); then
            break
          fi
          if grep -q "SubmitJobsLimit\|job submit limit\|Unable to contact slurm controller\|Socket timed out\|temporarily unavailable" <<<"$OUT"; then
            echo "$(date +%H:%M) queue full or controller down; retrying in ${RETRY_S:-600}s ($SUBJECT $MODEL $AGENT $ARM)"
            sleep "${RETRY_S:-600}"
            # The job we depended on may have finished and left the queue.
            if [ -n "$PREV" ] && [ -z "$(squeue -h -j "$PREV" 2>/dev/null)" ]; then DEP=(); fi
            continue
          fi
          echo "sbatch failed: $OUT" >&2; exit 1
        done
        PREV=$OUT
        printf '%s\t%s\t%s\t%s\t%s\n' "$PREV" "$SUBJECT" "$MODEL" "$AGENT" "$ARM" \
          | tee -a slurm_logs/grid_jobs.tsv
      done
    done
  done
done
echo "last job: $PREV   (next batch: AFTER=$PREV $0 <subject>)"
