#!/bin/bash
# Queue the full STRIDE grid for one or more subjects: models x agents x arms.
# Runs go in PARALLEL chains (default 4, the per-user running-job cap); within
# a chain jobs run one after another. A failed job does not stop the queue.
# Runs already finished (results/<subject>/<run>/summary.json) or still in the
# queue are skipped, so restarting the script is always safe.
#
#   scripts/slurm/submit_grid.sh concept-transformer
#   scripts/slurm/submit_grid.sh koandeaire neural-backed-decision-trees
#
# Override the grid with environment variables, e.g.
#   PARALLEL=2 scripts/slurm/submit_grid.sh concept-transformer  # 2 runs at once
#   SKIP=10 scripts/slurm/submit_grid.sh concept-transformer   # also skip the first 10
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
ROUNDS=${ROUNDS:-10}  # safety cap; stride stops when a round crosses nothing
PARALLEL=${PARALLEL:-4}
SKIP=${SKIP:-0}
PREVS=()
for ((i = 0; i < PARALLEL; i++)); do PREVS[i]=${AFTER:-}; done
SLOT=0
PREV=""
N=0
mkdir -p slurm_logs

for SUBJECT in "$@"; do
  for MODEL in $MODELS; do
    for AGENT in $AGENTS; do
      for ARM in $ARMS; do
        N=$((N + 1))
        [ "$N" -le "$SKIP" ] && continue
        EXTRA=(--agent "$AGENT")
        RUN="${ARM}__${AGENT}__${MODEL//:/-}"
        case $ARM in
          stride) EXTRA+=(--rounds "$ROUNDS") ;;
          *)      EXTRA+=(--no-triage); RUN+="__notriage" ;;   # the published, agent-only setting
        esac
        if grep -qs '"finished"' "results/$SUBJECT/$RUN/summary.json"; then
          echo "done already: $SUBJECT $RUN"; continue
        fi
        LAST=$(awk -F'\t' -v s="$SUBJECT" -v m="$MODEL" -v a="$AGENT" -v r="$ARM" \
               '$2==s && $3==m && $4==a && $5==r {id=$1} END {print id}' \
               slurm_logs/grid_jobs.tsv 2>/dev/null || true)
        if [ -n "$LAST" ] && [ -n "$(squeue -h -j "$LAST" 2>/dev/null)" ]; then
          echo "in queue already ($LAST): $SUBJECT $RUN"; continue
        fi
        PREV=${PREVS[SLOT]}
        DEP=()
        [ -n "$PREV" ] && DEP=(--dependency="afterany:$PREV")
        while true; do
          if OUT=$(STRIDE_MODEL="$MODEL" sbatch --parsable ${DEP[@]+"${DEP[@]}"} \
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
        OUT=${OUT##*$'\n'}; OUT=${OUT%%;*}   # job id only (last line, no cluster suffix)
        PREV=$OUT
        PREVS[SLOT]=$OUT
        SLOT=$(( (SLOT + 1) % PARALLEL ))
        printf '%s\t%s\t%s\t%s\t%s\n' "$PREV" "$SUBJECT" "$MODEL" "$AGENT" "$ARM" \
          | tee -a slurm_logs/grid_jobs.tsv
      done
    done
  done
done
echo "all queued; last jobs per chain: ${PREVS[*]}"
