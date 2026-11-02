#!/bin/bash
# Queue the full STRIDE grid for one or more subjects, one job after another
# (one GPU slot): models x agents x arms. A failed job does not stop the queue.
#
#   scripts/slurm/submit_grid.sh concept-transformer
#   scripts/slurm/submit_grid.sh koandeaire neural-backed-decision-trees
#
# Override the grid with environment variables, e.g.
#   MODELS="qwen3-coder:30b" AGENTS="codex" ARMS="stride" scripts/slurm/submit_grid.sh concept-transformer
#   AFTER=4730 scripts/slurm/submit_grid.sh koandeaire     # start after job 4730
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
mkdir -p slurm_logs

for SUBJECT in "$@"; do
  for MODEL in $MODELS; do
    for AGENT in $AGENTS; do
      for ARM in $ARMS; do
        EXTRA=(--agent "$AGENT")
        case $ARM in
          stride) EXTRA+=(--rounds "$ROUNDS") ;;
          *)      EXTRA+=(--no-triage) ;;     # the published, agent-only setting
        esac
        DEP=()
        [ -n "$PREV" ] && DEP=(--dependency="afterany:$PREV")
        PREV=$(STRIDE_MODEL="$MODEL" sbatch --parsable "${DEP[@]}" \
               --job-name="${AGENT:0:4}-${ARM:0:5}" \
               scripts/slurm/stride.sbatch "$SUBJECT" "$ARM" "${EXTRA[@]}")
        printf '%s\t%s\t%s\t%s\t%s\n' "$PREV" "$SUBJECT" "$MODEL" "$AGENT" "$ARM" \
          | tee -a slurm_logs/grid_jobs.tsv
      done
    done
  done
done
echo "last job: $PREV   (next batch: AFTER=$PREV $0 <subject>)"
