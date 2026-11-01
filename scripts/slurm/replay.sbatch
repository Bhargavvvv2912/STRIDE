#!/bin/bash
#
# Backend sensitivity, without running the whole system once per model.
#
# Every LLM escalation in the reference runs recorded its exact prompt inputs,
# so each model can be asked the identical question and the answers compared.
# Re-running the system per model would instead give each backend a different
# trajectory -- different decisions, so not a comparison of models at all -- and
# would spend nearly all of its time on installs that have nothing to do with
# the model.
#
# Run it after the arms have finished:
#   sbatch --account=$ACCOUNT --partition=$PARTITION scripts/slurm/replay.sbatch
#
# EXECUTE=1 additionally installs and tests each model's plan, which answers
# "would it have worked", at the cost of one install-and-validate per plan:
#   EXECUTE=1 sbatch ... scripts/slurm/replay.sbatch
#
# ARMS=all replays every recorded decision point, including the LLM-only arms
# (hours rather than minutes); ARMS=<arm,arm> picks specific ones.
#
#SBATCH --job-name=aura-replay
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/replay_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the AURA directory}"
source scripts/slurm/cluster_env.sh

export OLLAMA_HOST="127.0.0.1:$((11500 + SLURM_JOB_ID % 5000))"
"$OLLAMA_BIN" serve > "slurm_logs/ollama_replay_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null && break
  sleep 2
done
curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null || { echo "ollama did not start"; exit 1; }

# Every model the study compares, in the form the client expects.
BACKENDS=$(.devenv/bin/python -c "
from evaluation.arms import MODELS
print(','.join(MODELS.values()))")
echo "backends: $BACKENDS"

# Which runs' decision points to replay. The full arms are the meaningful
# subset: those are the questions asked after the deterministic search had
# already failed. ARMS=all replays every arm's, which is mostly the LLM-only
# arms and takes hours.
ARMS="${ARMS:-full}"
case "$ARMS" in
  full) ONLY=$(.devenv/bin/python -c "
from evaluation.arms import MODELS
print(','.join(f'aura_full_{m}' for m in MODELS))") ;;
  all)  ONLY="" ;;
  *)    ONLY="$ARMS" ;;
esac
echo "arms    : ${ONLY:-every arm}"

EXTRA=()
[ -n "$ONLY" ] && EXTRA+=(--arms "$ONLY")
[ -n "${EXECUTE:-}" ] && EXTRA+=(--execute --registry subjects.json --workspaces "$WORKSPACES")

# --prompt-style compact: the style the runs themselves used, so the replay
# asks the question in the form the models were actually asked.
.devenv/bin/python -m evaluation.replay \
  --results results \
  --backends "$BACKENDS" \
  --prompt-style compact \
  --out results/replay.json \
  --markdown results/replay.md \
  ${EXTRA[@]+"${EXTRA[@]}"}
STATUS=$?

echo "wrote results/replay.json and results/replay.md"
exit $STATUS
