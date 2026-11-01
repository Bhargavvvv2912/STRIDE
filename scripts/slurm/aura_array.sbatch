#!/bin/bash
#
# One (subject, arm) per array task. Submit through scripts/slurm/submit.sh,
# which writes the task list, sets the array size, and caps it at 4 concurrent
# tasks (the cluster's per-user limit).
#
# Resources: installs and test suites are CPU work, so CPUs and memory matter
# more than the GPU; the one GPU serves the local LLM. AURA sizes its parallel
# probes from this allocation, not from the node's totals.
#
#SBATCH --job-name=aura
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/aura_%A_%a.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the AURA directory}"
source scripts/slurm/cluster_env.sh

# One thread per numerical library, in every process this task starts. Left
# unset, numpy, scikit-learn and the rest start one thread per core they can
# see -- 256 on these nodes -- while Slurm gives the task 12, and AURA runs
# several test processes at once. On lung-cancer-detection, whose tests train
# a model, the suite went from 10 seconds to over an hour when its arms shared
# a node, and three of four one-shot baselines timed out. The tests measure
# whether an upgrade works, not how fast, so a single thread costs nothing and
# keeps one task from starving the others.
for var in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
           NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS; do
  export "$var=1"
done

TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASK_LIST:?set by submit.sh}")
[ -n "$TASK" ] || { echo "no task for array index $SLURM_ARRAY_TASK_ID"; exit 1; }
SUBJECT=${TASK%% *}
ARM=${TASK##* }

echo "subject   : $SUBJECT"
echo "arm       : $ARM"
echo "node      : $(hostname)   cpus: ${SLURM_CPUS_PER_TASK:-?}   mem: ${SLURM_MEM_PER_NODE:-?} MB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/gpu       : /'

# Home is capped at 50 GB, with ~41 GB taken by models and tools. The shared
# wheel cache is the part that grows, so it is cleared past 6 GB -- but only
# when no other AURA task is running, since deleting a cache another task is
# reading from mid-install would break that task.
CACHE_MB=$(du -sm "$PIP_CACHE" 2>/dev/null | cut -f1)
OTHERS=$(squeue -u "$USER" -h -t R -n aura -o %i 2>/dev/null | grep -vc "^${SLURM_JOB_ID}$" || true)
if [ "${CACHE_MB:-0}" -gt 6000 ] && [ "${OTHERS:-1}" -eq 0 ]; then
  echo "wheel cache is ${CACHE_MB} MB and no other AURA task is running: clearing it"
  rm -rf "${PIP_CACHE:?}"/*
fi
echo "home use: $(du -sh "$HOME" 2>/dev/null | cut -f1) of 50G"

# Throwaway environments and the pip cache on this node's own disk, not on the
# shared filesystem. A pip install writes thousands of small files, AURA runs
# up to a dozen probes at once, and each probe is a fresh environment: on NFS
# that stalled. With one htb-api arm installing, every other arm of it -- on
# either node -- sat in "D nfs_wait_bit_killable" inside ensurepip until its
# 600-second limit, while the same environment took 4 s on local disk and 8 s
# on NFS when nothing else was running. Results and the interpreter
# stay shared; only what is written in bulk and thrown away moves.
LOCAL_DISK="${TMPDIR:-/tmp}"
export AURA_VENV_ROOT="$LOCAL_DISK/aura-venvs-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
# The arm's checkout too, where the tests run. slicelabeler's suite uses
# pyfakefs and spun at full CPU for the whole hour-long test timeout when run
# from its checkout on NFS, every probe, while the same suite and venv took
# 7 s from /tmp on the same node. A checkout is rebuilt per task anyway.
export WORKSPACES="$LOCAL_DISK/aura-work-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
# Kept between jobs on the same node, so a node downloads a wheel once.
export PIP_CACHE="$LOCAL_DISK/aura-pip-cache-$USER"
# FRESH_PIP_CACHE=1: a cache of this task's own, removed with it. A download
# cut off mid-transfer can be cached as a truncated file, and every install on
# that node then fails on it ("IncompleteRead(15598 bytes read)", the same
# count each time): three lung-cancer-detection arms lost their numpy and
# matplotlib upgrades to that, while other arms on clean caches passed them.
if [ "${FRESH_PIP_CACHE:-0}" = "1" ]; then
  export PIP_CACHE="$LOCAL_DISK/aura-pip-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
fi
mkdir -p "$AURA_VENV_ROOT" "$WORKSPACES" "$PIP_CACHE"
cleanup_local() {
  rm -rf "${AURA_VENV_ROOT:?}" "${WORKSPACES:?}"
  [ "${FRESH_PIP_CACHE:-0}" = "1" ] && rm -rf "${PIP_CACHE:?}"
}
trap cleanup_local EXIT
echo "local disk: $(df -h "$LOCAL_DISK" | awk 'NR==2{print $4" free on "$6}')"

# Each task serves Ollama on its own port: two tasks can land on one node, and
# they must not share a server (or fight over 11434). OLLAMA_HOST is read both
# by `ollama` and by AURA's client. An arm with no LLM tier needs no server.
if printf '%s' "$ARM" | grep -qE 'llm|aura_full'; then
  export OLLAMA_HOST="127.0.0.1:$((11500 + SLURM_JOB_ID % 5000))"
  # The one-shot arm sends the whole requirements file in one prompt: about 75
  # tokens a package, so 2,830 for flask-dj's 38 and some 4,400 for
  # find-that-postcode's 59 -- past the 4,096 default, where Ollama drops the
  # *front* of the prompt, the requirements themselves. 8,192 holds every
  # subject, and a window is paid for in GPU memory, so it is no larger than
  # that. (flask-dj's qwen32b one-shot call returned an HTTP 500 at 16,384; its
  # cause is unconfirmed, and it is recorded as ERROR, not as the model's
  # answer.) The other arms' prompts stay under 2,400 tokens and keep the
  # default every earlier run had.
  case "$ARM" in llm_oneshot_*) export OLLAMA_CONTEXT_LENGTH=8192 ;; esac
  "$OLLAMA_BIN" serve > "slurm_logs/ollama_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
  OLLAMA_PID=$!
  trap 'kill "$OLLAMA_PID" 2>/dev/null || true; cleanup_local' EXIT
  for _ in $(seq 1 60); do
    curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null && break
    sleep 2
  done
  curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null || { echo "ollama did not start"; exit 1; }
fi

# A task stopped by the 12-hour limit is resubmitted as itself; a finished arm
# is skipped on resubmission (FRESH=1 at submit time starts over instead).
./scripts/run_aura.sh "$SUBJECT" "$ARM"
STATUS=$?

# The subject's per-arm checkouts are working copies, not results (those are in
# results/<subject>/). Removing them keeps home under its cap; a resumed run
# re-creates what it needs.
# Only this arm's checkout: another task may be running a different arm of the
# same subject right now.
rm -rf "${WORKSPACES:?}/$SUBJECT/$ARM"
echo "home use after cleanup: $(du -sh "$HOME" 2>/dev/null | cut -f1) of 50G"
exit $STATUS
