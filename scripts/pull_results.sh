#!/usr/bin/env bash
#
# Copy the cluster's run logs to this machine. Run it HERE, on the laptop --
# not inside an ssh session, where the destination path does not exist.
#
#   ./scripts/pull_results.sh
#   CLUSTER=me@host REMOTE_DIR=~/AURA ./scripts/pull_results.sh
#
# Nothing local is deleted: results collected here (the dependabot arm, which
# runs on GitHub) exist only here, so the cluster has nothing to overwrite
# them with. Everything the cluster does have is taken as the cluster has it.
# It used to keep whichever copy was newer, but the repair below rewrites the
# local files, so a result the cluster finished before the last local repair
# looked older and was silently not imported -- a stale result standing in for
# a fresh one. The repair runs again after every pull, so nothing is lost.

set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER="${CLUSTER:-bsrinid@frances.eecs.umich.edu}"
REMOTE_DIR="${REMOTE_DIR:-~/AURA}"

echo "pulling $CLUSTER:$REMOTE_DIR/{results,slurm_logs} -> $(pwd)"

# tar over ssh rather than rsync: rsync has to exist on both ends and be on a
# non-interactive PATH, which it is not on every cluster.
ssh "$CLUSTER" "cd $REMOTE_DIR && tar czf - results slurm_logs 2>/dev/null" \
  | tar xzf - -C . 2>/dev/null || true

# The cluster's copies land as the run wrote them, which undoes any correction
# already applied here -- three times now a pull has restored baselines that
# report the environment their arm rejected. Runs made before those fixes need
# the repair every time they arrive, and it is cheap and idempotent for the
# ones that do not: it recomputes from the pins each arm recorded, and a result
# already on the right basis does not move.
echo
echo "putting every arm on one baseline, denominator and date ..."
./.devenv/bin/python scripts/rebase_version_age.py || \
  echo "  (rebase failed -- results are as the cluster wrote them)"

echo
for subject in results/*/; do
  [ -d "$subject" ] || continue
  name=$(basename "$subject")
  arms=$(ls "$subject" | grep -c '\.result\.json$' || true)
  echo "  $name: $arms arm(s)"
done
echo
echo "next:"
echo "  ./.devenv/bin/python -m evaluation.cli status --results results --registry subjects.json"
