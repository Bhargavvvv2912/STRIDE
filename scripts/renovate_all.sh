#!/usr/bin/env bash
# The renovate arm on every subject in scripts/bot_subjects.txt (the paper's 50).
#
#   scripts/renovate_all.sh setup     create each renovate-check-<subject> repo,
#                                     one after another; skips repos that exist.
#                                     Needs RENOVATE_TOKEN exported in this shell.
#   scripts/renovate_all.sh status    one line per subject: PRs, merged, CI running.
#   scripts/renovate_all.sh collect   collect every subject not yet collected,
#                                     $JOBS at a time (default 8); each waits for
#                                     its own merge cascade to settle.
#   scripts/renovate_all.sh private   make every renovate-check-* repo private
#                                     again once everything is collected.
#
# Logs: results/<subject>/<subject>__renovate.{setup,collect}.out
set -uo pipefail
cd "$(dirname "$0")/.."

SUBJECTS=$(grep -v '^\s*$' scripts/bot_subjects.txt)
USER_GH=$(gh api user --jq .login)
JOBS=${JOBS:-8}
# Collect needs the project's own dependencies (packaging, ...); the conda base
# environment has them, a bare pyenv python does not.
PY=${PY:-python}

exists() { gh repo view "$USER_GH/renovate-check-$1" --json name >/dev/null 2>&1; }

case "${1:-}" in
setup)
  if [ -z "${RENOVATE_TOKEN:-}" ]; then
    echo "RENOVATE_TOKEN is not set: read -s RENOVATE_TOKEN && export RENOVATE_TOKEN"; exit 1
  fi
  n=0; failed=()
  for s in $SUBJECTS; do
    n=$((n + 1))
    if exists "$s"; then echo "[$n/50] $s: repo exists, skipped"; continue; fi
    echo "[$n/50] $s: setting up ..."
    if "$PY" scripts/real_renovate.py setup "$s" > "results/$s/${s}__renovate.setup.out" 2>&1; then
      echo "         done"
    else
      echo "         FAILED (see results/$s/${s}__renovate.setup.out)"; failed+=("$s")
    fi
  done
  echo; echo "setup finished; failed: ${failed[*]:-none}"
  ;;
status)
  for s in $SUBJECTS; do
    if ! exists "$s"; then printf "%-26s no repo\n" "$s"; continue; fi
    collected=$([ -f "results/$s/${s}__renovate.result.json" ] && echo " [collected]" || echo "")
    printf "%-26s %s%s\n" "$s" "$("$PY" scripts/real_renovate.py status "$s" 2>&1 | tail -1)" "$collected"
  done
  ;;
collect)
  todo=()
  for s in $SUBJECTS; do
    [ -f "results/$s/${s}__renovate.result.json" ] || todo+=("$s")
  done
  echo "collecting ${#todo[@]} subject(s), $JOBS at a time"
  # --force only skips the up-front "CI still running" refusal; collect then
  # waits for CI and merges itself, up to 3 hours. A run that still did not
  # settle is a lower bound, not a result: its files are removed so the next
  # "collect" retries it, and nothing unsettled is ever kept.
  export PY; printf "%s\n" "${todo[@]}" | xargs -P "$JOBS" -I{} sh -c '
    out=results/{}/{}__renovate.collect.out
    if ! "$PY" scripts/real_renovate.py collect {} --force > "$out" 2>&1; then
      echo "{}: not yet ($(tail -1 "$out"))"
    elif python -c "import json,sys; sys.exit(0 if json.load(open(\"results/{}/{}__renovate.github.json\"))[\"settled\"] else 1)"; then
      echo "{}: collected"
    else
      rm -f results/{}/{}__renovate.result.json
      echo "{}: not settled after 3 h, will retry on the next collect"
    fi'
  left=0; for s in $SUBJECTS; do [ -f "results/$s/${s}__renovate.result.json" ] || left=$((left + 1)); done
  echo; echo "$left subject(s) still to collect${left:+ (run collect again)}"
  ;;
private)
  for s in $SUBJECTS; do
    exists "$s" && gh repo edit "$USER_GH/renovate-check-$s" --visibility private \
      --accept-visibility-change-consequences && echo "$s: private"
  done
  ;;
*)
  sed -n '2,14p' "$0"; exit 1 ;;
esac
