#!/usr/bin/env bash
# Collect one subject for the renovate arm; used by renovate_all.sh collect.
# Keeps a result only if the merge cascade settled: an unsettled one is a lower
# bound, so its result is removed and the next collect retries it.
s=$1
PY=${PY:-python}
out="results/$s/${s}__renovate.collect.out"
if ! "$PY" scripts/real_renovate.py collect "$s" --force > "$out" 2>&1; then
  echo "$s: not yet ($(tail -1 "$out"))"
elif "$PY" -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['settled'] else 1)" \
     "results/$s/${s}__renovate.github.json"; then
  echo "$s: collected"
else
  rm -f "results/$s/${s}__renovate.result.json"
  echo "$s: not settled after 3 h, will retry on the next collect"
fi
