#!/usr/bin/env python
"""Is a subject's run trustworthy? One line per arm, then the warning signs.

    python scripts/check_subject.py flask-basic-registration

Reads results/<subject>/ only. For each arm: outcome, packages advanced, share
at latest, installs tried, model calls, wall time, and how long its test runs
took. Then it flags what has gone wrong before:

  - an arm whose very first test run (the baseline) did not pass;
  - test runs that hit the timeout -- slicelabeler's suite hung on one run in
    ten at random, and a hang reads as a failure;
  - test durations that split into fast and timed-out with nothing between,
    which is a hang, not a slow machine;
  - arms that errored or stopped at a budget.
"""
import json
import statistics
import sys
from pathlib import Path


def arm_rows(subject: str):
    root = Path("results") / subject
    for result in sorted(root.glob(f"{subject}__*.result.json")):
        arm = result.name[len(subject) + 2:-len(".result.json")]
        if arm in ("dependabot", "renovate"):
            continue
        d = json.loads(result.read_text())
        log = root / f"{subject}__{arm}.jsonl"
        events = [json.loads(line) for line in log.open()] if log.exists() else []
        tests = [e for e in events if e.get("event") == "validation"]
        yield arm, d, events, tests


def test_timeout(subject: str) -> float:
    """The subject's own limit (aura.validation's default unless it sets one)."""
    for s in json.loads(Path("subjects.json").read_text()):
        if s["name"] == subject:
            return float((s.get("validation") or {}).get("timeout_s", 3600))
    return 3600.0


def main(subject: str) -> int:
    rows = list(arm_rows(subject))
    limit = 0.95 * test_timeout(subject)
    if not rows:
        print(f"no results for {subject} under results/{subject}/")
        return 1
    warnings = []
    print(f"{'arm':<22} {'outcome':<18} {'adv':>6} {'latest':>7} {'installs':>8} "
          f"{'llm':>4} {'wall':>7}  test runs (s): median / max")
    for arm, d, events, tests in rows:
        after = d.get("version_age_after") or {}
        n = after.get("n_packages", "?")
        counters = d.get("counters") or {}
        durations = [t.get("duration_s") or 0 for t in tests]
        timed_out = [x for x in durations if x >= limit]
        med = statistics.median(durations) if durations else 0
        print(f"{arm:<22} {d['outcome']:<18} {len(d.get('updated') or {}):>3}/{n!s:<2} "
              f"{after.get('at_latest_pct', 0) or 0:>6.1f}% "
              f"{counters.get('attempts_all', counters.get('attempts', 0)) or 0:>8} "
              f"{counters.get('llm_calls', 0) or 0:>4} {d['wall_clock_s'] / 60:>6.1f}m  "
              f"{med:.1f} / {max(durations) if durations else 0:.1f}")
        # A baseline may fail some tests on purpose (flask-huey-example's 12 are
        # tolerated by every arm); what is wrong is one that hung, or ran none.
        first = tests[0] if tests else None
        if first and arm not in ("baseline", "pip_upgrade") and not arm.startswith("llm_oneshot") \
                and not first.get("passed") \
                and ((first.get("duration_s") or 0) >= limit or not first.get("failing_n")):
            warnings.append(f"{arm}: its first test run (the baseline) hung or ran nothing "
                            f"({first.get('duration_s')} s) -- the arm never really started")
        # pip_upgrade and the one-shot arms may legitimately fail at install (a
        # proposal pip cannot resolve); every other arm must reach the tests.
        if not tests and arm not in ("pip_upgrade",) and not arm.startswith("llm_oneshot"):
            warnings.append(f"{arm}: {d['outcome']} without running a single test -- "
                            f"{(d.get('reason') or d.get('error') or '')[:110]}")
        if d["outcome"] in ("error",) or d["outcome"].startswith("truncated"):
            warnings.append(f"{arm}: {d['outcome']} -- {d.get('reason') or d.get('error', '')}"[:160])
        if timed_out:
            fast = [x for x in durations if x < limit]
            gap = (max(fast) if fast else 0) < 0.5 * min(timed_out)
            warnings.append(f"{arm}: {len(timed_out)} of {len(durations)} test runs hit the timeout"
                            + (" with nothing in between -- runs that hang, not a slow node"
                               if gap and fast else ""))
    digests = {}
    for arm, _d, events, _t in rows:
        for e in events:
            if e.get("note") == "baseline_established":
                digests.setdefault(e.get("digest"), []).append(arm)
    if len(digests) > 1:
        warnings.append("arms started from different baselines: " + "; ".join(
            f"{d}: {', '.join(sorted(a))}" for d, a in digests.items()))
    base = next((d for arm, d, _, _ in rows if arm == "baseline"), None)
    if base is not None:
        first = next((t for a, _, _, t in rows if a == "baseline"), [])
        if first:
            print(f"\nbaseline tests: passed={first[0].get('passed')}, "
                  f"failing={first[0].get('failing_n')}, {first[0].get('duration_s')} s")
    print("\n" + ("\n".join("WARNING  " + w for w in warnings) if warnings
                  else "no warning signs: every arm started from a passing baseline, "
                       "no test run hit the timeout, nothing errored"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    raise SystemExit(main(sys.argv[1]))
