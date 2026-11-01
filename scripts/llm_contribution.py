#!/usr/bin/env python
"""What did the LLM tier contribute to an arm? Accepted moves, by tier.

    python scripts/llm_contribution.py flask-basic-registration
    python scripts/llm_contribution.py flask-basic-registration aura_full_qwen14b

For each AURA arm: how many probes each tier ran and how many it got accepted,
and, for every move the LLM tier got accepted, the packages it changed. A full
arm whose LLM tier was called but never accepted anything ended where the
deterministic tiers alone would have; one with accepted LLM moves owes those
packages to the model. Whether that beats greedy is then read off greedy's own
result, since greedy may reach the same versions another way.
"""
import json
import sys
from collections import Counter
from pathlib import Path


def report(subject: str, arm: str) -> None:
    log = Path("results") / subject / f"{subject}__{arm}.jsonl"
    if not log.exists():
        print(f"{arm}: no log")
        return
    events = [json.loads(line) for line in log.open()]
    attempts = [e for e in events if e.get("event") == "attempt"
                and e.get("tier") not in ("baseline", None)]
    tried, kept = Counter(), Counter()
    for e in attempts:
        tried[e["tier"]] += 1
        kept[e["tier"]] += bool(e.get("accepted"))
    calls = sum(e.get("event") == "llm_call" for e in events)
    tiers = ", ".join(f"{t} {kept[t]}/{tried[t]}" for t in sorted(tried))
    print(f"\n{arm}: {calls} LLM calls; accepted/tried by tier: {tiers}")
    before = {}
    for e in events:
        if e.get("event") == "attempt" and e.get("accepted"):
            plan = e.get("plan") or {}
            if e.get("tier") == "llm":
                moved = {k: f"{before.get(k, '?')} -> {v}" for k, v in plan.items()
                         if before.get(k) != v}
                print(f"   LLM move accepted ({e.get('package')}): {moved}")
            before.update(plan)


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    subject = sys.argv[1]
    arms = sys.argv[2:] or sorted(
        p.name[len(subject) + 2:-len(".jsonl")]
        for p in (Path("results") / subject).glob(f"{subject}__aura_*.jsonl"))
    for arm in arms:
        report(subject, arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
