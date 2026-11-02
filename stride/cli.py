"""STRIDE command line.

    python -m stride.cli barriers concept-transformer
    python -m stride.cli run concept-transformer --arm stride
    python -m stride.cli run concept-transformer --arm pin_latest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _subject(name: str, registry: str):
    from evaluation.subjects import Registry
    for s in Registry.load(registry).subjects:
        if s.name == name:
            return s
    sys.exit(f"subject {name!r} not in {registry}")


def cmd_barriers(args) -> None:
    from stride.barriers import barriers_from_run
    base = Path(args.aura_dir) / args.subject / f"{args.subject}__{args.aura_arm}"
    found = barriers_from_run(base.with_suffix(".jsonl"), base.with_suffix(".result.json"),
                              str(base) + ".requirements.txt")
    for b in found:
        print(f"{b.label():45} behind={b.releases_behind:<4} major={int(b.crosses_major)} "
              f"new_failures={len(b.new_failures)} plan={b.plan}")
    if not found:
        print("no barriers")


def cmd_run(args) -> None:
    from evaluation.runner import subject_environment
    from stride import coders
    from stride.loop import ARMS, Run

    subject = _subject(args.subject, args.registry)
    agent = coders.build(args.agent, model=args.model, step_limit=args.step_limit,
                         wall_time_s=args.agent_minutes * 60)
    run = Run(subject, args.arm, Path(args.out), Path(args.work), Path(args.aura_dir),
              agent, repo_cache=Path(args.repo_cache) if args.repo_cache else None,
              pip_cache=args.pip_cache, use_triage=not args.no_triage)
    kwargs = {"aura_arm": args.aura_arm}
    if args.arm == "pin_latest":
        kwargs["first"] = args.first
    if args.arm == "stride":
        kwargs.update(max_rounds=args.rounds, barriers_per_round=args.barriers_per_round,
                      step=args.step, first=args.first, skip_aura=args.skip_aura,
                      rerun=args.rerun)
    with subject_environment(subject):
        summary = ARMS[args.arm](run, **kwargs)
    print(json.dumps({k: summary.get(k) for k in ("subject", "arm", "agent")}, indent=1))
    for r in summary.get("rounds", []):
        for a in r.get("attempts", []):
            print(f"round {r['round']}: {a['label']:45} accepted={a['accepted']} "
                  f"reason={a['reason']} files={a['files_changed']} "
                  f"+{a['lines_added']}/-{a['lines_removed']}")
    for key in ("age_start", "age_end"):
        age = summary.get(key) or {}
        print(f"{key}: at_latest={age.get('at_latest_pct')}% "
              f"mean_behind={age.get('mean_releases_behind')}")
    print(f"results: {run.out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="stride")
    p.add_argument("--registry", default=str(ROOT / "subjects.json"))
    p.add_argument("--aura-dir", default=str(ROOT / "pilot" / "aura_results"))
    p.add_argument("--aura-arm", default="aura_full_qwen14b")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("barriers", help="list code barriers left by an AURA run")
    b.add_argument("subject")
    b.set_defaults(func=cmd_barriers)

    r = sub.add_parser("run", help="run one arm on one subject")
    r.add_argument("subject")
    r.add_argument("--arm", default="stride", choices=["stride", "pin_latest", "all_latest"])
    r.add_argument("--agent", default="mini-swe-agent")
    r.add_argument("--model", default="ollama_chat/qwen2.5-coder:14b")
    r.add_argument("--step-limit", type=int, default=60)
    r.add_argument("--agent-minutes", type=int, default=60)
    r.add_argument("--no-triage", action="store_true",
                   help="skip dependency triage: every failing step goes to the agent "
                        "(the DepBench-style baseline)")
    r.add_argument("--rounds", type=int, default=4)
    r.add_argument("--barriers-per-round", type=int, default=3)
    r.add_argument("--step", default="smallest", choices=["smallest", "aura_plan"],
                   help="jump size across a barrier (stride arm)")
    r.add_argument("--first", default="", help="package whose barrier to try first")
    r.add_argument("--rerun", default="scoped", choices=["scoped", "full"],
                   help="after a code change, search only affected packages (scoped) or all")
    r.add_argument("--skip-aura", action="store_true",
                   help="stop after the first code change (smoke test)")
    r.add_argument("--out", default=str(ROOT / "results"))
    r.add_argument("--work", default=str(ROOT / "workspaces"))
    r.add_argument("--repo-cache", default="")
    r.add_argument("--pip-cache", default=None)
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
