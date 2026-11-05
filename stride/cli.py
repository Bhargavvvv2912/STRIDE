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
              pip_cache=args.pip_cache, use_triage=not args.no_triage,
              max_attempts=args.max_attempts, patience=args.patience)
    kwargs = {"aura_arm": args.aura_arm}
    if args.arm == "stride":
        kwargs.update(time_budget_h=args.time_budget_h,
                      max_tries_per_package=args.max_tries)
    with subject_environment(subject):
        summary = ARMS[args.arm](run, **kwargs)
    print(json.dumps({k: summary.get(k) for k in ("subject", "arm", "agent")}, indent=1))
    for a in summary.get("repairs", []):
        print(f"{a['label'][:55]:55} accepted={a['accepted']} reason={a['reason']} "
              f"calls={(a.get('agent') or {}).get('calls', 0)} "
              f"+{a['lines_added']}/-{a['lines_removed']}")
    for key in ("age_start", "age_end"):
        age = summary.get(key) or {}
        print(f"{key}: at_latest={age.get('at_latest_pct')}% "
              f"mean_behind={age.get('mean_releases_behind')}")
    print(f"results: {run.out}")


def cmd_check(args) -> None:
    """Behaviour checks (diff coverage, differential probe) on accepted patches."""
    import shutil
    import subprocess
    from types import SimpleNamespace

    from aura.index import PackageIndex
    from aura.reqfile import ReqFile
    from evaluation.runner import _env_spec, prepare_workspace, subject_environment
    from stride import behaviour
    from stride.repair import build_environment

    subject = _subject(args.subject, args.registry)
    # absolute: git apply and the tests run from inside the workspace
    results = Path(args.results).resolve() / subject.name
    work = Path(args.work).resolve() / "check" / subject.name
    aura_reqs = (Path(args.aura_dir) / subject.name
                 / f"{subject.name}__{args.aura_arm}.requirements.txt")
    repo_cache = Path(args.repo_cache).resolve() if args.repo_cache else None
    index = PackageIndex()
    probe = behaviour.probe_for(subject.name)

    baseline_reqs = Path(args.aura_dir) / subject.name / f"{subject.name}.baseline.txt"

    def setup(name: str, patch: Path | None = None, start: Path | None = None):
        ws = work / name
        prepare_workspace(subject, ws, repo_cache)
        # The run's start state: the baseline for the current arms, AURA's final
        # state for runs from before the revamp.
        shutil.copy(start or aura_reqs, ws / subject.requirements_file)
        if patch is not None:
            done = subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)],
                                  cwd=ws, capture_output=True, text=True)
            if done.returncode != 0:
                return ws, None, "patch did not apply: " + done.stderr[-500:]
        spec = _env_spec(subject, SimpleNamespace(installer="pip"), ws, index, args.pip_cache)
        env, error = build_environment(work / f".env_{name}", spec,
                                       ReqFile.load(ws / subject.requirements_file), ws,
                                       subject.agent_config(ws))
        return ws, env, error

    def accepted_code(summary: dict) -> bool:
        attempts = [a for r in summary.get("rounds", []) for a in r.get("attempts", [])]
        attempts += summary.get("repairs", [])
        if any(a.get("accepted") and a.get("files_changed") for a in attempts):
            return True
        return bool((summary.get("alone") or {}).get("accepted_any"))

    def start_of(summary: dict) -> Path:
        new = "repairs" in summary or "alone_attempts" in summary
        return baseline_reqs if new else aura_reqs

    runs = []
    for path in sorted(results.glob("*/summary.json")):
        if args.runs and path.parent.name not in args.runs:
            continue
        if accepted_code(json.loads(path.read_text())):
            runs.append(path.parent)
    if not runs:
        sys.exit(f"no run under {results} has an accepted code change")

    table = {}
    with subject_environment(subject):
        old_value = None
        if probe:
            ws, env, error = setup("original")
            old_value = (behaviour.run_probe(env.python, ws, probe) if env
                         else {"error": "original setup failed: " + error[-300:]})
            if env:
                env.destroy()
            print(f"probe on the original code: {str(old_value)[:200]}")
        for run in runs:
            patch = run / "patch_total.diff"
            ws, env, error = setup(run.name, patch,
                                   start_of(json.loads((run / "summary.json").read_text())))
            report = {"run": run.name, "probe": probe.name if probe else None}
            if env is None:
                report["error"] = error[-800:]
            else:
                config = subject.agent_config(ws).get("VALIDATION_CONFIG", {})
                targets = config.get("pytest_target") or "tests"
                targets = targets if isinstance(targets, list) else str(targets).split()
                changed = behaviour.changed_lines(patch.read_text(errors="replace"))
                report["diff_coverage"] = behaviour.diff_coverage(env.python, ws, targets,
                                                                  changed)
                if probe:
                    new_value = behaviour.run_probe(env.python, ws, probe)
                    report["probe_same"] = behaviour.same(old_value, new_value)
                    if not report["probe_same"]:
                        report["probe_difference"] = behaviour.first_difference(
                            old_value, new_value)
                        # None: different shape (an error, a missing value);
                        # a number: how far the closest-matching values drift.
                        report["probe_max_rel_diff"] = behaviour.max_relative_difference(
                            old_value, new_value)
                    report["probe_new"] = new_value
                env.destroy()
            report["probe_old"] = old_value
            (run / "behaviour.json").write_text(json.dumps(report, indent=2) + "\n")
            table[run.name] = report
            cov = report.get("diff_coverage") or {}
            print(f"{run.name:58} coverage={cov.get('covered_lines')}/{cov.get('changed_lines')}"
                  f"  probe_same={report.get('probe_same')}  {report.get('probe_difference', '')[:90]}"
                  f"{report.get('error', '')[:120]}")
    (results / "behaviour_summary.json").write_text(json.dumps(
        {k: {"coverage": (v.get("diff_coverage") or {}).get("coverage"),
             "changed_lines": (v.get("diff_coverage") or {}).get("changed_lines"),
             "probe_same": v.get("probe_same"),
             "probe_max_rel_diff": v.get("probe_max_rel_diff"), "error": v.get("error")}
         for k, v in table.items()}, indent=2) + "\n")


def cmd_rescore(args) -> None:
    """Add the same-package-list score to finished summaries and print a table."""
    from stride.score import add_score
    root = Path(args.results)
    for f in sorted(root.glob("*/*/summary.json")):
        summary = json.loads(f.read_text())
        if "finished" not in summary:
            print(f"{f.parent.name:62} not finished")
            continue
        add_score(summary)
        f.write_text(json.dumps(summary, indent=2) + "\n")
        sc = summary.get("score") or {}
        end = sc.get("end") or {}
        print(f"{f.parent.name:62} start={sc.get('start', {}).get('mean_releases_behind')} "
              f"end={end.get('mean_releases_behind')} n={end.get('n_packages')} "
              f"removed={','.join(sc.get('removed') or []) or '-'}")


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
    r.add_argument("--arm", default="stride",
                   choices=["stride", "agent_alone", "dependabot_agent", "all_latest"])
    r.add_argument("--agent", default="mini-swe-agent",
                   choices=["mini-swe-agent", "codex", "opencode"])
    r.add_argument("--model", default="ollama_chat/qwen3-coder:30b")
    r.add_argument("--step-limit", type=int, default=60)
    r.add_argument("--agent-minutes", type=int, default=60)
    r.add_argument("--no-triage", action="store_true",
                   help="skip dependency triage before the agent")
    r.add_argument("--max-attempts", type=int, default=5,
                   help="agent calls per repair, each building on the last (every arm)")
    r.add_argument("--patience", type=int, default=2,
                   help="stop retrying after this many calls without fewer failing tests")
    r.add_argument("--max-tries", type=int, default=6,
                   help="stride: versions the agent may try per package")
    r.add_argument("--time-budget-h", type=float, default=10.0,
                   help="stride: the search's time budget")
    r.add_argument("--out", default=str(ROOT / "results"))
    r.add_argument("--work", default=str(ROOT / "workspaces"))
    r.add_argument("--repo-cache", default="")
    r.add_argument("--pip-cache", default=None)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("check", help="behaviour checks on a subject's accepted patches")
    c.add_argument("subject")
    c.add_argument("--results", default=str(ROOT / "results"))
    c.add_argument("--runs", nargs="*", default=[], help="only these run folders")
    c.add_argument("--work", default=str(ROOT / "workspaces"))
    c.add_argument("--repo-cache", default="")
    c.add_argument("--pip-cache", default=None)
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("rescore", help="score finished runs over the same package list")
    s.add_argument("--results", default=str(ROOT / "results"))
    s.set_defaults(func=cmd_rescore)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
