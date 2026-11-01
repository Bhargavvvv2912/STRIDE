"""Command line for the evaluation harness.

    python -m evaluation.cli arms                       # list experimental arms
    python -m evaluation.cli audit    --registry s.json # check the sample protocol
    python -m evaluation.cli run      --registry s.json --arms core
    python -m evaluation.cli report   --results results/ --out METRICS.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from evaluation.arms import (
    CORE_ARMS, ESSENTIAL_ARMS, FULL_ARM, MODELS, full_suite,
)
from evaluation.render import write_report
from evaluation.subjects import Registry


def _models(extra: list[str] | None) -> dict[str, str]:
    """The default open models, plus any given as --extra-model LABEL=SPEC."""
    models = dict(MODELS)
    for item in extra or []:
        label, _, spec = item.partition("=")
        if not (label and spec):
            sys.exit(f"--extra-model expects LABEL=SPEC, got {item!r}")
        models[label.strip()] = spec.strip()
    return models


GITHUB_ONLY = ("The dependabot and renovate arms run on GitHub, not here:\n"
               "  python scripts/real_dependabot.py setup <subject>\n"
               "  python scripts/real_dependabot.py collect <subject>\n"
               "  (or scripts/real_renovate.py, the same way)")


def _select_arms(names: str, models: dict[str, str] | None = None):
    suite = {a.name: a for a in full_suite(models)
             if a.upgrade_source != "github"}
    if names == "all":
        return list(suite.values())
    if names == "core":
        return [suite[n] for n in CORE_ARMS if n in suite]
    if names == "essential":
        return [suite[n] for n in ESSENTIAL_ARMS if n in suite]
    chosen = []
    for name in names.split(","):
        name = name.strip()
        if name in ("dependabot", "renovate"):
            sys.exit(GITHUB_ONLY)
        if name not in suite:
            sys.exit(f"unknown arm {name!r}. Available:\n  " + "\n  ".join(sorted(suite)))
        chosen.append(suite[name])
    return chosen


def cmd_arms(args) -> int:
    """List the arms with the settings that actually distinguish them.

    Two arms can share every tier and differ only in whether routing is on --
    which is the whole point of that pair, and impossible to see from the names
    alone. The configuration columns make each arm's identity explicit.
    """
    print(f"{'':3}{'arm':<28} {'tiers':<20} {'route':<6} {'graph':<6} "
          f"{'k':<4} {'cover':<6} group")
    print("-" * 104)
    for arm in full_suite(_models(args.extra_model)):
        marker = "E" if arm.name in ESSENTIAL_ARMS else (
            "*" if arm.name in CORE_ARMS else " ")
        if arm.kind == "aura":
            tiers = "+".join(arm.tiers) or "-"
            route = "on" if arm.routing else "off"
            graph = "on" if arm.constraint_graph else "off"
            k = "1" if arm.batch_k == 1 else ("auto" if arm.batching else "1")
            cover = "yes" if arm.measure_coverage else "-"
        else:
            tiers, route, graph, k, cover = arm.upgrade_source, "-", "-", "-", "-"
        print(f"{marker:<3}{arm.name:<28} {tiers:<20} {route:<6} {graph:<6} "
              f"{k:<4} {cover:<6} {arm.group}")

    print("\n  E = --arms essential    * = --arms core (E arms are also in core)")
    print("\n  Arms with identical tiers differ in the configuration columns:")
    print(f"    {FULL_ARM}  vs  aura_no_routing   -> routing on vs off")
    print(f"    {FULL_ARM}  vs  aura_serial       -> batch width auto vs 1")
    print(f"    {FULL_ARM}  vs  aura_no_graph     -> declared constraints or not")
    print(f"  Single-model controls use {FULL_ARM.rsplit('_', 1)[-1]}.")
    for arm in full_suite(_models(args.extra_model)):
        if arm.name in ESSENTIAL_ARMS:
            print(f"\n  {arm.name}\n    {arm.description}")
    return 0


def cmd_audit(args) -> int:
    registry = Registry.load(args.registry)
    composition = registry.composition()
    print(json.dumps(composition, indent=2))
    problems = registry.audit()
    if problems:
        print("\nPROTOCOL VIOLATIONS:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\nProtocol checks passed.")
    return 0


def _run_is_complete(log_path: Path) -> bool:
    """Did this run reach its end, or was it interrupted partway?

    ``--resume`` must skip finished work, not merely *started* work. A run
    killed by Ctrl-C, an OOM or a machine restart leaves a log that exists but
    stops mid-stream; treating that as done would leave a permanent hole in the
    campaign that reads as a completed failure. Completion means a ``run_end``
    event was written.
    """
    if not log_path.exists():
        return False
    try:
        with log_path.open(encoding="utf-8") as fh:
            return any('"event": "run_end"' in line or '"event":"run_end"' in line
                       for line in fh)
    except OSError:
        return False


def cmd_run(args) -> int:
    from evaluation.runner import log_path_for, run_arm

    registry = Registry.load(args.registry)
    subjects = registry.included
    if args.subjects:
        wanted = {s.strip() for s in args.subjects.split(",")}
        subjects = [s for s in subjects if s.name in wanted]
    arms = _select_arms(args.arms, _models(args.extra_model))

    if not args.skip_audit:
        problems = registry.audit()
        if problems:
            print("Refusing to run: the sample protocol has unresolved violations.",
                  file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            print("Re-run with --skip-audit to override (and say so in the paper).",
                  file=sys.stderr)
            return 1

    total = len(subjects) * len(arms)
    print(f"{len(subjects)} subjects x {len(arms)} arms = {total} runs")
    if args.dry_run:
        for subject in subjects:
            print(f"  {subject.name}: " + ", ".join(a.name for a in arms))
        return 0

    done = 0
    started = time.time()
    for subject in subjects:
        for arm in arms:
            done += 1
            log_path = log_path_for(Path(args.results), subject.name, arm.name)
            if args.resume and _run_is_complete(log_path):
                print(f"[{done}/{total}] skip {subject.name}/{arm.name} (done)")
                continue
            print(f"[{done}/{total}] {subject.name}/{arm.name} ...", flush=True)
            result = run_arm(
                subject, arm, results_dir=args.results, workspaces_dir=args.workspaces,
                repo_cache=args.repo_cache, timeout_s=args.timeout,
                pip_cache=args.pip_cache, max_passes=args.max_passes,
            )
            print(f"    -> {result.outcome} ({result.wall_clock_s:.0f}s, "
                  f"{len(result.updated)} pkgs)"
                  + (f" [{result.reason}]" if result.reason else ""))
            if arm.name == "baseline" and not result.passed:
                print(f"    !! {subject.name} fails the working-baseline precondition; "
                      f"skipping its remaining arms.")
                break

    cmd_digest(argparse.Namespace(results=args.results))
    print(f"\nDone in {(time.time() - started) / 60:.1f} min. "
          f"Now: python -m evaluation.cli report --results {args.results}")
    return 0


def _subject_summary(subject_dir: Path, order: dict) -> str:
    """A side-by-side table of every arm's result, for the top of the digest.

    Read from the event logs rather than recomputed, so it can never disagree
    with them. An arm still running has no ``run_end`` and is shown as such --
    "in progress" and "finished with nothing to show" are different states and
    a summary that blurs them is worse than no summary.
    """
    from evaluation.report import summarize_log

    rows = []
    for path in sorted(subject_dir.glob("*.jsonl"),
                       key=lambda p: order.get(p.stem.split("__", 1)[-1], 999)):
        run = summarize_log(path)
        if run is None:
            continue
        arm = run.arm or path.stem.split("__", 1)[-1]
        if not run.complete:
            rows.append((arm, "in progress", "-", "-", "-"))
            continue
        advanced = ", ".join(
            f"{n} {v[0]}->{v[1]}" for n, v in sorted((run.updated or {}).items())
        ) or "-"
        rows.append((arm, run.outcome, str(len(run.updated or {})),
                     f"{run.wall_clock_s:.0f}s",
                     advanced if len(advanced) <= 60 else advanced[:57] + "..."))
    if not rows:
        return ""

    widths = [max(len(r[i]) for r in [("arm", "outcome", "n", "time", "advanced")]
                  + rows) for i in range(5)]
    def line(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    out = ["", "=" * 78, "SUMMARY", "=" * 78, "",
           line(("arm", "outcome", "n", "time", "advanced")),
           line(tuple("-" * w for w in widths))]
    out += [line(r) for r in rows]
    return "\n".join(out) + "\n"


def cmd_digest(args) -> int:
    """Assemble each subject's per-arm console logs into one readable file.

    A *derived* artefact, regenerated from the per-arm logs and never written
    to during a run. The per-arm files stay the source of truth because the arm
    is the unit of work: re-running one arm after a crash rewrites exactly one
    file, ``--resume`` can tell which arms are done, and arms run weeks apart
    never interleave. Appending several arms into one file as they run would
    reintroduce exactly the merge problem that made re-runs silently double
    their attempt counts.
    """
    from evaluation.arms import full_suite

    order = {a.name: i for i, a in enumerate(full_suite())}
    results = Path(args.results)
    if not results.is_dir():
        print(f"No results directory at {results}")
        return 1

    written = 0
    for subject_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        logs = [p for p in subject_dir.glob("*.log")
                if not p.name.endswith("__ALL.log")]
        if not logs:
            continue

        def arm_of(path: Path) -> str:
            stem = path.stem
            return stem.split("__", 1)[1] if "__" in stem else stem

        logs.sort(key=lambda p: (order.get(arm_of(p), 999), arm_of(p)))
        parts = [
            f"AURA — all arms for {subject_dir.name}",
            f"assembled {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"from {len(logs)} arm log(s)",
            "This file is generated. Edit nothing here; the per-arm logs are "
            "the record.",
        ]
        parts.append(_subject_summary(subject_dir, order))
        for log in logs:
            parts.append("\n" + "#" * 78)
            parts.append(f"#  ARM: {arm_of(log)}")
            parts.append("#" * 78 + "\n")
            parts.append(log.read_text(encoding="utf-8").rstrip())

        target = subject_dir / f"{subject_dir.name}__ALL.log"
        target.write_text("\n".join(parts) + "\n", encoding="utf-8")
        print(f"{subject_dir.name}: {len(logs)} arm(s) -> {target}")
        written += 1

    if not written:
        print("Nothing to assemble yet.")
    return 0


def cmd_status(args) -> int:
    """What has been run, what is left, and what went wrong.

    A campaign spans many subjects and arms over days; this is the view that
    says whether the results directory is complete enough to write from. An arm
    that errored is not the same as an arm that has not been run, and neither
    is the same as an arm that ran and kept the baseline -- the grid keeps the
    three apart, because only the first two are work still owed.
    """
    from evaluation.arms import CORE_ARMS, full_suite
    from evaluation.report import load_runs

    runs = load_runs(args.results)
    by_key = {(r.subject, r.arm): r for r in runs}
    subjects = sorted({r.subject for r in runs})
    if args.registry:
        subjects = sorted({s.name for s in Registry.load(args.registry).included}
                          | set(subjects))

    # One selector for every command, so a preset can never mean different
    # things in `run` and `status`.
    arm_names = [a.name for a in _select_arms(args.arms, _models(args.extra_model))]

    if not subjects:
        print("No subjects found. Pass --registry, or run something first.")
        return 1

    symbols = {"modernized": "M", "kept_baseline": "k", "failed": "F",
               "truncated_passes": "p", "truncated_timeout": "t", "error": "E"}
    width = max(len(s) for s in subjects) + 2
    print(f"Results: {args.results}\n")
    print("  M modernized   k kept baseline   F failed   "
          "p/t truncated   E error   . not run\n")
    header = "".join(f"{i:>4}" for i in range(1, len(arm_names) + 1))
    print(" " * width + header)
    missing, errored = 0, []
    for subject in subjects:
        row = ""
        for name in arm_names:
            run = by_key.get((subject, name))
            if run is None:
                row += "   ."
                missing += 1
            else:
                mark = symbols.get(run.outcome, "?")
                row += f"{mark:>4}"
                if run.outcome == "error":
                    errored.append((subject, name, run.reason))
        print(f"{subject:<{width}}{row}")

    print("\nArms:")
    for i, name in enumerate(arm_names, 1):
        done = sum(1 for s in subjects if (s, name) in by_key)
        print(f"  {i:>2}. {name:<28} {done}/{len(subjects)}")

    total = len(subjects) * len(arm_names)
    print(f"\n{total - missing}/{total} runs complete.")
    if errored:
        print(f"\n{len(errored)} errored -- these are work still owed, not results:")
        for subject, arm, reason in errored[:20]:
            print(f"  {subject}/{arm}: {reason[:100]}")
    return 0


def cmd_oracle(args) -> int:
    """Measure how much work each subject's validation logic actually does."""
    from evaluation.oracle import probe_oracle, summarize_reports, write_reports
    from evaluation.runner import prepare_workspace

    registry = Registry.load(args.registry)
    subjects = registry.included
    if args.subjects:
        wanted = {s.strip() for s in args.subjects.split(",")}
        subjects = [s for s in subjects if s.name in wanted]

    reports = []
    for i, subject in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] {subject.name} ...", flush=True)
        workspace = Path(args.workspaces) / subject.name / "_oracle"
        try:
            prepare_workspace(subject, workspace,
                              Path(args.repo_cache) if args.repo_cache else None)
        except Exception as exc:
            print(f"    workspace failed: {exc}")
            continue
        result = probe_oracle(subject, workspace)
        reports.append(result)
        verdict = "FLOOR FAILED" if not result.passes_floor else "ok"
        print(f"    detection {result.detection_rate:.0%}  [{verdict}]"
              + (f"  ({result.error})" if result.error else ""))

    out = write_reports(reports, args.out)
    summary = summarize_reports(reports)
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nWrote {out}")
    return 1 if summary["failed_hard_floor"] else 0


def cmd_replay(args) -> int:
    from evaluation.replay import main as replay_main
    argv = ["--results", args.results, "--backends", args.backends, "--out", args.out]
    if args.markdown:
        argv += ["--markdown", args.markdown]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.prompt_style:
        argv += ["--prompt-style", args.prompt_style]
    if args.execute:
        argv += ["--execute", "--registry", args.registry,
                 "--workspaces", args.workspaces, "--cache", args.cache]
    return replay_main(argv)


def cmd_report(args) -> int:
    out = write_report(args.results, args.out, args.registry, full_arm=args.full_arm)
    print(f"Wrote {out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation.cli")
    parser.add_argument(
        "--extra-model", action="append", default=[], metavar="LABEL=SPEC",
        help="Add a model to the ones every LLM arm is run with (default: "
             + ", ".join(f"{k}={v}" for k, v in MODELS.items())
             + "). e.g. --extra-model gemini=gemini:gemini-2.5-flash")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("arms", help="List experimental arms").set_defaults(func=cmd_arms)

    p_audit = sub.add_parser("audit", help="Check the sample-selection protocol")
    p_audit.add_argument("--registry", required=True)
    p_audit.set_defaults(func=cmd_audit)

    p_run = sub.add_parser("run", help="Execute arms on subjects")
    p_run.add_argument("--registry", required=True)
    p_run.add_argument("--arms", default="core",
                       help="essential | core | all | comma-separated names")
    p_run.add_argument("--subjects", default="", help="Comma-separated subject names")
    p_run.add_argument("--results", default="results")
    p_run.add_argument("--workspaces", default=".workspaces")
    p_run.add_argument("--repo-cache", default=".repo_cache")
    p_run.add_argument("--pip-cache", default=None,
                       help="Shared wheel cache. Strongly recommended: without it "
                            "every candidate re-downloads multi-GB wheels.")
    p_run.add_argument("--timeout", type=int, default=6 * 3600)
    p_run.add_argument("--max-passes", type=int, default=20,
                       help="Ceiling on re-scan passes. A pass that changes "
                            "nothing ends the run, so most subjects stop well "
                            "below this; hitting it is reported as "
                            "truncated_passes, never as convergence.")
    p_run.add_argument("--resume", action="store_true",
                       help="Skip runs whose log already exists")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--skip-audit", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_digest = sub.add_parser(
        "digest", help="Assemble per-subject readable logs from the arm logs")
    p_digest.add_argument("--results", default="results")
    p_digest.set_defaults(func=cmd_digest)

    p_status = sub.add_parser("status", help="Campaign progress across subjects/arms")
    p_status.add_argument("--results", default="results")
    p_status.add_argument("--registry", default="",
                          help="Include subjects that have not been run yet")
    p_status.add_argument("--arms", default="core")
    p_status.set_defaults(func=cmd_status)

    p_oracle = sub.add_parser("oracle", help="Measure validation-script strength")
    p_oracle.add_argument("--registry", required=True)
    p_oracle.add_argument("--subjects", default="")
    p_oracle.add_argument("--workspaces", default=".workspaces")
    p_oracle.add_argument("--repo-cache", default=".repo_cache")
    p_oracle.add_argument("--out", default="oracle_strength.json")
    p_oracle.set_defaults(func=cmd_oracle)

    p_replay = sub.add_parser("replay", help="Replay LLM decisions on other backends")
    p_replay.add_argument("--results", default="results")
    p_replay.add_argument("--backends", required=True,
                          help="e.g. ollama:qwen2.5-coder:14b,gemini:gemini-2.5-flash")
    p_replay.add_argument("--out", default="replay.json")
    p_replay.add_argument("--markdown", default="")
    p_replay.add_argument("--limit", type=int, default=0)
    p_replay.add_argument("--prompt-style", default="full",
                          choices=["full", "compact", "both"])
    p_replay.add_argument("--execute", action="store_true",
                          help="Install and validate each distinct proposed plan")
    p_replay.add_argument("--registry", default="subjects.json")
    p_replay.add_argument("--workspaces", default=".workspaces")
    p_replay.add_argument("--cache", default="replay_execution_cache.json")
    p_replay.set_defaults(func=cmd_replay)

    p_report = sub.add_parser("report", help="Generate the metrics document")
    p_report.add_argument("--results", default="results")
    p_report.add_argument("--registry", default=None)
    p_report.add_argument("--out", default="METRICS.md")
    p_report.add_argument("--full-arm", default=FULL_ARM)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
