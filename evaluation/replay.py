"""Replay recorded LLM decision points against other backends.

Reviewer B asked for sensitivity to the LLM backend. The obvious way to answer
that -- re-run the whole system once per model -- is both expensive and
*confounded*: each backend takes a different search trajectory, so the arms
differ in which decisions they ever face, and almost all of the cost is
install-and-validate cycles that have nothing to do with the model.

Replay instead holds the inputs fixed. Every escalation logged by a reference
run carries the exact prompt inputs (target, error log, current pins, ceilings,
history), so each backend can be asked the identical question. Differences are
then attributable to the model rather than to the path it happened to take.

    python -m evaluation.replay --results results/ \
        --backends ollama:qwen2.5-coder:14b,ollama:llama3.1:8b \
        --out replay.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aura.index import PackageIndex
from aura.llm import build_client
from aura.planner import DecisionPoint, Planner
from aura.telemetry import estimate_cost, read_log
from evaluation.stats import describe, wilson_interval


@dataclass
class Decision:
    """One recorded escalation, with whatever the reference run did about it."""

    subject: str
    arm: str
    package: str
    iteration: int
    point: DecisionPoint
    reference_proposed: list[str] = field(default_factory=list)
    reference_rejected: list[dict] = field(default_factory=list)
    reference_model: str = ""
    accepted_plan: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.subject}/{self.package}#{self.iteration}"


def extract_decisions(results_dir: str | Path,
                      arms: set[str] | None = None) -> list[Decision]:
    """Pull every replayable escalation out of a set of run logs."""
    decisions: list[Decision] = []
    for path in sorted(Path(results_dir).glob("**/*.jsonl")):
        events = read_log(path)
        if not events:
            continue
        arm = events[0].get("arm", "")
        if arms and arm not in arms:
            continue
        subject = events[0].get("repo", "")

        # What the reference run ultimately accepted from the LLM tier.
        accepted: dict[str, dict] = {}
        for event in events:
            if (event.get("event") == "attempt" and event.get("accepted")
                    and event.get("tier") == "llm"):
                accepted[event.get("package", "")] = event.get("plan", {})

        for event in events:
            if event.get("event") != "llm_call":
                continue
            raw_inputs = event.get("inputs") or {}
            if not raw_inputs.get("target"):
                continue  # logged before input capture existed; not replayable
            decisions.append(Decision(
                subject=subject, arm=arm, package=event.get("package", ""),
                iteration=event.get("iteration", 0),
                point=DecisionPoint.from_dict(raw_inputs),
                reference_proposed=event.get("proposed", []),
                reference_rejected=event.get("rejected", []),
                reference_model=event.get("model", ""),
                accepted_plan=accepted.get(event.get("package", ""), {}),
            ))
    return decisions


def replay_backend(decisions: list[Decision], spec: str, *,
                   index: PackageIndex | None = None,
                   explanation_first: bool = True,
                   prompt_style: str = "full",
                   progress: bool = True) -> list[dict]:
    """Ask one backend every recorded question, and score its answers."""
    index = index or PackageIndex()
    planner = Planner(build_client(spec, temperature=0.0), index=index,
                      explanation_first=explanation_first, prompt_style=prompt_style)
    rows: list[dict] = []

    for i, decision in enumerate(decisions, 1):
        if progress:
            print(f"  [{i}/{len(decisions)}] {spec} <- {decision.key}",
                  file=sys.stderr, flush=True)
        started = time.time()
        try:
            plan = planner.replay(decision.point)
        except Exception as exc:
            rows.append({"key": decision.key, "backend": spec, "error": str(exc),
                         "parsed": False})
            continue

        reasons: dict[str, int] = defaultdict(int)
        for item in plan.rejected:
            reasons[item.get("reason", "unknown")] += 1

        proposed_pins = plan.as_pins()
        rows.append({
            "key": decision.key, "backend": spec, "prompt_style": prompt_style,
            "subject": decision.subject,
            "package": decision.package, "iteration": decision.iteration,
            "parsed": bool(plan.raw) and plan.reason != "unparseable_response",
            "resolvable": plan.resolvable,
            "n_accepted": len(plan.requirements),
            "n_rejected": len(plan.rejected),
            "n_proposed": len(plan.requirements) + len(plan.rejected),
            "rejection_reasons": dict(reasons),
            "hallucinated": reasons.get("not_on_index", 0),
            "ceiling_violations": reasons.get("exceeds_ceiling", 0),
            "downgrades": reasons.get("not_an_upgrade", 0),
            "proposed": plan.requirements,
            "proposed_pins": proposed_pins,
            # NOT a correctness measure. AURA has no target version -- it wants
            # any working version above the current one -- so a model proposing a
            # *different* plan that also works is not wrong, and may be better.
            # Whether a plan is good is settled by executing it (see
            # execute_proposals), never by string-matching a reference run.
            "agrees_with_reference": bool(decision.accepted_plan)
                                     and proposed_pins == decision.accepted_plan,
            "has_explanation": bool(plan.explanation.strip()),
            "declared_unresolvable": not plan.resolvable,
            "missing_target": any(i.get("reason") == "missing_target"
                                  for i in plan.rejected),
            "prompt_tokens": plan.usage.prompt_tokens,
            "completion_tokens": plan.usage.completion_tokens,
            "cost_usd": plan.usage.cost_usd,
            "latency_s": round(time.time() - started, 3),
            "error": plan.usage.error,
        })
    return rows


def _plan_key(subject: str, pins: dict) -> str:
    return subject + "|" + ",".join(f"{k}=={v}" for k, v in sorted(pins.items()))


def _releases_recovered(index: PackageIndex, before: dict, after: dict) -> float:
    """How many stable releases the plan advances, summed over its packages."""
    total = 0
    for name, new_version in after.items():
        old_version = before.get(name)
        if not old_version:
            continue
        try:
            history = index.history(name)
        except Exception:
            continue
        if history is None:
            continue
        total += max(0, history.releases_behind(old_version)
                     - history.releases_behind(new_version))
    return float(total)


def execute_proposals(
    rows: list[dict], decisions: list[Decision], *,
    subjects: dict, workspaces_dir: str | Path,
    index: PackageIndex | None = None, spec=None,
    cache_path: str | Path | None = None, progress: bool = True,
) -> list[dict]:
    """Install and validate each distinct proposed plan; annotate the rows.

    This is what settles whether a proposal is any good. String-matching a plan
    against the reference run cannot do it: AURA has no target version, so a
    different plan that installs and passes is a perfectly good answer -- and if
    it recovers more releases, a better one.

    Distinct plans are executed once and the verdict shared across every backend
    that proposed them, which is what keeps the cost tolerable: backends
    converge often, and the reference run's own plans are frequently repeats.
    """
    from evaluation.runner import prepare_workspace
    from aura.environment import EnvSpec, probe
    from aura.reqfile import ReqFile
    from aura.validation import Validator

    index = index or PackageIndex()
    spec = spec or EnvSpec(use_cache=True, build_isolation=True)
    workspaces_dir = Path(workspaces_dir)
    by_key = {d.key: d for d in decisions}

    cache: dict[str, dict] = {}
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        cache = json.loads(cache_file.read_text())

    def save_cache():
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    prepared: dict[str, Path] = {}
    baselines: dict[str, Validator] = {}
    pending = [r for r in rows if r.get("proposed_pins")]

    for i, row in enumerate(pending, 1):
        decision = by_key.get(row["key"])
        subject = subjects.get(row.get("subject", ""))
        if decision is None or subject is None:
            continue
        pins = row["proposed_pins"]
        key = _plan_key(subject.name, pins)

        if key in cache:
            row.update(cache[key])
            continue

        if progress:
            print(f"  [{i}/{len(pending)}] executing {row['backend']} :: "
                  f"{row['key']} -> {list(pins.items())}", file=sys.stderr, flush=True)

        # One workspace per subject, reused across plans.
        if subject.name not in prepared:
            workspace = workspaces_dir / subject.name / "_replay"
            try:
                prepare_workspace(subject, workspace)
            except Exception as exc:
                row.update({"plan_works": None, "execution_error": str(exc)})
                continue
            prepared[subject.name] = workspace
        workspace = prepared[subject.name]
        config = subject.agent_config(workspace)

        # The floor this decision was made from, so drift is judged against the
        # environment the agent actually had -- not the repo's original pins.
        base_pins = dict(decision.point.current)
        base_key = _plan_key(subject.name, base_pins) + "|BASE"
        if base_key not in baselines:
            validator = Validator(config, tolerance=subject.metric_tolerance)
            base_reqs = ReqFile.from_text(
                "".join(f"{n}=={v}\n" for n, v in sorted(base_pins.items())))
            base_result = probe(base_reqs, config=config, validator=validator,
                                spec=spec, env_path=workspace / ".aura_replay_base",
                                record_baseline=True)
            baselines[base_key] = validator if base_result.accepted else None
        validator = baselines[base_key]
        if validator is None:
            verdict = {"plan_works": None,
                       "execution_error": "decision-point baseline did not validate"}
            cache[key] = verdict
            row.update(verdict)
            save_cache()
            continue

        candidate_pins = {**base_pins, **pins}
        candidate = ReqFile.from_text(
            "".join(f"{n}=={v}\n" for n, v in sorted(candidate_pins.items())))
        result = probe(candidate, config=config, validator=validator, spec=spec,
                       env_path=workspace / ".aura_replay_cand")

        verdict = {
            "plan_works": bool(result.accepted),
            "plan_installs": bool(result.install_ok),
            "plan_error_class": result.error_class,
            "releases_recovered": (_releases_recovered(index, base_pins, pins)
                                   if result.accepted else 0.0),
            "execution_error": "",
        }
        cache[key] = verdict
        row.update(verdict)
        save_cache()

    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Per-backend summary: the table that answers the backend-sensitivity RQ."""
    by_backend: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        # Prompt style is part of the condition, not a property of the model.
        style = row.get("prompt_style", "full")
        label = row["backend"] if style == "full" else f"{row['backend']} [{style}]"
        by_backend[label].append(row)

    out = []
    for backend, group in sorted(by_backend.items()):
        n = len(group)
        parsed = sum(1 for r in group if r.get("parsed"))
        bailed = sum(1 for r in group if r.get("declared_unresolvable"))
        dropped_target = sum(1 for r in group if r.get("missing_target"))
        proposed = sum(r.get("n_proposed", 0) for r in group)
        hallucinated = sum(r.get("hallucinated", 0) for r in group)
        ceiling = sum(r.get("ceiling_violations", 0) for r in group)
        downgrades = sum(r.get("downgrades", 0) for r in group)
        rejected = sum(r.get("n_rejected", 0) for r in group)
        comparable = [r for r in group if r.get("agrees_with_reference") is not None]
        agreed = sum(1 for r in group if r.get("agrees_with_reference"))
        # Execution verdicts; present only after execute_proposals has run.
        executed = [r for r in group if r.get("plan_works") is not None]
        works = sum(1 for r in executed if r.get("plan_works"))
        recovered = [float(r["releases_recovered"]) for r in executed
                     if r.get("releases_recovered") is not None]
        out.append({
            "backend": backend,
            "decisions": n,
            "schema_valid": str(wilson_interval(parsed, n)),
            # A backend that declares everything unresolvable never proposes a
            # bad version, so it scores perfectly on every validity metric while
            # resolving nothing. Reported so that cannot be mistaken for quality.
            "declared_unresolvable_rate": round(bailed / n, 4) if n else None,
            "dropped_target_rate": round(dropped_target / n, 4) if n else None,
            "versions_proposed": proposed,
            "wrong_version_rate": round(rejected / proposed, 4) if proposed else None,
            "hallucination_rate": round(hallucinated / proposed, 4) if proposed else None,
            "ceiling_violation_rate": round(ceiling / proposed, 4) if proposed else None,
            "downgrade_rate": round(downgrades / proposed, 4) if proposed else None,
            # The utility measure: does the proposed environment install and pass?
            "plans_executed": len(executed),
            "plan_works_rate": (str(wilson_interval(works, len(executed)))
                                if executed else "not executed"),
            "releases_recovered": describe(recovered),
            # Descriptive only: do backends converge on the same answer?
            "agreement_with_reference": (str(wilson_interval(agreed, len(comparable)))
                                         if comparable else "n/a"),
            "latency_s": describe([r.get("latency_s", 0.0) for r in group]),
            "prompt_tokens": describe([float(r.get("prompt_tokens", 0)) for r in group]),
            "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in group), 4),
            "errors": sum(1 for r in group if r.get("error")),
        })
    return out


def render_markdown(summary: list[dict]) -> str:
    header = ("| Backend | Decisions | Schema valid | Bailed early | Dropped target | "
              "Hallucination | Ceiling viol. | Downgrades | **Plan works** | "
              "Releases recovered | Agreement | Median latency | Cost |\n"
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")

    def pct(value):
        return "-" if value is None else f"{100 * value:.1f}%"

    rows = []
    for item in summary:
        recovered = item.get("releases_recovered") or {}
        rows.append(
            f"| `{item['backend']}` | {item['decisions']} | {item['schema_valid']} | "
            f"{pct(item['declared_unresolvable_rate'])} | "
            f"{pct(item['dropped_target_rate'])} | "
            f"{pct(item['hallucination_rate'])} | {pct(item['ceiling_violation_rate'])} | "
            f"{pct(item['downgrade_rate'])} | {item['plan_works_rate']} | "
            f"{recovered.get('mean', 0):.1f} | {item['agreement_with_reference']} | "
            f"{item['latency_s']['median']:.1f}s | ${item['total_cost_usd']:.4f} |")

    return (header + "\n".join(rows) + "\n\n"
            "_**Bailed early** is the rate at which a backend declared the "
            "conflict unresolvable; **dropped target** the rate at which it "
            "proposed a plan omitting the package it was asked about. Both are "
            "ways of scoring well on validity while resolving nothing, so they "
            "belong beside the validity columns.\n\n"
            "_**Plan works** is the utility measure: the proposed environment was "
            "installed and the project's own tests were run against it. "
            "**Agreement** is descriptive only -- it reports whether backends "
            "converge on the same answer, not whether an answer is correct. AURA "
            "has no target version, so a different working plan is not a wrong "
            "one and may recover more releases._\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation.replay")
    parser.add_argument("--results", required=True,
                        help="Directory of run logs to take decision points from")
    parser.add_argument("--backends", required=True,
                        help="Comma-separated LLM specs, e.g. "
                             "ollama:qwen2.5-coder:14b,gemini:gemini-2.5-flash")
    parser.add_argument("--arms", default="",
                        help="Restrict to these arms (comma-separated)")
    parser.add_argument("--out", default="replay.json")
    parser.add_argument("--markdown", default="")
    parser.add_argument("--limit", type=int, default=0,
                        help="Replay only the first N decision points (smoke test)")
    parser.add_argument("--no-explanation", action="store_true",
                        help="Ask for a plan directly, to test explanation-first")
    parser.add_argument("--prompt-style", default="full",
                        choices=["full", "compact", "both"],
                        help="'compact' is tuned for small local models; 'both' "
                             "runs each backend under both styles for comparison")
    parser.add_argument("--execute", action="store_true",
                        help="Install and validate each distinct proposed plan. "
                             "Without this, only validity is measured -- whether a "
                             "plan actually works cannot be inferred from the text.")
    parser.add_argument("--registry", default="subjects.json",
                        help="Subject registry (required with --execute)")
    parser.add_argument("--workspaces", default=".workspaces")
    parser.add_argument("--cache", default="replay_execution_cache.json",
                        help="Verdicts for already-executed plans, so reruns are cheap")
    args = parser.parse_args(argv)

    arms = {a.strip() for a in args.arms.split(",") if a.strip()} or None
    decisions = extract_decisions(args.results, arms)
    if args.limit:
        decisions = decisions[: args.limit]
    if not decisions:
        print("No replayable decision points found. Runs must have been produced "
              "with prompt-input logging enabled.", file=sys.stderr)
        return 1
    print(f"{len(decisions)} decision points from "
          f"{len({d.subject for d in decisions})} subjects", file=sys.stderr)

    index = PackageIndex()
    styles = ["full", "compact"] if args.prompt_style == "both" else [args.prompt_style]
    rows: list[dict] = []
    for spec in [b.strip() for b in args.backends.split(",") if b.strip()]:
        for style in styles:
            print(f"Replaying against {spec} [{style} prompt] ...", file=sys.stderr)
            rows += replay_backend(decisions, spec, index=index,
                                   explanation_first=not args.no_explanation,
                                   prompt_style=style)

    if args.execute:
        from evaluation.subjects import Registry
        registry = Registry.load(args.registry)
        subjects = {s.name: s for s in registry.subjects}
        distinct = len({_plan_key(r.get("subject", ""), r["proposed_pins"])
                        for r in rows if r.get("proposed_pins")})
        print(f"Executing {distinct} distinct plans "
              f"(from {len(rows)} proposals) ...", file=sys.stderr)
        rows = execute_proposals(rows, decisions, subjects=subjects,
                                 workspaces_dir=args.workspaces, index=index,
                                 cache_path=args.cache)

    summary = summarize(rows)
    Path(args.out).write_text(
        json.dumps({"decisions": [asdict(d) | {"point": d.point.as_dict()}
                                  for d in decisions],
                    "rows": rows, "summary": summary}, indent=2, default=str) + "\n",
        encoding="utf-8")
    print(f"Wrote {args.out}", file=sys.stderr)

    markdown = render_markdown(summary)
    if args.markdown:
        Path(args.markdown).write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.markdown}", file=sys.stderr)
    print("\n" + markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
