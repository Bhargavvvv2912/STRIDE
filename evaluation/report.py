"""Turns run logs into the paper's tables.

Every number the paper reports is produced here from the JSONL logs, so a table
can be regenerated after a rerun instead of being transcribed by hand. Each
section maps to a specific reviewer request; the mapping is stated in the
rendered document so the response letter can point at it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from aura.telemetry import Outcome, read_log
from evaluation.stats import describe, mcnemar_exact, paired_outcomes, wilson_interval
from evaluation.arms import FULL_ARM

MODERNIZING = {Outcome.MODERNIZED.value, Outcome.TRUNCATED_PASSES.value,
               Outcome.TRUNCATED_TIMEOUT.value}
NON_REGRESSING = MODERNIZING | {Outcome.KEPT_BASELINE.value}


@dataclass
class RunSummary:
    """One (subject, arm) run, flattened from its event log."""

    subject: str
    arm: str
    outcome: str = Outcome.ERROR.value
    reason: str = ""
    wall_clock_s: float = 0.0
    updated: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    version_age: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)
    llm_calls: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    validations: list = field(default_factory=list)
    drifts: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    prioritizations: list = field(default_factory=list)
    routings: list = field(default_factory=list)
    graphs: list = field(default_factory=list)
    coverages: list = field(default_factory=list)
    batches: list = field(default_factory=list)
    proposals: list = field(default_factory=list)
    complete: bool = False

    # ------------------------------------------------------------- accessors

    @property
    def modernized(self) -> bool:
        return self.outcome in MODERNIZING and bool(self.updated)

    @property
    def non_regressing(self) -> bool:
        return self.outcome in NON_REGRESSING

    @property
    def truncated(self) -> bool:
        return self.outcome in (Outcome.TRUNCATED_PASSES.value,
                                Outcome.TRUNCATED_TIMEOUT.value)

    @property
    def tiers_used(self) -> set[str]:
        """Tiers that produced an *accepted* configuration in this run."""
        return {a["tier"] for a in self.attempts if a.get("accepted")}

    @property
    def cost_usd(self) -> float:
        return float(self.counters.get("llm_cost_usd", 0.0))

    @property
    def tokens(self) -> int:
        return int(self.counters.get("prompt_tokens", 0)
                   + self.counters.get("completion_tokens", 0))

    def age_delta(self) -> float | None:
        before = (self.version_age.get("before") or {}).get("mean_releases_behind")
        after = (self.version_age.get("after") or {}).get("mean_releases_behind")
        if before is None or after is None:
            return None
        return round(before - after, 3)


def summarize_log(path: str | Path) -> RunSummary | None:
    events = read_log(path)
    if not events:
        return None
    head = events[0]
    summary = RunSummary(subject=head.get("repo", ""), arm=head.get("arm", ""),
                         config=head.get("config", {}), env=head.get("env", {}))
    for event in events:
        kind = event.get("event")
        if kind == "attempt":
            summary.attempts.append(event)
        elif kind == "llm_call":
            summary.llm_calls.append(event)
        elif kind == "conflict_detection":
            summary.conflicts.append(event)
        elif kind == "validation":
            summary.validations.append(event)
        elif kind == "behavior_check":
            summary.drifts.append(event)
        elif kind == "prioritization":
            summary.prioritizations.append(event)
        elif kind == "routing":
            summary.routings.append(event)
        elif kind == "coverage":
            summary.coverages.append(event)
        elif kind == "batch":
            summary.batches.append(event)
        elif kind == "proposal":
            summary.proposals.append(event)
        elif kind == "constraint_graph":
            summary.graphs.append(event)
        elif kind == "note":
            summary.notes.append(event)
        elif kind == "run_end":
            summary.outcome = event.get("outcome", Outcome.ERROR.value)
            summary.reason = event.get("reason", "")
            summary.wall_clock_s = event.get("wall_clock_s", 0.0)
            summary.updated = event.get("updated", {})
            summary.counters = event.get("counters", {})
            summary.version_age = event.get("version_age", {})
            summary.complete = True
    return summary


def load_runs(results_dir: str | Path) -> list[RunSummary]:
    runs = []
    for path in sorted(Path(results_dir).glob("**/*.jsonl")):
        summary = summarize_log(path)
        if summary:
            runs.append(summary)
    return runs


class Analysis:
    """Aggregations over a set of runs, indexed by (subject, arm)."""

    def __init__(self, runs: list[RunSummary]):
        self.runs = runs
        self.by_arm: dict[str, list[RunSummary]] = defaultdict(list)
        self.index: dict[tuple[str, str], RunSummary] = {}
        for run in runs:
            self.by_arm[run.arm].append(run)
            self.index[(run.subject, run.arm)] = run

    @property
    def arms(self) -> list[str]:
        return sorted(self.by_arm)

    @property
    def subjects(self) -> list[str]:
        return sorted({r.subject for r in self.runs})

    def success_map(self, arm: str, predicate=lambda r: r.non_regressing) -> dict[str, bool]:
        return {r.subject: predicate(r) for r in self.by_arm.get(arm, [])}

    # ------------------------------------------------------------ RQ1 and RQ2

    def outcome_table(self) -> list[dict]:
        rows = []
        for arm in self.arms:
            runs = self.by_arm[arm]
            counts = defaultdict(int)
            for r in runs:
                counts[r.outcome] += 1
            nonreg = sum(1 for r in runs if r.non_regressing)
            rows.append({
                "arm": arm, "n": len(runs),
                **{o.value: counts.get(o.value, 0) for o in Outcome},
                # Columns partition the runs: a truncated run is reported as
                # truncated, never also as modernized. "any_progress" is the
                # union of the two and is stated separately.
                "modernized_n": counts.get(Outcome.MODERNIZED.value, 0),
                "any_progress_n": sum(1 for r in runs if r.modernized),
                "non_regressing": str(wilson_interval(nonreg, len(runs))),
            })
        return rows

    def compare(self, arm_a: str, arm_b: str,
                predicate=lambda r: r.modernized) -> dict:
        a, b = self.success_map(arm_a, predicate), self.success_map(arm_b, predicate)
        result = paired_outcomes(a, b)
        shared = sorted(set(a) & set(b))
        return {
            "arm_a": arm_a, "arm_b": arm_b, "n_shared": len(shared),
            "a_only": sorted(s for s in shared if a[s] and not b[s]),
            "b_only": sorted(s for s in shared if b[s] and not a[s]),
            "both": sum(1 for s in shared if a[s] and b[s]),
            "neither": sum(1 for s in shared if not a[s] and not b[s]),
            "mcnemar": str(result), "p_value": result.p_value,
        }

    # --------------------------------------------------------- component data

    def conflict_precision(self, arms: list[str] | None = None) -> dict:
        """How accurately the log parser names the packages in a conflict.

        Two measures, and only one is a real test. "Is the extracted name
        installed?" is passed by every name in the requirements file, so it
        mostly measures whether the parser read the file back. When pip printed
        its own conflict report, the extracted names are scored against the
        packages pip actually named -- that is the figure to report.
        """
        values, extracted, real = [], 0, 0
        vs_report, report_events, from_report = [], 0, 0
        for run in self.runs:
            if arms and run.arm not in arms:
                continue
            for event in run.conflicts:
                if event.get("precision") is not None:
                    values.append(event["precision"])
                extracted += event.get("extracted_n", 0)
                real += event.get("in_env_n", 0)
                if event.get("reported") is not None:
                    report_events += 1
                    if event.get("precision_vs_report") is not None:
                        vs_report.append(event["precision_vs_report"])
                if event.get("source") == "pip_conflict_report":
                    from_report += 1

        return {
            "events": len(values), "per_event": describe(values),
            "micro_precision": round(real / extracted, 4) if extracted else None,
            "names_extracted": extracted, "names_in_environment": real,
            "events_with_pip_report": report_events,
            "blockers_from_pip_report": from_report,
            "precision_vs_pip_report": describe(vs_report),
        }

    def llm_quality(self, arms: list[str] | None = None) -> dict:
        """First-attempt success, refinement depth, and the wrong-version rate."""
        by_run_package: dict[tuple, list[dict]] = defaultdict(list)
        proposed = rejected = 0
        rejection_reasons: dict[str, int] = defaultdict(int)
        early_exits = 0

        for run in self.runs:
            if arms and run.arm not in arms:
                continue
            accepted_at = {}
            for note in run.notes:
                if note.get("note") == "llm_plan_accepted":
                    accepted_at[(run.subject, run.arm, note.get("package"))] = note["iteration"]
                elif note.get("note") == "llm_early_exit":
                    early_exits += 1
            for call in run.llm_calls:
                key = (run.subject, run.arm, call.get("package"))
                by_run_package[key].append(call)
                proposed += len(call.get("proposed", [])) + len(call.get("rejected", []))
                rejected += len(call.get("rejected", []))
                for item in call.get("rejected", []):
                    rejection_reasons[item.get("reason", "unknown")] += 1
            self._accepted_at = getattr(self, "_accepted_at", {})
            self._accepted_at.update(accepted_at)

        accepted_at = getattr(self, "_accepted_at", {})
        first_try = sum(1 for k, v in accepted_at.items() if v == 1)
        iterations = [len(v) for v in by_run_package.values()]
        hallucinated = rejection_reasons.get("not_on_index", 0)

        return {
            "escalations": len(by_run_package),
            "resolved": len(accepted_at),
            "first_attempt_success": (
                str(wilson_interval(first_try, len(accepted_at))) if accepted_at else "n/a"),
            "iterations_per_escalation": describe([float(i) for i in iterations]),
            "versions_proposed": proposed,
            "versions_rejected": rejected,
            "wrong_version_rate": round(rejected / proposed, 4) if proposed else None,
            "hallucination_rate": round(hallucinated / proposed, 4) if proposed else None,
            "rejection_reasons": dict(sorted(rejection_reasons.items(), key=lambda kv: -kv[1])),
            "early_exits": early_exits,
        }

    def cost_and_runtime(self) -> list[dict]:
        rows = []
        for arm in self.arms:
            runs = self.by_arm[arm]
            rows.append({
                "arm": arm,
                "runtime_s": describe([r.wall_clock_s for r in runs]),
                "attempts": describe([float(r.counters.get("attempts", 0)) for r in runs]),
                "llm_calls": describe([float(r.counters.get("llm_calls", 0)) for r in runs]),
                "tokens": describe([float(r.tokens) for r in runs]),
                "cost_usd": describe([r.cost_usd for r in runs]),
                "total_cost_usd": round(sum(r.cost_usd for r in runs), 4),
            })
        return rows

    def modernization_magnitude(self) -> list[dict]:
        """'How much newer' -- what makes the word 'modernized' a number."""
        rows = []
        for arm in self.arms:
            deltas = [d for r in self.by_arm[arm] if (d := r.age_delta()) is not None]
            after = [(r.version_age.get("after") or {}).get("at_latest_pct")
                     for r in self.by_arm[arm]]
            after = [a for a in after if a is not None]
            rows.append({
                "arm": arm,
                "mean_releases_recovered": describe(deltas),
                "pct_at_latest_after": describe(after),
            })
        return rows

    def strategy_usage(self, arm: str = FULL_ARM) -> dict:
        """Which tiers produced accepted updates, and in how many subjects."""
        counts: dict[str, int] = defaultdict(int)
        combos: dict[str, int] = defaultdict(int)
        for run in self.by_arm.get(arm, []):
            tiers = run.tiers_used
            for tier in tiers:
                counts[tier] += 1
            combos["+".join(sorted(tiers)) or "none"] += 1
        return {"arm": arm, "subjects": len(self.by_arm.get(arm, [])),
                "per_tier": dict(counts), "combinations": dict(sorted(combos.items()))}

    def order_sensitivity(self, arms: list[str]) -> dict:
        """Does the final configuration depend on the order packages were tried?"""
        per_subject: dict[str, set] = defaultdict(set)
        outcomes: dict[str, set] = defaultdict(set)
        for arm in arms:
            for run in self.by_arm.get(arm, []):
                pins = tuple(sorted((k, v[1]) for k, v in run.updated.items()))
                per_subject[run.subject].add(pins)
                outcomes[run.subject].add(run.outcome)
        divergent = [s for s, v in per_subject.items() if len(v) > 1]
        return {
            "arms": arms, "subjects": len(per_subject),
            "identical_final_config": len(per_subject) - len(divergent),
            "divergent_final_config": len(divergent),
            "divergent_subjects": sorted(divergent),
            "divergent_outcome": sorted(s for s, v in outcomes.items() if len(v) > 1),
        }

    def structural_false_positives(self, arm: str = "install_only") -> dict:
        """How often an unverified, resolver-only answer is actually broken.

        The install-only arm never runs tests during its search; its final
        configuration is validated afterwards. A failure here is a configuration
        that a purely structural analysis would have accepted and shipped.
        """
        checked, broken, subjects = 0, 0, []
        for run in self.by_arm.get(arm, []):
            verdicts = [v for v in run.validations if v.get("phase") == "post_hoc"]
            if not verdicts:
                continue
            checked += 1
            if not verdicts[-1].get("passed"):
                broken += 1
                subjects.append(run.subject)
        return {
            "arm": arm, "checked": checked, "broken": broken,
            "false_positive_rate": (str(wilson_interval(broken, checked))
                                    if checked else "n/a"),
            "broken_subjects": sorted(subjects),
        }

    def behavioural_drift(self, strict_arm: str = FULL_ARM,
                          lenient_arm: str = "aura_tests_only") -> dict:
        """How many 'successes' change the project's numbers while tests stay green."""
        drifted = [{"subject": r.subject, "metrics": e.get("drifted", []),
                    "detail": e.get("candidate", {})}
                   for r in self.runs if r.arm == strict_arm for e in r.drifts]
        strict = self.success_map(strict_arm, lambda r: r.modernized)
        lenient = self.success_map(lenient_arm, lambda r: r.modernized)
        shared = sorted(set(strict) & set(lenient))
        return {
            "drift_events": len(drifted),
            "subjects_with_drift": sorted({d["subject"] for d in drifted}),
            "lenient_only": sorted(s for s in shared if lenient[s] and not strict[s]),
            "note": ("Subjects listed under lenient_only pass their tests after "
                     "upgrading but change the numbers the project produces. Under "
                     "an exit-code oracle these would be counted as successes."),
            "examples": drifted[:20],
        }

    # ------------------------------------------------- search cost & failures

    def failure_taxonomy(self, arms: list[str] | None = None) -> list[dict]:
        """Why candidates fail, by class.

        v1 computed this classification on every failed attempt and then used it
        for nothing: it was neither routed on nor reported. The data is already
        in every attempt event, so the table costs nothing to produce.
        """
        pool = [r for r in self.runs if arms is None or r.arm in arms]
        counts: dict[str, dict] = {}
        for run in pool:
            for attempt in run.attempts:
                if attempt.get("accepted"):
                    continue
                cls = attempt.get("error_class") or "unknown"
                row = counts.setdefault(
                    cls, {"class": cls, "failures": 0, "seconds": 0.0,
                          "subjects": set()})
                row["failures"] += 1
                row["seconds"] += attempt.get("duration_s", 0.0) or 0.0
                row["subjects"].add(run.subject)
        total = sum(r["failures"] for r in counts.values()) or 1
        out = []
        for row in counts.values():
            out.append({
                "class": row["class"],
                "failures": row["failures"],
                "share": round(row["failures"] / total, 4),
                "wasted_minutes": round(row["seconds"] / 60.0, 1),
                "subjects": len(row["subjects"]),
            })
        return sorted(out, key=lambda r: -r["failures"])

    def cost_effect(self, on_arm: str = FULL_ARM,
                    off_arm: str = "aura_linear_greedy_llm") -> dict:
        """Does a search-cost reduction change the answer, or only the cost?

        The first key is the one that matters. A speedup that also changes the
        final configuration is not a speedup, it is a different search, and any
        subject listed under ``diverged`` has to be explained rather than
        averaged away.
        """
        on = {r.subject: r for r in self.runs if r.arm == on_arm}
        off = {r.subject: r for r in self.runs if r.arm == off_arm}
        shared = sorted(set(on) & set(off))

        diverged, rows = [], []
        for subject in shared:
            a, b = on[subject], off[subject]
            same = a.updated == b.updated
            if not same:
                diverged.append({"subject": subject,
                                 "routing_on": a.updated, "routing_off": b.updated})
            rows.append({
                "subject": subject,
                "same_result": same,
                "probes_on": a.counters.get("attempts", 0),
                "probes_off": b.counters.get("attempts", 0),
                "probes_saved": b.counters.get("attempts", 0) - a.counters.get("attempts", 0),
                "seconds_on": round(a.wall_clock_s, 1),
                "seconds_off": round(b.wall_clock_s, 1),
                "skipped_scans": a.counters.get("routing_skipped_linear", 0),
                "fallback_rescued": a.counters.get("routing_fallback_rescued", 0),
                "wheel_only_rescued": a.counters.get("wheel_only_rescued", 0),
            })
        saved = sum(r["probes_saved"] for r in rows)
        on_secs = sum(r["seconds_on"] for r in rows)
        off_secs = sum(r["seconds_off"] for r in rows)
        return {
            "subjects": len(shared),
            "identical_results": sum(1 for r in rows if r["same_result"]),
            "diverged": diverged,
            "probes_saved_total": saved,
            "probes_saved_pct": (round(saved / off_total, 4)
                                 if (off_total := sum(r["probes_off"] for r in rows))
                                 else None),
            "seconds_on": round(on_secs, 1),
            "seconds_off": round(off_secs, 1),
            "speedup": round(off_secs / on_secs, 2) if on_secs else None,
            "rows": rows,
        }

    def graph_vs_regex(self, arms: list[str] | None = None) -> dict:
        """Score the error-log regex against the declared-constraint graph.

        Reviewer A asked whether the coupling detection actually works. v1 could
        not answer: the regex was the only source of coupling, so there was
        nothing to check it against. The graph gives a declared ground truth for
        the pairs metadata states, which is a floor on coupling, not a ceiling --
        undeclared coupling (a C-ABI break, say) is real and appears in neither.
        """
        pool = [r for r in self.runs if arms is None or r.arm in arms]
        declared: set[tuple[str, frozenset]] = set()
        extracted: set[tuple[str, frozenset]] = set()
        unknown_meta: set[str] = set()

        for run in pool:
            for event in run.graphs:
                for pair in event.get("couples", []) or []:
                    if len(pair) == 2:
                        declared.add((run.subject, frozenset(pair)))
                unknown_meta.update(event.get("unknown", []) or [])
            for event in run.conflicts:
                package = event.get("package")
                for other in event.get("extracted", []) or []:
                    if other and other != package:
                        extracted.add((run.subject, frozenset((package, other))))

        hits = declared & extracted
        precision = len(hits) / len(extracted) if extracted else None
        recall = len(hits) / len(declared) if declared else None
        return {
            "declared_pairs": len(declared),
            "regex_pairs": len(extracted),
            "agreement": len(hits),
            "regex_precision_vs_declared": (round(precision, 4)
                                            if precision is not None else None),
            "regex_recall_vs_declared": (round(recall, 4)
                                         if recall is not None else None),
            "packages_without_metadata": sorted(unknown_meta),
            "regex_only": [sorted(p) for _, p in sorted(extracted - declared)][:25],
            "declared_only": [sorted(p) for _, p in sorted(declared - extracted)][:25],
            "note": ("Declared pairs are a floor on true coupling, not ground "
                     "truth for it: metadata cannot express a binary-ABI break. "
                     "'regex_only' pairs are therefore not automatically false "
                     "positives, and must be read case by case."),
        }

    def oracle_coverage(self, arm: str = "aura_coverage") -> dict:
        """How much code each subject's validation actually executes.

        Reviewer B: "A passing validation script only shows that no failure was
        observed for the covered executions." This is that caveat, quantified.
        The dependency figure is the one that bears on a dependency upgrade: a
        script can cover most of a repo while touching a handful of functions in
        the package being upgraded, and the two are not interchangeable evidence.
        """
        rows, drops = [], []
        for run in sorted((r for r in self.runs if r.arm == arm),
                          key=lambda r: r.subject):
            baseline = next((e for e in run.coverages
                             if e.get("phase") == "baseline"), None)
            delta = next((e for e in run.coverages
                          if e.get("phase") == "delta"), None)
            if not baseline:
                continue
            if not baseline.get("ok"):
                rows.append({"subject": run.subject, "repo_pct": None,
                             "dependencies": {}, "note": baseline.get("reason", "")})
                continue
            deps = baseline.get("dependencies") or {}
            measured = {n: d.get("pct") for n, d in deps.items()
                        if d.get("pct") is not None}
            compiled = sorted(n for n, d in deps.items() if d.get("compiled"))
            rows.append({
                "subject": run.subject,
                "repo_pct": (baseline.get("repo") or {}).get("pct"),
                "dependencies": measured,
                "median_dependency_pct": (
                    sorted(measured.values())[len(measured) // 2]
                    if measured else None),
                "weakest": (min(measured, key=measured.get) if measured else None),
                "compiled": compiled,
                "note": baseline.get("reason", ""),
            })
            for dropped in ((delta or {}).get("dropped") or []):
                drops.append({"subject": run.subject, **dropped})

        repo_pcts = [r["repo_pct"] for r in rows if r["repo_pct"] is not None]
        dep_pcts = [p for r in rows for p in r["dependencies"].values()]
        return {
            "subjects": len(rows),
            "repo_coverage": describe(repo_pcts),
            "dependency_coverage": describe(dep_pcts),
            "coverage_drops": drops,
            "rows": rows,
            "note": ("coverage.py measures executed Python lines. Packages "
                     "flagged 'compiled' ship extension modules, so their figure "
                     "reflects the Python wrapper layer and understates what ran."),
        }

    def routing_effect(self, on_arm: str = FULL_ARM,
                       off_arm: str = "aura_no_routing") -> dict:
        """Routing isolated: same tiers, same graph, routing on vs off."""
        return self.cost_effect(on_arm, off_arm)

    def graph_effect(self, on_arm: str = FULL_ARM,
                     off_arm: str = "aura_no_graph") -> dict:
        """The declared-constraint graph isolated: reading metadata vs not."""
        return self.cost_effect(on_arm, off_arm)

    def batching_effect(self, on_arm: str = FULL_ARM,
                        off_arm: str = "aura_serial") -> dict:
        """k=production against k=1: identical answers, less wall-clock.

        Batching is the one search-cost reduction that changes *how* the oracle
        is consulted rather than how often, so it is the one most able to change
        an answer without anyone noticing -- a probe that only failed because
        five installs were competing for the same disk looks exactly like a
        probe that failed on its merits. The first row is therefore the claim
        being defended, and the timings mean nothing without it.
        """
        effect = self.cost_effect(on_arm, off_arm)
        widths, guards = {}, {"rechecks": 0, "rescued": 0,
                              "confirmations": 0, "confirmation_failures": 0}
        for run in self.runs:
            if run.arm != on_arm:
                continue
            for event in run.batches:
                if event.get("phase") == "plan":
                    widths.setdefault(event.get("package", "?"), event.get("k"))
            guards["rechecks"] += run.counters.get("contention_rechecks", 0)
            guards["rescued"] += run.counters.get("contention_rescued", 0)
            guards["confirmations"] += run.counters.get("batch_confirmations", 0)
            guards["confirmation_failures"] += run.counters.get(
                "batch_confirmation_failures", 0)
        effect["batch_widths"] = widths
        effect["guards"] = guards
        return effect

    def per_package_advance(self, arms: list[str] | None = None) -> dict:
        """How many *dependencies* each arm actually moved forward.

        The repository-level outcome hides most of what happened: a subject with
        twelve outdated pins counts the same whether an arm advanced one of them
        or all twelve. The denominator here is the subject's own outdated pins,
        computed identically for every arm from the baseline version-age report,
        so the arms are comparable package by package.

        This is also the only unit on which a per-dependency proposer can be
        scored fairly. Dependabot never proposes a combined change, so its
        repository-level result is an artefact of how one chooses to merge its
        pull requests; its per-package result is simply what it proposed.
        """
        pool = [r for r in self.runs if arms is None or r.arm in arms]
        by_arm: dict[str, dict] = {}
        for run in pool:
            behind = {n: b for n, b
                      in ((run.version_age.get("before") or {}).get(
                          "per_package") or {}).items() if b > 0}
            row = by_arm.setdefault(run.arm, {
                "arm": run.arm, "subjects": 0, "outdated": 0, "advanced": 0,
                "per_subject": []})
            advanced = [n for n in run.updated if n in behind] if run.updated else []
            row["subjects"] += 1
            row["outdated"] += len(behind)
            row["advanced"] += len(advanced)
            row["per_subject"].append({
                "subject": run.subject, "outdated": len(behind),
                "advanced": len(advanced),
                "rate": round(len(advanced) / len(behind), 4) if behind else None,
            })

        for row in by_arm.values():
            row["rate"] = (round(row["advanced"] / row["outdated"], 4)
                           if row["outdated"] else None)
            row["per_subject_rate"] = describe(
                [s["rate"] for s in row["per_subject"] if s["rate"] is not None])
        return by_arm

    def coupled_failures(self, arms: list[str] | None = None) -> dict:
        """Proposals that failed alone but that AURA advanced by moving a group.

        This is the co-resolution gap made measurable rather than anecdotal. A
        per-dependency proposer offers torch and torchvision as separate
        changes; where the two must move together, each change fails on its own
        and the tool has no way to express the pair. Every package listed here
        is one AURA advanced and the proposer could not.
        """
        proposal_arms = arms or ["dependabot"]
        rejected: dict[tuple, dict] = {}
        for run in self.runs:
            if run.arm not in proposal_arms:
                continue
            for event in run.proposals:
                package = event.get("package")
                if package == "<union>" or event.get("accepted"):
                    continue
                rejected[(run.subject, package)] = {
                    "subject": run.subject, "package": package,
                    "proposed": event.get("to_version"),
                    "error_class": event.get("error_class"),
                    "arm": run.arm,
                }

        aura = {r.subject: r for r in self.runs if r.arm == FULL_ARM}
        recovered, union_rows = [], []
        for (subject, package), entry in sorted(rejected.items()):
            run = aura.get(subject)
            if run and package in (run.updated or {}):
                recovered.append({**entry,
                                  "aura_reached": (run.updated[package] or ["", ""])[-1],
                                  "aura_moved_with": sorted(
                                      n for n in run.updated if n != package)})
        for run in self.runs:
            if run.arm not in proposal_arms:
                continue
            for event in run.proposals:
                if event.get("package") == "<union>":
                    union_rows.append({"subject": run.subject, "arm": run.arm,
                                       "accepted": event.get("accepted"),
                                       "error_class": event.get("error_class")})
        return {
            "rejected_proposals": len(rejected),
            "recovered_by_aura": len(recovered),
            "examples": recovered[:25],
            "union_probe": union_rows,
            "note": ("Packages listed here were proposed alone, failed alone, "
                     "and were advanced by AURA as part of a group. The union "
                     "probe applies every proposal at once -- informative, but "
                     "a combination the proposer never offered."),
        }
