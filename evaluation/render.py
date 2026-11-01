"""Renders the analysis into a single Markdown metrics document.

Each section states the reviewer request it answers, so the response letter can
cite section numbers and the numbers themselves are never transcribed by hand.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from aura.telemetry import Outcome
from evaluation.report import Analysis, RunSummary
from evaluation.stats import wilson_interval
from evaluation.arms import FULL_ARM

OUTCOME_ORDER = [o.value for o in Outcome]


def _table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    if not rows:
        return "_No data yet._\n"
    sep = "|" + "|".join(f" {align or '---'} " for _ in headers) + "|"
    out = ["| " + " | ".join(headers) + " |", sep]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _describe_cell(d: dict, digits: int = 1) -> str:
    if not d or not d.get("n"):
        return "-"
    return f"{_fmt(d['median'], digits)} (med), {_fmt(d['mean'], digits)} (mean), max {_fmt(d['max'], digits)}"


def render(analysis: Analysis, registry=None, *, title: str = "AURA evaluation metrics",
           full_arm: str = FULL_ARM) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = [
        f"# {title}\n",
        f"_Generated {now} from run logs. Every number below is computed from the "
        f"JSONL event logs; nothing here is entered by hand._\n",
        f"**Runs loaded:** {len(analysis.runs)} across {len(analysis.subjects)} "
        f"subjects and {len(analysis.arms)} arms.\n",
    ]

    incomplete = [r for r in analysis.runs if not r.complete]
    if incomplete:
        parts.append(
            f"> ⚠️ {len(incomplete)} run(s) have no `run_end` event and are "
            f"excluded from outcome tables: "
            + ", ".join(f"`{r.subject}/{r.arm}`" for r in incomplete[:10]) + "\n")

    # ---------------------------------------------------------------- sample
    if registry is not None:
        comp = registry.composition()
        parts.append("\n## 1. Sample composition\n")
        parts.append("_Answers: Reviewer A ('convenience sample', 'how do you ensure "
                     "the sample is not biased'), Reviewer B ('skewed to ML research "
                     "artifacts'), Reviewer C ('32 is small')._\n")
        parts.append(_table(
            ["Metric", "Value"],
            [["Candidates screened", comp["screened"]],
             ["Included", comp["included"]],
             ["Rejected", comp["rejected"]],
             ["ML-research artifacts", comp["ml_research"]],
             ["Non-ML projects", f"{comp['non_ml']} ({100 * comp['non_ml_fraction']:.0f}%)"],
             ["Meets non-ML target", "yes" if comp["meets_non_ml_target"] else "**no**"]]))
        parts.append("\n### 1.1 Domains\n")
        parts.append(_table(["Domain", "Subjects"],
                            [[k, v] for k, v in comp["domains"].items()]))
        parts.append("\n### 1.2 Rejection log (why candidates were excluded)\n")
        parts.append(_table(["Reason", "Count"],
                            [[k, v] for k, v in comp["rejections"].items()]))
        parts.append("\n### 1.3 Validation-script provenance\n")
        parts.append("_Answers: Reviewer C ('reliance on manually authored validation "
                     "scripts introduces potential bias')._\n")
        parts.append(_table(["Provenance", "Subjects"],
                            [[k, v] for k, v in comp["validation_provenance"].items()]))
        problems = registry.audit()
        if problems:
            parts.append("\n**Protocol violations to resolve before reporting:**\n")
            parts += [f"- {p}\n" for p in problems]

    # -------------------------------------------------------------- outcomes
    parts.append("\n## 2. Outcomes by arm\n")
    parts.append("_Answers: RQ1/RQ2, and Reviewer C's request to separate fully "
                 "converged runs from truncated ones. Note that `kept_baseline` and "
                 "`failed` are distinct: v1 could report neither._\n")
    rows = []
    for row in analysis.outcome_table():
        rows.append([row["arm"], row["n"], row["modernized_n"],
                     row[Outcome.KEPT_BASELINE.value],
                     row[Outcome.TRUNCATED_PASSES.value] + row[Outcome.TRUNCATED_TIMEOUT.value],
                     row[Outcome.FAILED.value], row[Outcome.ERROR.value],
                     row["any_progress_n"], row["non_regressing"]])
    parts.append(_table(
        ["Arm", "n", "Modernized", "Kept baseline", "Truncated", "Failed", "Error",
         "Any progress", "No regression (95% CI)"], rows))
    parts.append("\n_Columns 3-7 partition each arm's runs. 'Any progress' counts "
                 "modernized plus truncated runs that had already committed at least "
                 "one validated upgrade -- a truncated run is an incompleteness bound, "
                 "not a failure, and not a completed modernization either._\n")

    # --------------------------------------------------------- head-to-heads
    parts.append("\n## 3. Paired comparisons\n")
    parts.append("_Answers: Reviewer C ('lack of statistical significance testing'), "
                 "Reviewer B ('weak baselines'). Exact McNemar over shared subjects; "
                 "`b` favours the first arm, `c` the second._\n")
    comparisons = [
        (full_arm, "pip_upgrade"), (full_arm, "uv_upgrade"),
        (full_arm, "install_only"), (full_arm, "dependabot"),
        (full_arm, "aura_llm"), (full_arm, "aura_linear"),
        (full_arm, "aura_linear_greedy"),
    ]
    rows = []
    for a, b in comparisons:
        if a not in analysis.by_arm or b not in analysis.by_arm:
            continue
        mod = analysis.compare(a, b, lambda r: r.modernized)
        reg = analysis.compare(a, b, lambda r: r.non_regressing)
        rows.append([f"{a} vs {b}", mod["n_shared"], len(mod["a_only"]),
                     len(mod["b_only"]), mod["both"], mod["neither"],
                     _fmt(mod["p_value"], 4), _fmt(reg["p_value"], 4)])
    parts.append(_table(["Comparison", "n", "A only", "B only", "Both", "Neither",
                         "p (modernized)", "p (no regression)"], rows))
    parts.append("\n_Two predicates are reported because they answer different "
                 "questions. 'Modernized' asks which arm advanced the environment; "
                 "'no regression' asks which arm avoided breaking a working project -- "
                 "an outcome a one-shot upgrade cannot achieve by construction, since "
                 "it has no way to decline an upgrade._\n")

    # -------------------------------------------------------------- ablation
    parts.append("\n## 4. Isolated ablation\n")
    parts.append("_Answers: Reviewer A ('testing resolution strategies separately "
                 "rather than stacked') and Reviewer B ('the ablation is incomplete "
                 "because it does not include an LLM-only baseline'). All seven "
                 "non-empty tier subsets are run independently; `aura_llm` is the "
                 "standalone LLM baseline._\n")
    ablation = [a for a in analysis.arms if a.startswith("aura_") and
                a.replace("aura_", "").replace("_", "+") in {
                    "linear", "greedy", "llm", "linear+greedy", "linear+llm",
                    "greedy+llm", "linear+greedy+llm"}]
    rows = []
    for arm in sorted(ablation, key=lambda a: (len(a.split("_")), a)):
        runs = analysis.by_arm[arm]
        mod = sum(1 for r in runs if r.modernized)
        nonreg = sum(1 for r in runs if r.non_regressing)
        cost = sum(r.cost_usd for r in runs)
        rows.append([arm, " + ".join(arm.replace("aura_", "").split("_")), len(runs),
                     str(wilson_interval(mod, len(runs))) if runs else "-",
                     str(wilson_interval(nonreg, len(runs))) if runs else "-",
                     f"${cost:.3f}",
                     sum(int(r.counters.get("attempts", 0)) for r in runs)])
    parts.append(_table(["Arm", "Tiers", "n", "Modernized (95% CI)",
                         "No regression (95% CI)", "LLM cost", "Install cycles"], rows))

    # --------------------------------------------------------- strategy usage
    usage = analysis.strategy_usage(full_arm)
    parts.append("\n## 5. Strategy attribution within the full system\n")
    parts.append("_Answers: RQ3. Counts subjects in which a tier produced at least one "
                 "**accepted** configuration, so the tiers overlap and do not partition "
                 "the sample._\n")
    parts.append(_table(["Tier", "Subjects with an accepted update"],
                        [[k, v] for k, v in sorted(usage["per_tier"].items())]))
    parts.append("\n**Tier combinations per subject:**\n")
    parts.append(_table(["Combination", "Subjects"],
                        [[k, v] for k, v in usage["combinations"].items()]))

    # ------------------------------------------------------ component quality
    parts.append("\n## 6. Component-level quality\n")
    parts.append("_Answers: Reviewer B ('the evaluation is almost entirely end-to-end "
                 "and does not adequately validate the technical components ... does "
                 "not measure the precision of the extracted conflict packages ... does "
                 "not report first-attempt success rate or number of refinement "
                 "rounds')._\n")

    conflict = analysis.conflict_precision()
    parts.append("\n### 6.1 Conflict extraction (Greedy Co-Resolution input)\n")
    parts.append(_table(["Metric", "Value"], [
        ["Extraction events", conflict["events"]],
        ["Names extracted (total)", conflict["names_extracted"]],
        ["Names present in the environment", conflict["names_in_environment"]],
        ["Micro-averaged precision (name is installed; weak test)",
         _fmt(conflict["micro_precision"], 3)],
        ["Events where pip printed a conflict report",
         conflict.get("events_with_pip_report", 0)],
        ["**Precision against pip's own conflict report**",
         _describe_cell(conflict.get("precision_vs_pip_report", {}), 3)],
        ["Per-event precision", _describe_cell(conflict["per_event"], 3)]]))

    llm = analysis.llm_quality()
    parts.append("\n### 6.2 LLM planning quality\n")
    parts.append(_table(["Metric", "Value"], [
        ["Escalations to the LLM tier", llm["escalations"]],
        ["Escalations resolved", llm["resolved"]],
        ["First-attempt success", llm["first_attempt_success"]],
        ["Refinement iterations", _describe_cell(llm["iterations_per_escalation"], 2)],
        ["Versions proposed", llm["versions_proposed"]],
        ["Versions rejected by validation", llm["versions_rejected"]],
        ["Wrong-version rate", _fmt(llm["wrong_version_rate"], 4)],
        ["**Hallucination rate** (version not on the index)",
         _fmt(llm["hallucination_rate"], 4)],
        ["Early exits (declared unresolvable)", llm["early_exits"]]]))
    if llm["rejection_reasons"]:
        parts.append("\n**Why proposed versions were rejected:**\n")
        parts.append(_table(["Reason", "Count"],
                            [[k, v] for k, v in llm["rejection_reasons"].items()]))

    # -------------------------------------------------- per-package advance
    advance = analysis.per_package_advance()
    if advance:
        parts.append("\n## 6b. Per-dependency advance rate\n")
        parts.append(
            "The repository-level outcome hides most of what happened: a subject "
            "with twelve outdated pins scores the same whether an arm moved one "
            "of them or all twelve. The denominator below is each subject's own "
            "outdated pins, computed identically for every arm.\n\n"
            "This is also the only unit on which a per-dependency proposer can "
            "be scored fairly. Dependabot never proposes a combined change, so "
            "its repository-level result is an artefact of how one chooses to "
            "merge its pull requests; its per-package result is simply what it "
            "proposed, validated under the same oracle as everything else.\n")
        parts.append(_table(
            ["Arm", "Subjects", "Outdated pins", "Advanced", "Rate",
             "Per-subject rate"],
            [[row["arm"], row["subjects"], row["outdated"], row["advanced"],
              _fmt(row["rate"], 3), _describe_cell(row["per_subject_rate"], 3)]
             for row in sorted(advance.values(), key=lambda r: -(r["rate"] or 0))]))

        coupled = analysis.coupled_failures()
        if coupled["rejected_proposals"]:
            parts.append("\n### 6b.1 The co-resolution gap, measured\n")
            parts.append(
                "A per-dependency proposer offers each package alone. Where two "
                "packages must move together, each proposal fails on its own and "
                "the tool has no way to express the pair. Every package below "
                "was proposed alone, failed alone, and was advanced by AURA as "
                "part of a group.\n")
            parts.append(_table(["Metric", "Value"], [
                ["Proposals rejected by validation", coupled["rejected_proposals"]],
                ["**Of those, advanced by AURA**", coupled["recovered_by_aura"]]]))
            if coupled["examples"]:
                parts.append("\n**Examples:**\n")
                parts.append(_table(
                    ["Subject", "Package", "Proposed alone", "Failed as",
                     "AURA reached", "By also moving"],
                    [[e["subject"], e["package"], e["proposed"], e["error_class"],
                      e["aura_reached"], ", ".join(e["aura_moved_with"][:4]) or "-"]
                     for e in coupled["examples"]]))
            merged = [u for u in coupled["union_probe"] if u["accepted"] is not None]
            if merged:
                ok = sum(1 for u in merged if u["accepted"])
                parts.append(
                    f"\n> Applying every proposal at once succeeds on {ok}/"
                    f"{len(merged)} subjects. Recorded as context only: it is a "
                    "combination the proposer never offered, and merging all of "
                    "its pull requests is not what the tool recommends.\n")

    # ------------------------------------------- failure taxonomy & routing
    taxonomy = analysis.failure_taxonomy()
    if taxonomy:
        parts.append("\n### 6.3 Why candidates fail\n")
        parts.append(
            "Reviewer B asked what the search actually spends its time on. Each "
            "failed attempt is classified at the point of failure; the wasted "
            "minutes column is the install time spent on candidates that were "
            "then rejected.\n")
        parts.append(_table(
            ["Failure class", "Failures", "Share", "Wasted minutes", "Subjects"],
            [[r["class"], r["failures"], _fmt(r["share"], 3),
              r["wasted_minutes"], r["subjects"]] for r in taxonomy]))

    graph = analysis.graph_vs_regex()
    if graph["declared_pairs"] or graph["regex_pairs"]:
        parts.append("\n### 6.4 Declared coupling vs. extracted coupling\n")
        parts.append(
            "Reviewer A asked whether the coupling detection works. The "
            "declared-constraint graph, read from package metadata before any "
            "probe, gives something to check the error-log regex against.\n")
        parts.append(_table(["Metric", "Value"], [
            ["Pairs declared in metadata", graph["declared_pairs"]],
            ["Pairs extracted from error logs", graph["regex_pairs"]],
            ["Agreement", graph["agreement"]],
            ["Regex precision vs. declared",
             _fmt(graph["regex_precision_vs_declared"], 3)],
            ["Regex recall vs. declared",
             _fmt(graph["regex_recall_vs_declared"], 3)],
            ["Packages with unreadable metadata",
             len(graph["packages_without_metadata"])]]))
        parts.append(f"\n> {graph['note']}\n")

    for label, effect, off_label in (
            ("Failure-aware routing", analysis.routing_effect(), "routing off"),
            ("Declared-constraint graph", analysis.graph_effect(), "no graph")):
        if not effect["subjects"]:
            continue
        parts.append(f"\n### 6.5 {label}: same answer, fewer probes\n")
        parts.append(
            "The search skips work only when declared metadata proves it is "
            "wasted -- a reverse linear scan is abandoned only when every "
            "remaining candidate is forbidden by a pinned package's declared "
            "constraint. The first row is the correctness claim; the rest are "
            "the cost claim, and the second is worthless without the first.\n")
        parts.append(_table(["Metric", "Value"], [
            ["Subjects compared", effect["subjects"]],
            ["**Identical final configuration**",
             f"{effect['identical_results']}/{effect['subjects']}"],
            ["Probes saved", effect["probes_saved_total"]],
            ["Probes saved (%)", _fmt(effect["probes_saved_pct"], 3)],
            ["Wall-clock, enabled (s)", effect["seconds_on"]],
            [f"Wall-clock, {off_label} (s)", effect["seconds_off"]],
            ["Speedup", _fmt(effect["speedup"], 2)]]))
        if effect["diverged"]:
            parts.append(
                "\n**Divergent subjects — these must be explained, not averaged:**\n")
            parts.append(_table(
                ["Subject", "Enabled", off_label.capitalize()],
                [[d["subject"], str(d["routing_on"]), str(d["routing_off"])]
                 for d in effect["diverged"]]))

    batching = analysis.batching_effect()
    if batching["subjects"]:
        parts.append("\n### 6.5b Batched probing: k=1 vs k=production\n")
        parts.append(
            "Reverse Linear Scan is optimal in oracle *calls*, so the only way "
            "left to make it faster is to stop asking one question at a time. "
            "k is a resource bound computed from the release's measured wheel "
            "size and the machine's detected cores, disk and memory -- not a "
            "tuned constant, and never chosen per subject. The claim to defend "
            "is the first row; the timings are worthless without it.\n")
        parts.append(
            "Note that batching *raises* the probe count. It has to: a batch "
            "probes its whole block even where a serial scan would have stopped "
            "at the first passing member, and the winner is then confirmed "
            "alone. Reverse Linear Scan is already optimal in oracle calls, so "
            "there was never a probe saving available here -- what batching buys "
            "is overlap, and the cost is paid in probes to be repaid in time.\n")
        parts.append(_table(["Metric", "Value"], [
            ["Subjects compared", batching["subjects"]],
            ["**Identical final configuration**",
             f"{batching['identical_results']}/{batching['subjects']}"],
            ["Probes, k=production", sum(r["probes_on"] for r in batching["rows"])],
            ["Probes, k=1", sum(r["probes_off"] for r in batching["rows"])],
            ["Wall-clock, k=production (s)", batching["seconds_on"]],
            ["Wall-clock, k=1 (s)", batching["seconds_off"]],
            ["**Speedup**", _fmt(batching["speedup"], 2)],
            ["Serial confirmations of a batch winner",
             batching["guards"]["confirmations"]],
            ["Winners rejected on confirmation",
             batching["guards"]["confirmation_failures"]],
            ["Contention re-probes", batching["guards"]["rechecks"]],
            ["Versions rescued by a re-probe", batching["guards"]["rescued"]]]))
        if batching["rows"]:
            parts.append("\n**Per subject:**\n")
            parts.append(_table(
                ["Subject", "Same result", "k=prod (s)", "k=1 (s)", "Speedup"],
                [[r["subject"], "yes" if r["same_result"] else "**NO**",
                  r["seconds_on"], r["seconds_off"],
                  _fmt(r["seconds_off"] / r["seconds_on"], 2)
                  if r["seconds_on"] else "-"]
                 for r in batching["rows"]]))
        if batching["guards"]["rescued"]:
            parts.append(
                "\n> Versions rescued by a re-probe are the guard doing its job: "
                "each one failed while sharing the machine with other installs "
                "and passed on its own. Without the guard each would have been a "
                "silently worse final version.\n")
        if batching["diverged"]:
            parts.append(
                "\n**Divergent subjects — the exactness claim fails here and "
                "must be explained, not averaged:**\n")
            parts.append(_table(
                ["Subject", "k=production", "k=1"],
                [[d["subject"], str(d["routing_on"]), str(d["routing_off"])]
                 for d in batching["diverged"]]))

    coverage = analysis.oracle_coverage()
    if coverage["subjects"]:
        parts.append("\n### 6.6 How much does the oracle actually cover?\n")
        parts.append(
            "Reviewer B: *\"A passing validation script only shows that no "
            "failure was observed for the covered executions.\"* The repo figure "
            "is the familiar one; the dependency figure is the one that bears on "
            "a dependency upgrade, and the two routinely disagree -- a script can "
            "cover most of a repo while exercising a fraction of the package "
            "being upgraded.\n")
        parts.append(_table(["Scope", "Coverage across subjects"], [
            ["Repo", _describe_cell(coverage["repo_coverage"], 3)],
            ["Per dependency", _describe_cell(coverage["dependency_coverage"], 3)]]))
        parts.append("\n**Per subject:**\n")
        parts.append(_table(
            ["Subject", "Repo", "Median dependency", "Weakest dependency",
             "Compiled (understated)"],
            [[r["subject"], _fmt(r["repo_pct"], 3),
              _fmt(r.get("median_dependency_pct"), 3),
              (f"{r['weakest']} ({_fmt(r['dependencies'].get(r['weakest']), 3)})"
               if r.get("weakest") else "-"),
              ", ".join(r.get("compiled") or []) or "-"]
             for r in coverage["rows"]]))
        parts.append(f"\n> {coverage['note']}\n")
        if coverage["coverage_drops"]:
            parts.append(
                "\n**Coverage dropped after the upgrade.** Less code ran than at "
                "baseline while the tests still passed and the numbers still "
                "matched, which usually means an optional import began failing "
                "and a fallback took over silently. Nothing else in the pipeline "
                "detects this.\n")
            parts.append(_table(
                ["Subject", "Unit", "Before", "After", "Delta"],
                [[d["subject"], d["unit"], _fmt(d["before"], 3),
                  _fmt(d["after"], 3), _fmt(d["delta"], 3)]
                 for d in coverage["coverage_drops"]]))

    # -------------------------------------------------------- cost & runtime
    parts.append("\n## 7. Cost and runtime\n")
    parts.append("_Answers: Reviewer A ('what fraction of projects requires LLM "
                 "intervention, and what is the distribution of costs'), Reviewer C "
                 "('execution time is high'). Costs are USD, from measured token "
                 "counts rather than list-price estimates._\n")
    rows = []
    for row in analysis.cost_and_runtime():
        rows.append([row["arm"], _describe_cell(row["runtime_s"], 0),
                     _describe_cell(row["attempts"], 1),
                     _describe_cell(row["llm_calls"], 1),
                     _describe_cell(row["tokens"], 0),
                     f"${row['total_cost_usd']:.3f}"])
    parts.append(_table(["Arm", "Wall clock (s)", "Install cycles", "LLM calls",
                         "Tokens", "Total cost"], rows))

    # ------------------------------------------------ modernization magnitude
    parts.append("\n## 8. How much newer? (modernization magnitude)\n")
    parts.append("_Answers: Reviewer C ('the term modernized is used loosely; it is "
                 "not defined quantitatively ... a metric like average version age "
                 "reduction would strengthen the claims') and Reviewer A on optimality._\n")
    rows = [[row["arm"], _describe_cell(row["mean_releases_recovered"], 2),
             _describe_cell(row["pct_at_latest_after"], 1)]
            for row in analysis.modernization_magnitude()]
    parts.append(_table(["Arm", "Releases recovered per package",
                         "% of packages at latest, after"], rows))

    # ------------------------------------------------------ order sensitivity
    order_arms = [a for a in analysis.arms if a.startswith("aura_order_")]
    if order_arms:
        parts.append("\n## 9. Order sensitivity\n")
        parts.append("_Answers: Reviewer A ('the method is sensitive to the order of "
                     "selection, but the evaluation does not shed light on the "
                     "magnitude of this problem')._\n")
        sens = analysis.order_sensitivity([full_arm] + order_arms)
        parts.append(_table(["Metric", "Value"], [
            ["Orderings compared", len(sens["arms"])],
            ["Subjects", sens["subjects"]],
            ["Identical final configuration", sens["identical_final_config"]],
            ["Divergent final configuration", sens["divergent_final_config"]],
            ["Divergent outcome class", len(sens["divergent_outcome"])]]))
        if sens["divergent_subjects"]:
            parts.append("\nSubjects whose result depends on ordering: "
                         + ", ".join(f"`{s}`" for s in sens["divergent_subjects"]) + "\n")

    # ---------------------------------------------------- backend comparison
    backend_arms = [a for a in analysis.arms if a.startswith("aura_model_")]
    if backend_arms:
        parts.append("\n## 10. LLM backend sensitivity\n")
        parts.append("_Answers: Reviewer B ('the paper does not evaluate sensitivity "
                     "to the LLM backend ... should evaluate multiple backend models "
                     "and measure how effectiveness and cost vary')._\n")
        rows = []
        for arm in sorted(backend_arms):
            runs = analysis.by_arm[arm]
            quality = analysis.llm_quality([arm])
            mod = sum(1 for r in runs if r.modernized)
            rows.append([arm.replace("aura_model_", ""), len(runs),
                         str(wilson_interval(mod, len(runs))) if runs else "-",
                         quality["first_attempt_success"],
                         _fmt(quality["hallucination_rate"], 4),
                         f"${sum(r.cost_usd for r in runs):.3f}"])
        parts.append(_table(["Backend", "n", "Modernized (95% CI)",
                             "First-attempt success", "Hallucination rate", "Cost"], rows))

    # ---------------------------------------------------------- oracle strength
    from evaluation.oracle import observed_discrimination
    disc = observed_discrimination(analysis, full_arm)
    if disc["subjects"]:
        parts.append("\n## 9b. Oracle strength\n")
        parts.append("_Answers: Reviewer C ('the reliance on manually authored "
                     "validation scripts introduces potential bias') and Reviewer B "
                     "('a passing validation script only shows that no failure was "
                     "observed'). Rather than arguing that the oracles are adequate, "
                     "we measure how much work they did._\n")
        parts.append(_table(["Metric", "Value"], [
            ["Subjects", disc["subjects"]],
            ["Oracle rejected at least one installable candidate",
             disc["oracle_fired_at_least_once"]],
            ["Subjects whose oracle never fired", len(disc["silent_oracles"])]]))
        if disc["silent_oracles"]:
            parts.append("\n> ⚠️ For these subjects every rejection came from the "
                         "resolver, not from execution: "
                         + ", ".join(f"`{s}`" for s in disc["silent_oracles"])
                         + ". Their results rest on installability alone and should "
                         "be reported as such.\n")
        parts.append("\n_Mutation-detection rates (uninstall the most-used package, "
                     "pin it to its oldest release, jump it to the newest major) are "
                     "produced separately by `python -m evaluation.cli oracle` and "
                     "belong beside this table._\n")

    # ------------------------------------------ unverified-regime comparison
    fp = analysis.structural_false_positives()
    if fp["checked"]:
        parts.append("\n## 10b. The unverified regime (UPCY stand-in)\n")
        parts.append("_Answers: Reviewer A ('it would be valuable to know how often "
                     "UPCY produces a false-positive configuration that AURA is able "
                     "to improve upon'). UpCy is a Java/Maven tool and cannot be run "
                     "on Python subjects, so we reproduce its **verification regime**: "
                     "the `install_only` arm accepts whatever the resolver accepts and "
                     "never runs tests during its search. Its final answer is validated "
                     "afterwards, with that result never fed back._\n")
        parts.append(_table(["Metric", "Value"], [
            ["Configurations checked after the fact", fp["checked"]],
            ["Broken despite resolving cleanly", fp["broken"]],
            ["False-positive rate (95% CI)", fp["false_positive_rate"]]]))
        if fp["broken_subjects"]:
            parts.append("\nSubjects where a structurally valid configuration fails "
                         "the project's own tests: "
                         + ", ".join(f"`{s}`" for s in fp["broken_subjects"]) + "\n")

    # ----------------------------------------------------- behavioural drift
    if "aura_tests_only" in analysis.by_arm:
        parts.append("\n## 11. Behavioural preservation\n")
        parts.append("_Answers: Reviewer B ('a passing validation script only shows "
                     "that no failure was observed for the covered executions ... "
                     "dependency upgrades can subtly change results even when smoke "
                     "tests pass')._\n")
        drift = analysis.behavioural_drift(full_arm, "aura_tests_only")
        parts.append(_table(["Metric", "Value"], [
            ["Drift events detected", drift["drift_events"]],
            ["Subjects with drift", len(drift["subjects_with_drift"])],
            ["Accepted only under an exit-code oracle", len(drift["lenient_only"])]]))
        if drift["lenient_only"]:
            parts.append("\n" + drift["note"] + "\n\nAffected: "
                         + ", ".join(f"`{s}`" for s in drift["lenient_only"]) + "\n")

    # ------------------------------------------------------- per-subject table
    parts.append("\n## 12. Per-subject outcomes\n")
    parts.append("_Answers: Reviewer B ('the main paper does not include a per-subject "
                 "table of outcomes and validation details')._\n")
    key_arms = [a for a in ["baseline", "pip_upgrade", "uv_upgrade", "install_only",
                            "dependabot", full_arm] if a in analysis.by_arm]
    rows = []
    for subject in analysis.subjects:
        row = [subject]
        for arm in key_arms:
            run = analysis.index.get((subject, arm))
            row.append(run.outcome if run else "-")
        full = analysis.index.get((subject, full_arm))
        row.append(len(full.updated) if full else "-")
        row.append(_fmt(full.age_delta(), 2) if full else "-")
        row.append(f"{full.wall_clock_s / 60:.0f}m" if full else "-")
        rows.append(row)
    parts.append(_table(["Subject"] + key_arms + ["Pkgs updated", "Releases recovered",
                                                  "Runtime"], rows))

    # ------------------------------------------------------------ environment
    parts.append("\n## 13. Execution environment\n")
    parts.append("_Answers: Reviewer B ('the paper does not clearly specify whether "
                 "the LLM-guided component has access to external tools or web "
                 "retrieval ... should explicitly state the LLM execution environment')._\n")
    envs = {json.dumps(r.env, sort_keys=True) for r in analysis.runs if r.env}
    models = sorted({r.config.get("llm_model", "") for r in analysis.runs
                     if r.config.get("llm_model") not in ("", "none")})
    temps = sorted({r.config.get("llm_temperature") for r in analysis.runs})
    parts.append(_table(["Property", "Value"], [
        ["LLM backends used", ", ".join(models) or "none"],
        ["Temperature", ", ".join(str(t) for t in temps)],
        ["Tools / function calling", "disabled (no tools passed to any backend)"],
        ["Web or retrieval access", "none (prompt-only; see aura/llm.py)"],
        ["Distinct execution environments", len(envs)],
        ["Python versions", ", ".join(sorted({r.env.get("python", "?")
                                              for r in analysis.runs if r.env}))],
    ]))
    parts.append("\n> Contamination note: the subjects are public repositories and "
                 "their dependency histories are plausibly present in every backend's "
                 "training data. We therefore report the deterministic tiers' "
                 "contribution separately (Section 4) and compare open-weight backends "
                 "against the hosted one (Section 10), rather than claiming the LLM "
                 "results are contamination-free.\n")

    return "".join(p if p.endswith("\n") else p + "\n" for p in parts)


def write_report(results_dir: str | Path, out_path: str | Path,
                 registry_path: str | Path | None = None, **kwargs) -> Path:
    from evaluation.report import load_runs
    from evaluation.subjects import Registry

    runs = load_runs(results_dir)
    registry = None
    if registry_path and Path(registry_path).exists():
        registry = Registry.load(registry_path)
    text = render(Analysis(runs), registry, **kwargs)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out
