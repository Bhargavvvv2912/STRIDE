"""End-to-end tests for the analysis and report pipeline."""

import pytest

from aura.telemetry import LLMUsage, Outcome, RunConfig, RunLog, Tier
from evaluation.report import Analysis, load_runs, summarize_log
from evaluation.render import render


def write_run(root, subject, arm, outcome, *, updated=None, tiers_accepted=(),
              llm=None, drift=False, conflicts=None, age=(5.0, 1.0), wall=600.0):
    cfg = RunConfig(repo=subject, arm=arm, llm_model="gemini-2.5-flash")
    log = RunLog(root / subject / f"{arm}.jsonl", cfg)
    for tier in (tiers_accepted or []):
        log.attempt(package="torch", tier=tier, plan={"torch": "2.0.0"},
                    install_ok=True, validate_ok=True, duration_s=30.0, accepted=True)
    log.attempt(package="torch", tier=Tier.LINEAR, plan={"torch": "2.1.0"},
                install_ok=True, validate_ok=False, duration_s=25.0)
    for extracted, installed in (conflicts or []):
        log.conflict_detection(package="torch", extracted=extracted, installed=installed)
    for call in (llm or []):
        log.llm_call(
            LLMUsage(model="gemini-2.5-flash", prompt_tokens=8000,
                     completion_tokens=400, latency_s=2.0).finalize(),
            purpose="co_resolution", package="torch", iteration=call["iteration"],
            proposed=call.get("proposed", []), rejected=call.get("rejected", []))
        if call.get("accepted"):
            log.emit("note", note="llm_plan_accepted", package="torch",
                     iteration=call["iteration"])
    if drift:
        log.behavior_check(baseline={"accuracy": 0.91}, candidate={"accuracy": 0.62},
                           drifted=["accuracy"], tolerance=0.01, verdict="drift")
    log._counters.setdefault("attempts", 2)
    log.emit("run_end_marker_placeholder")  # keeps ordering realistic
    log.finish(outcome, updated=updated or {},
               version_age={"before": {"mean_releases_behind": age[0], "at_latest_pct": 0.0},
                            "after": {"mean_releases_behind": age[1], "at_latest_pct": 60.0}})
    # wall_clock is derived from real time in the log; patch it for determinism
    return root / subject / f"{arm}.jsonl"


@pytest.fixture
def results(tmp_path):
    root = tmp_path / "results"
    # Three subjects. pip breaks on r1 and r2; AURA recovers both.
    write_run(root, "r1", "pip_upgrade", Outcome.FAILED)
    write_run(root, "r2", "pip_upgrade", Outcome.FAILED)
    write_run(root, "r3", "pip_upgrade", Outcome.MODERNIZED, updated={"numpy": ["1.0", "2.0"]})

    write_run(root, "r1", "aura_linear", Outcome.KEPT_BASELINE)
    write_run(root, "r2", "aura_linear", Outcome.MODERNIZED,
              updated={"torch": ["1.0", "1.1"]}, tiers_accepted=[Tier.LINEAR])
    write_run(root, "r3", "aura_linear", Outcome.MODERNIZED,
              updated={"numpy": ["1.0", "2.0"]}, tiers_accepted=[Tier.LINEAR])

    write_run(root, "r1", "aura_llm", Outcome.MODERNIZED,
              updated={"torch": ["1.0", "1.2"]}, tiers_accepted=[Tier.LLM],
              llm=[{"iteration": 1, "proposed": ["torch==1.2"],
                    "rejected": [{"req": "torch==9.9", "reason": "not_on_index"}],
                    "accepted": True}])
    write_run(root, "r2", "aura_llm", Outcome.KEPT_BASELINE)
    write_run(root, "r3", "aura_llm", Outcome.KEPT_BASELINE)

    for subject, outcome, tiers in (
        ("r1", Outcome.MODERNIZED, [Tier.LINEAR, Tier.LLM]),
        ("r2", Outcome.MODERNIZED, [Tier.LINEAR, Tier.GREEDY]),
        ("r3", Outcome.MODERNIZED, [Tier.LINEAR]),
    ):
        write_run(root, subject, "aura_linear_greedy_llm", outcome,
                  updated={"torch": ["1.0", "2.0"]}, tiers_accepted=tiers,
                  conflicts=[(["torchvision", "numpy", "notreal"],
                              ["torch", "torchvision", "numpy"])],
                  llm=[{"iteration": 1, "proposed": ["torch==2.0"],
                        "rejected": [{"req": "torch==9.9", "reason": "not_on_index"}],
                        "accepted": True}] if Tier.LLM in tiers else None,
                  drift=(subject == "r1"))

    write_run(root, "r1", "aura_tests_only", Outcome.MODERNIZED,
              updated={"torch": ["1.0", "2.0"]}, tiers_accepted=[Tier.LINEAR])
    return root


def test_load_runs_finds_every_arm(results):
    runs = load_runs(results)
    assert len(runs) == 13
    assert {r.arm for r in runs} == {
        "pip_upgrade", "aura_linear", "aura_llm", "aura_linear_greedy_llm",
        "aura_tests_only"}


def test_outcome_table_separates_kept_baseline_from_failed(results):
    table = {r["arm"]: r for r in Analysis(load_runs(results)).outcome_table()}
    assert table["pip_upgrade"]["failed"] == 2
    assert table["aura_linear"]["kept_baseline"] == 1
    assert table["aura_linear"]["failed"] == 0
    assert table["aura_linear_greedy_llm"]["modernized_n"] == 3


def test_paired_comparison_reports_significance(results):
    analysis = Analysis(load_runs(results))
    c = analysis.compare("aura_linear_greedy_llm", "pip_upgrade")
    assert c["n_shared"] == 3
    assert sorted(c["a_only"]) == ["r1", "r2"]     # AURA recovers what pip breaks
    assert c["b_only"] == []
    assert 0.0 < c["p_value"] <= 1.0


def test_conflict_precision_is_measured(results):
    conflict = Analysis(load_runs(results)).conflict_precision()
    # 3 extracted, 2 real, per event
    assert conflict["micro_precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert conflict["events"] == 3


def test_llm_quality_reports_a_hallucination_rate(results):
    quality = Analysis(load_runs(results)).llm_quality()
    assert quality["versions_proposed"] > 0
    assert quality["hallucination_rate"] > 0
    assert quality["rejection_reasons"]["not_on_index"] >= 1
    assert "first_attempt_success" in quality


def test_strategy_usage_counts_overlapping_tiers(results):
    usage = Analysis(load_runs(results)).strategy_usage("aura_linear_greedy_llm")
    assert usage["per_tier"]["linear"] == 3
    assert usage["per_tier"]["greedy"] == 1
    assert usage["per_tier"]["llm"] == 1
    # Tiers overlap: counts must not sum to the number of subjects
    assert sum(usage["per_tier"].values()) > usage["subjects"]


def test_modernization_magnitude_is_quantified(results):
    rows = {r["arm"]: r for r in Analysis(load_runs(results)).modernization_magnitude()}
    assert rows["aura_linear_greedy_llm"]["mean_releases_recovered"]["mean"] == 4.0


def test_behavioural_drift_is_surfaced(results):
    drift = Analysis(load_runs(results)).behavioural_drift(
        "aura_linear_greedy_llm", "aura_tests_only")
    assert drift["drift_events"] == 1
    assert drift["subjects_with_drift"] == ["r1"]


def test_render_produces_every_required_section(results):
    text = render(Analysis(load_runs(results)))
    for heading in ["## 2. Outcomes by arm", "## 3. Paired comparisons",
                    "## 4. Isolated ablation",
                    "## 5. Strategy attribution", "## 6. Component-level quality",
                    "## 7. Cost and runtime", "## 8. How much newer",
                    "## 11. Behavioural preservation", "## 12. Per-subject outcomes",
                    "## 13. Execution environment"]:
        assert heading in text, f"missing section: {heading}"


def test_render_states_the_llm_environment(results):
    text = render(Analysis(load_runs(results)))
    assert "Temperature" in text and "0.0" in text
    assert "no tools passed to any backend" in text
    assert "Contamination note" in text


def test_render_flags_incomplete_runs(tmp_path):
    (tmp_path / "r9").mkdir(parents=True)
    (tmp_path / "r9" / "aura_full_qwen14b.jsonl").write_text(
        '{"run_id":"x","repo":"r9","arm":"aura_full_qwen14b","event":"run_start","config":{},"env":{}}\n')
    text = render(Analysis(load_runs(tmp_path)))
    assert "have no `run_end` event" in text


def test_structural_false_positive_rate_is_measured(tmp_path):
    """install_only accepts on resolution alone; post-hoc validation exposes the cost."""
    from aura.telemetry import Outcome, RunConfig, RunLog
    root = tmp_path / "results"
    for subject, post_hoc_ok in (("r1", True), ("r2", False), ("r3", False)):
        log = RunLog(root / subject / "install_only.jsonl",
                     RunConfig(repo=subject, arm="install_only"))
        log.validation(phase="post_hoc", passed=post_hoc_ok,
                       failing=[] if post_hoc_ok else ["tests/test_x.py::test_y"],
                       new_failures=[], mode="post_hoc_unverified_regime")
        log.finish(Outcome.MODERNIZED, updated={"torch": ["1.0", "2.0"]})

    fp = Analysis(load_runs(root)).structural_false_positives()
    assert fp["checked"] == 3 and fp["broken"] == 2
    assert fp["broken_subjects"] == ["r2", "r3"]
    assert "66.7%" in fp["false_positive_rate"]


def test_outcome_columns_partition_the_runs(tmp_path):
    """A truncated run must not also be counted as modernized."""
    from aura.telemetry import Outcome
    root = tmp_path / "results"
    write_run(root, "r1", "aura_full_qwen14b", Outcome.TRUNCATED_TIMEOUT,
              updated={"torch": ["1.0", "2.0"]})
    write_run(root, "r2", "aura_full_qwen14b", Outcome.MODERNIZED,
              updated={"numpy": ["1.0", "2.0"]})
    row = Analysis(load_runs(root)).outcome_table()[0]
    assert row["modernized_n"] == 1                      # only the completed run
    assert row["truncated_timeout"] == 1
    assert row["any_progress_n"] == 2                    # union, stated separately
    assert row["modernized_n"] + row["truncated_timeout"] == row["n"]


# --------------------------------------- failure taxonomy, graph and routing

def write_cost_run(root, subject, arm, *, updated, attempts, wall, counters,
                   graph_couples=(), regex_pairs=(), unknown=()):
    """A run carrying the routing/graph telemetry the cost sections read."""
    cfg = RunConfig(repo=subject, arm=arm)
    log = RunLog(root / subject / f"{arm}.jsonl", cfg)
    if graph_couples or unknown:
        log.emit("constraint_graph", edges=len(graph_couples),
                 couples=[sorted(p) for p in graph_couples], unknown=list(unknown))
    for cls, duration in attempts:
        log.attempt(package="torch", tier=Tier.LINEAR, plan={"torch": "2.0.0"},
                    install_ok=cls not in ("resolution_conflict",),
                    validate_ok=False, duration_s=duration, accepted=False,
                    error_class=cls)
    for pair in regex_pairs:
        log.conflict_detection(package=pair[0], extracted=[pair[1]],
                               installed=list(pair))
    for key, value in counters.items():
        log.bump(key, value)
    log.finish(Outcome.MODERNIZED, updated=updated)
    return root / subject / f"{arm}.jsonl"


@pytest.fixture
def cost_results(tmp_path):
    root = tmp_path / "cost"
    for subject in ("s1", "s2"):
        write_cost_run(
            root, subject, "aura_full_qwen14b",
            updated={"torch": ["1.12.0", "2.0.0"]},
            attempts=[("resolution_conflict", 60.0)],
            wall=100.0,
            counters={"routing_skipped_linear": 1, "probes_saved": 3,
                      "wheel_only_rescued": 1},
            graph_couples=[("torchvision", "torch")],
            regex_pairs=[("torch", "torchvision")])
        write_cost_run(
            root, subject, "aura_no_routing",
            updated={"torch": ["1.12.0", "2.0.0"]},
            attempts=[("resolution_conflict", 60.0)] * 4,
            wall=400.0, counters={},
            graph_couples=[("torchvision", "torch")],
            regex_pairs=[("torch", "torchvision")])
    return Analysis(load_runs(root))


def test_failure_taxonomy_reports_what_the_search_wasted_time_on(cost_results):
    rows = cost_results.failure_taxonomy()
    conflict = next(r for r in rows if r["class"] == "resolution_conflict")
    assert conflict["failures"] == 10          # 2 subjects x (1 + 4) attempts
    assert conflict["wasted_minutes"] == 10.0  # 10 x 60s
    assert conflict["share"] == 1.0


def test_routing_effect_separates_the_answer_from_the_cost(cost_results):
    effect = cost_results.routing_effect()
    assert effect["subjects"] == 2
    assert effect["identical_results"] == 2     # the correctness claim
    assert effect["diverged"] == []
    assert effect["probes_saved_total"] == 6    # (4-1) attempts per subject
    assert effect["rows"][0]["skipped_scans"] == 1


def test_a_routing_divergence_is_surfaced_not_averaged(tmp_path):
    """A faster search that returns something else is a different search."""
    root = tmp_path / "diverged"
    write_cost_run(root, "s1", "aura_full_qwen14b",
                   updated={"torch": ["1.12.0", "1.13.0"]},
                   attempts=[("resolution_conflict", 10.0)], wall=50.0,
                   counters={})
    write_cost_run(root, "s1", "aura_no_routing",
                   updated={"torch": ["1.12.0", "2.0.0"]},
                   attempts=[("resolution_conflict", 10.0)] * 5, wall=90.0,
                   counters={})
    effect = Analysis(load_runs(root)).routing_effect()
    assert effect["identical_results"] == 0
    assert effect["diverged"][0]["subject"] == "s1"


def test_graph_gives_the_regex_something_to_be_scored_against(cost_results):
    quality = cost_results.graph_vs_regex()
    assert quality["declared_pairs"] == 2       # one pair per subject
    assert quality["regex_pairs"] == 2
    assert quality["regex_precision_vs_declared"] == 1.0
    assert quality["regex_recall_vs_declared"] == 1.0


def test_cost_sections_render(cost_results):
    text = render(cost_results)
    assert "6.3 Why candidates fail" in text
    assert "6.4 Declared coupling vs. extracted coupling" in text
    assert "6.5 Failure-aware routing" in text
    assert "Identical final configuration" in text


# ------------------------------------------------------------ oracle coverage

def write_coverage_run(root, subject, *, repo_pct, deps, dropped=()):
    cfg = RunConfig(repo=subject, arm="aura_coverage")
    log = RunLog(root / subject / "aura_coverage.jsonl", cfg)
    log.emit("coverage", phase="baseline", ok=True, reason="",
             repo={"name": "repo", "statements": 100,
                   "covered": int(repo_pct * 100), "files": 3,
                   "compiled": False, "pct": repo_pct},
             dependencies={n: {"name": n, "statements": 1000,
                               "covered": int(p * 1000), "files": 5,
                               "compiled": n in ("numpy", "torch"), "pct": p}
                           for n, p in deps.items()})
    if dropped:
        log.emit("coverage", phase="delta", dropped=list(dropped),
                 repo_delta=0.0, dependency_deltas={})
    log.finish(Outcome.MODERNIZED, updated={"numpy": ["1.0", "2.0"]})


def test_coverage_separates_repo_evidence_from_dependency_evidence(tmp_path):
    """The headline finding: 90% of the repo, a third of the dependency."""
    root = tmp_path / "cov"
    write_coverage_run(root, "s1", repo_pct=0.90, deps={"click": 0.32})
    write_coverage_run(root, "s2", repo_pct=0.55, deps={"numpy": 0.08,
                                                        "requests": 0.61})
    report = Analysis(load_runs(root)).oracle_coverage()

    assert report["subjects"] == 2
    s1 = next(r for r in report["rows"] if r["subject"] == "s1")
    assert s1["repo_pct"] == 0.90 and s1["dependencies"]["click"] == 0.32
    s2 = next(r for r in report["rows"] if r["subject"] == "s2")
    assert s2["weakest"] == "numpy"
    assert s2["compiled"] == ["numpy"]      # flagged: line coverage understates


def test_a_coverage_drop_reaches_the_report(tmp_path):
    root = tmp_path / "cov2"
    write_coverage_run(root, "s1", repo_pct=0.9, deps={"numpy": 0.5},
                       dropped=[{"unit": "numpy", "before": 0.5,
                                 "after": 0.1, "delta": -0.4}])
    report = Analysis(load_runs(root)).oracle_coverage()
    assert report["coverage_drops"][0]["unit"] == "numpy"


def test_coverage_section_renders(tmp_path):
    root = tmp_path / "cov3"
    write_coverage_run(root, "s1", repo_pct=0.90, deps={"click": 0.32},
                       dropped=[{"unit": "click", "before": 0.32,
                                 "after": 0.10, "delta": -0.22}])
    text = render(Analysis(load_runs(root)))
    assert "6.6 How much does the oracle actually cover?" in text
    assert "Coverage dropped after the upgrade" in text


# ---------------------------------------------------------- batching effect

def write_batch_run(root, subject, arm, *, updated, attempts, counters,
                    plans=(), wall=None):
    cfg = RunConfig(repo=subject, arm=arm)
    log = RunLog(root / subject / f"{arm}.jsonl", cfg)
    for package, k in plans:
        log.emit("batch", phase="plan", package=package, k=k,
                 bucket="large", binding_constraint="cpu")
    for _ in range(attempts):
        log.attempt(package="torch", tier=Tier.LINEAR, plan={"torch": "2.0.0"},
                    install_ok=True, validate_ok=False, duration_s=10.0,
                    accepted=False, error_class="validation_failed")
    for key, value in counters.items():
        log.bump(key, value)
    log.finish(Outcome.MODERNIZED, updated=updated)


@pytest.fixture
def batch_results(tmp_path):
    root = tmp_path / "batch"
    for subject in ("s1", "s2"):
        write_batch_run(root, subject, "aura_full_qwen14b",
                        updated={"torch": ["1.12.0", "2.0.0"]}, attempts=2,
                        plans=[("torch", 6)],
                        counters={"batch_confirmations": 1,
                                  "contention_rechecks": 2,
                                  "contention_rescued": 1})
        write_batch_run(root, subject, "aura_serial",
                        updated={"torch": ["1.12.0", "2.0.0"]}, attempts=8,
                        counters={})
    return Analysis(load_runs(root))


def test_batching_reports_the_exactness_claim_before_the_speedup(batch_results):
    effect = batch_results.batching_effect()
    assert effect["identical_results"] == 2 and effect["diverged"] == []
    assert effect["batch_widths"] == {"torch": 6}
    assert effect["guards"]["rescued"] == 2      # 1 per subject
    assert effect["guards"]["confirmations"] == 2


def test_batching_section_renders(batch_results):
    text = render(batch_results)
    assert "6.5b Batched probing: k=1 vs k=production" in text
    assert "Identical final configuration" in text
    assert "rescued by a re-probe" in text


# ------------------------------------------------- derived relaxed spec

def test_primary_requirements_are_derived_when_the_project_ships_none(tmp_path):
    """pip_upgrade needs an unpinned spec; almost no project publishes one."""
    from evaluation.runner import ensure_primary_requirements
    from evaluation.subjects import Subject

    work = tmp_path / "repo"
    work.mkdir()
    (work / "requirements.txt").write_text(
        "# a comment\n"
        "requests[socks]==2.28.0  # keep extras and comments\n"
        "numpy>=1.21,<1.24\n"
        'importlib-metadata==4.0 ; python_version < "3.10"\n')

    path, derived = ensure_primary_requirements(Subject(name="s"), work)
    assert derived is True
    text = path.read_text()
    assert "requests[socks]" in text and "==2.28.0" not in text
    assert "numpy" in text and ">=1.21" not in text
    assert 'python_version < "3.10"' in text      # markers survive


def test_a_project_that_ships_its_own_relaxed_spec_is_left_alone(tmp_path):
    """The project's own file is evidence; ours is a harness artefact."""
    from evaluation.runner import ensure_primary_requirements
    from evaluation.subjects import Subject

    work = tmp_path / "repo"
    work.mkdir()
    (work / "requirements.txt").write_text("numpy==1.21.0\n")
    (work / "primary_requirements.txt").write_text("numpy>=1.20\n")

    path, derived = ensure_primary_requirements(Subject(name="s"), work)
    assert derived is False
    assert path.read_text() == "numpy>=1.20\n"


# --------------------------------------------- per-dependency proposal arms

def write_proposal_run(root, subject, arm, *, outdated, proposals, updated):
    """A proposal arm's log: each bump judged alone, plus the union probe."""
    cfg = RunConfig(repo=subject, arm=arm)
    log = RunLog(root / subject / f"{arm}.jsonl", cfg)
    for package, to_version, accepted in proposals:
        log.emit("proposal", package=package, to_version=to_version,
                 from_version="1.0.0", source="dependabot",
                 accepted=accepted,
                 error_class="" if accepted else "resolution_conflict")
    log.emit("proposal", package="<union>", to_version="", accepted=False,
             error_class="resolution_conflict", source="dependabot")
    log.finish(Outcome.MODERNIZED if updated else Outcome.KEPT_BASELINE,
               updated=updated,
               version_age={"before": {"per_package": outdated},
                            "after": {"per_package": {}}})


@pytest.fixture
def proposal_results(tmp_path):
    root = tmp_path / "prop"
    outdated = {"torch": 4, "torchvision": 3, "click": 2, "requests": 0}
    # The proposer offers each package alone; the coupled pair fails alone.
    write_proposal_run(root, "s1", "dependabot", outdated=outdated,
                       proposals=[("torch", "2.0.0", False),
                                  ("torchvision", "0.15.1", False),
                                  ("click", "8.1.7", True)],
                       updated={"click": ["8.0.0", "8.1.7"]})
    # AURA moves the pair together.
    cfg = RunConfig(repo="s1", arm="aura_full_qwen14b")
    log = RunLog(root / "s1" / "aura_full_qwen14b.jsonl", cfg)
    log.finish(Outcome.MODERNIZED,
               updated={"torch": ["1.12.0", "2.0.0"],
                        "torchvision": ["0.13.0", "0.15.1"],
                        "click": ["8.0.0", "8.1.7"]},
               version_age={"before": {"per_package": outdated},
                            "after": {"per_package": {}}})
    return Analysis(load_runs(root))


def test_per_package_rate_uses_the_same_denominator_for_every_arm(proposal_results):
    rows = proposal_results.per_package_advance()
    # Three outdated pins; requests is already current and is not counted.
    assert rows["dependabot"]["outdated"] == 3
    assert rows["aura_full_qwen14b"]["outdated"] == 3
    assert rows["dependabot"]["advanced"] == 1
    assert rows["aura_full_qwen14b"]["advanced"] == 3
    assert rows["dependabot"]["rate"] == round(1 / 3, 4)
    assert rows["aura_full_qwen14b"]["rate"] == 1.0


def test_the_co_resolution_gap_is_measured_not_asserted(proposal_results):
    """Packages proposed alone, failed alone, advanced by AURA in a group."""
    gap = proposal_results.coupled_failures()
    assert gap["rejected_proposals"] == 2
    assert gap["recovered_by_aura"] == 2
    torch = next(e for e in gap["examples"] if e["package"] == "torch")
    assert torch["aura_reached"] == "2.0.0"
    assert "torchvision" in torch["aura_moved_with"]


def test_proposal_sections_render(proposal_results):
    text = render(proposal_results)
    assert "6b. Per-dependency advance rate" in text
    assert "6b.1 The co-resolution gap, measured" in text
    assert "never offered" in text


def test_a_one_shot_arm_reports_what_it_actually_advanced(tmp_path, monkeypatch):
    """pip_upgrade genuinely moves most pins; recording nothing would score it
    zero in the per-package comparison and flatter AURA by exactly that much."""
    from evaluation.arms import Arm
    from evaluation.runner import run_baseline_arm
    from evaluation.subjects import Subject
    import aura.environment as env_mod
    import evaluation.runner as runner_mod

    work = tmp_path / "ws"
    work.mkdir()
    (work / "requirements.txt").write_text("click==8.0.0\nrequests==2.28.0\n")
    (work / "primary_requirements.txt").write_text("click\nrequests\n")

    class Result:
        install_ok = accepted = True
        validation = None
        error_class = "none"
        stderr = ""
        duration_s = 1.0
        frozen = {"click": "8.1.7", "requests": "2.28.0"}   # only click moved

    monkeypatch.setattr(runner_mod, "probe", lambda *a, **k: Result())
    monkeypatch.setattr(runner_mod, "version_age_report",
                        lambda index, pins: {"per_package": {"click": 3}})
    monkeypatch.setattr(runner_mod, "_interpreter", lambda subject: "python")

    subject = Subject(name="s", python_version="3.11")
    result = run_baseline_arm(
        subject, Arm("pip_upgrade", "baseline", upgrade_source="primary"),
        work, tmp_path / "log.jsonl", index=None, timeout_s=60)

    assert result.updated == {"click": ["8.0.0", "8.1.7"]}
    assert "requests" not in result.updated       # installed, not advanced
    assert Analysis(load_runs(tmp_path)).per_package_advance(
        )["pip_upgrade"]["advanced"] == 1


def test_upgrade_arms_are_judged_by_the_same_rule_as_aura(tmp_path, monkeypatch):
    """A pre-existing test failure must not count against an upgrade arm.

    AURA is judged baseline-relative: only *new* failures reject a candidate.
    Judging pip_upgrade absolutely would penalise it for failures the subject
    already had and AURA is forgiven, biasing every comparison one way.
    """
    from evaluation.arms import Arm
    from evaluation.runner import run_baseline_arm
    from evaluation.subjects import Subject
    import evaluation.runner as runner_mod

    work = tmp_path / "ws"
    work.mkdir()
    (work / "requirements.txt").write_text("click==8.0.0\n")
    (work / "primary_requirements.txt").write_text("click\n")

    calls = []

    class Result:
        install_ok = accepted = True
        validation = None
        error_class = "none"
        stderr = ""
        duration_s = 1.0
        frozen = {"click": "8.1.7"}

    def fake_probe(reqfile, **kw):
        calls.append(kw.get("record_baseline"))
        return Result()

    monkeypatch.setattr(runner_mod, "probe", fake_probe)
    monkeypatch.setattr(runner_mod, "version_age_report",
                        lambda index, pins: {"per_package": {"click": 3}})
    monkeypatch.setattr(runner_mod, "_interpreter", lambda subject: "python")

    run_baseline_arm(Subject(name="s", python_version="3.11"),
                     Arm("pip_upgrade", "baseline", upgrade_source="primary"),
                     work, tmp_path / "log.jsonl", index=None, timeout_s=60)

    # First probe records the project's own reference; the second is judged
    # against it rather than against "everything must pass".
    assert calls == [True, False]


def test_the_baseline_arm_itself_is_still_judged_absolutely(tmp_path, monkeypatch):
    """It is the inclusion gate: a subject whose own tests fail is excluded."""
    from evaluation.arms import Arm
    from evaluation.runner import run_baseline_arm
    from evaluation.subjects import Subject
    import evaluation.runner as runner_mod

    work = tmp_path / "ws"
    work.mkdir()
    (work / "requirements.txt").write_text("click==8.0.0\n")

    calls = []

    class Result:
        install_ok = accepted = True
        validation = None
        error_class = "none"
        stderr = ""
        duration_s = 1.0
        frozen = {"click": "8.0.0"}

    monkeypatch.setattr(runner_mod, "probe",
                        lambda reqfile, **kw: (calls.append(kw.get("record_baseline"))
                                               or Result()))
    monkeypatch.setattr(runner_mod, "version_age_report",
                        lambda index, pins: {"per_package": {}})
    monkeypatch.setattr(runner_mod, "_interpreter", lambda subject: "python")

    run_baseline_arm(Subject(name="s", python_version="3.11"),
                     Arm("baseline", "baseline"), work, tmp_path / "log.jsonl",
                     index=None, timeout_s=60)
    assert calls == [True]


def test_log_filenames_name_their_subject(tmp_path):
    """Hundreds of these get opened in an editor; a tab reading
    "pip_upgrade.log" does not say which of fifty subjects it belongs to."""
    from evaluation.runner import log_path_for

    path = log_path_for(tmp_path, "rake-nltk", "pip_upgrade")
    assert path.name == "rake-nltk__pip_upgrade.jsonl"
    assert path.parent.name == "rake-nltk"
    # The console and summary files derive from it and stay unambiguous too.
    assert path.with_suffix(".log").name == "rake-nltk__pip_upgrade.log"
    assert path.with_suffix(".result.json").name == \
        "rake-nltk__pip_upgrade.result.json"


def test_the_subject_digest_is_derived_and_never_the_record(tmp_path):
    """One readable file per subject, assembled from the per-arm logs.

    It must be regenerable and must not feed back into the results the report
    reads -- otherwise re-running one arm would have to edit a shared file,
    which is the merge problem that made re-runs double their attempt counts.
    """
    import argparse
    from evaluation.cli import cmd_digest
    from evaluation.report import load_runs

    d = tmp_path / "s1"
    d.mkdir(parents=True)
    (d / "s1__aura_full_qwen14b.log").write_text("AURA RUN\n")
    (d / "s1__baseline.log").write_text("BASELINE RUN\n")

    cmd_digest(argparse.Namespace(results=tmp_path))
    text = (d / "s1__ALL.log").read_text()

    # Canonical arm order, not filesystem order: the baseline comes first.
    assert text.index("BASELINE RUN") < text.index("AURA RUN")
    assert "This file is generated" in text

    # Regenerating is idempotent and never consumes its own output.
    cmd_digest(argparse.Namespace(results=tmp_path))
    assert (d / "s1__ALL.log").read_text().count("BASELINE RUN") == 1
    assert load_runs(tmp_path) == []          # .log files are not results


def test_resume_skips_finished_runs_but_not_interrupted_ones(tmp_path):
    """A run killed partway leaves a log that exists but never ended.

    Treating that as done would leave a permanent hole in the campaign that
    reads as a completed failure, and re-running would never fill it.
    """
    from evaluation.cli import _run_is_complete
    from aura.telemetry import Outcome, RunConfig, RunLog

    finished = tmp_path / "done.jsonl"
    log = RunLog(finished, RunConfig(repo="s", arm="a"))
    log.finish(Outcome.MODERNIZED)
    assert _run_is_complete(finished) is True

    interrupted = tmp_path / "partial.jsonl"
    RunLog(interrupted, RunConfig(repo="s", arm="a"))     # run_start only
    assert _run_is_complete(interrupted) is False

    assert _run_is_complete(tmp_path / "absent.jsonl") is False


def test_max_passes_reaches_the_agent_from_the_cli(tmp_path, monkeypatch):
    """The ceiling has to be recorded in the log, not just honoured in memory:
    'hit the cap' and 'converged' are different findings and the config block is
    where a reader checks which one a run could have been."""
    from evaluation.arms import Arm
    from evaluation.runner import _run_config
    from evaluation.subjects import Subject

    rc = _run_config(Subject(name="s"), Arm("aura_full_qwen14b", "aura"),
                     timeout_s=60, pip_cache=None, max_passes=5)
    assert rc.max_passes == 5

    from aura.telemetry import RunConfig
    assert RunConfig().max_passes == 20         # the default matches the flag


def test_the_digest_summary_distinguishes_running_from_finished(tmp_path):
    """"Still running" and "finished with nothing to show" are different states.

    A campaign is read while parts of it are still executing, and a summary that
    blurs the two invites reading an unfinished arm as a negative result.
    """
    import argparse
    from evaluation.cli import cmd_digest
    from aura.telemetry import Outcome, RunConfig, RunLog

    d = tmp_path / "s1"
    d.mkdir(parents=True)
    done = RunLog(d / "s1__baseline.jsonl", RunConfig(repo="s1", arm="baseline"))
    done.finish(Outcome.MODERNIZED, updated={"click": ["8.0.0", "8.1.7"]})
    RunLog(d / "s1__aura_full_qwen14b.jsonl", RunConfig(repo="s1", arm="aura_full_qwen14b"))
    (d / "s1__baseline.log").write_text("BASELINE\n")
    (d / "s1__aura_full_qwen14b.log").write_text("AURA\n")

    cmd_digest(argparse.Namespace(results=tmp_path))
    text = (d / "s1__ALL.log").read_text()

    assert "SUMMARY" in text
    assert "modernized" in text
    assert "click 8.0.0->8.1.7" in text
    assert "in progress" in text          # the unfinished arm, not "failed"


# ------------------------------------------- arms of a subject run in parallel

def test_publishing_to_the_repo_cache_survives_a_race(tmp_path):
    """Arms run as separate jobs, so several reach the cache at once. The loser
    must carry on, not crash the arm (and not leave a half-copied checkout for
    a third arm to run against)."""
    from concurrent.futures import ThreadPoolExecutor

    from evaluation.runner import publish_to_cache

    cache = tmp_path / "cache"
    clones = []
    for i in range(4):                      # four arms, four clones of one subject
        clone = tmp_path / f"clone{i}"
        (clone / ".git").mkdir(parents=True)
        (clone / "requirements.txt").write_text("flask==1.1.1\n")
        clones.append(clone)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda c: publish_to_cache(c, cache, "subject"), clones))

    assert (cache / "subject" / "requirements.txt").read_text() == "flask==1.1.1\n"
    # No staging directories left behind, so the next run sees a clean cache.
    assert [p.name for p in cache.iterdir()] == ["subject"]


def test_a_cache_entry_is_never_visible_half_written(tmp_path, monkeypatch):
    """The dangerous half of the race: an arm copying *out* of the cache while
    another is still copying in would run against an incomplete checkout."""
    import shutil as shutil_mod

    from evaluation.runner import publish_to_cache

    cache = tmp_path / "cache"
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / "requirements.txt").write_text("flask==1.1.1\n")

    seen: list[list[str]] = []
    real_copytree = shutil_mod.copytree

    def watching_copytree(src, dst, **kw):
        result = real_copytree(src, dst, **kw)
        # Mid-publish: whatever is visible under the cache's own name must be
        # either absent or complete, never a partial copy.
        seen.append([p.name for p in cache.iterdir()] if cache.exists() else [])
        return result

    monkeypatch.setattr("evaluation.runner.shutil.copytree", watching_copytree)
    publish_to_cache(clone, cache, "subject")
    assert seen and "subject" not in seen[0]      # staged under a private name
    assert (cache / "subject" / "requirements.txt").exists()


def test_a_one_shot_arm_is_judged_against_the_repaired_baseline(tmp_path,
                                                                monkeypatch):
    """A project whose declared file no longer installs still has a yardstick.

    Without this the arm aborts on "baseline precondition failed" and is
    recorded as a failure although it never ran its upgrade -- which would read
    as a fact about the upgrade strategy rather than about the reference.
    What it installs is unchanged: stripping constraints resolves to the newest
    releases whatever the reference was.
    """
    from evaluation.arms import Arm
    from evaluation.runner import run_baseline_arm
    from evaluation.subjects import Subject
    import evaluation.runner as runner_mod

    work = tmp_path / "ws"
    work.mkdir()
    (work / "requirements.txt").write_text("flask==1.1.1\n")
    (work / "primary_requirements.txt").write_text("flask\n")

    installed = []

    class Result:
        def __init__(self, ok):
            self.install_ok = self.accepted = ok
            self.validation = None
            self.error_class = "none" if ok else "validation_failed"
            self.stderr = ""
            self.duration_s = 1.0
            self.frozen = {"flask": "3.1.3"}
            self.new_failures = []

    def fake_probe(reqfile, **kw):
        pins = reqfile.pins()
        installed.append(dict(pins))
        # The declared file is dead; the reconstruction (jinja2 pinned back)
        # works. The upgrade itself has no pins at all and passes.
        if kw.get("record_baseline") and pins.get("flask") == "1.1.1":
            return Result(pins.get("jinja2") == "2.11.2")
        return Result(True)

    class FakeEnv:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def resolve_closure(self, reqfile, workdir):
            return {"flask": "1.1.1", "jinja2": "3.1.6"}

    class DatedIndex:
        def newest_at(self, name, on_or_before, python_version=None, specifier=None, prefer_wheels=False):
            return {"jinja2": "2.11.2"}.get(name)

    monkeypatch.setattr(runner_mod, "probe", fake_probe)
    monkeypatch.setattr(runner_mod, "Environment", FakeEnv)
    monkeypatch.setattr(runner_mod, "version_age_report",
                        lambda index, pins: {"at_latest_pct": 0.0})
    monkeypatch.setattr(runner_mod, "_interpreter", lambda s: "python")

    subject = Subject(name="s", python_version="3.9", commit_date="2020-04-14",
                      validation={"type": "pytest", "pytest_target": "tests"})
    arm = Arm("pip_upgrade", "baseline", upgrade_source="primary")
    result = run_baseline_arm(subject, arm, work, tmp_path / "s__pip_upgrade.jsonl",
                              index=DatedIndex(), timeout_s=60)

    assert result.outcome != "failed", "the arm must get to run its upgrade"
    # The reference was the repaired environment, and the upgrade it installed
    # is still the unpinned specification.
    assert {"flask": "1.1.1", "jinja2": "2.11.2"} in installed
    assert installed[-1] == {}


def test_a_subject_is_not_skipped_because_its_declared_file_is_dead(tmp_path,
                                                                    monkeypatch):
    """The precondition is that a working environment exists to measure
    against, not that the project's own file still installs -- most published
    projects pin part of their dependencies and no longer install at all."""
    from evaluation.arms import Arm
    from evaluation.runner import ArmResult, run_subject
    import evaluation.runner as runner_mod
    from evaluation.subjects import Subject

    ran = []

    def fake_run_arm(subject, arm, **kw):
        ran.append(arm.name)
        if arm.name == "baseline":
            # The declared file fails, but the ladder reconstructed one.
            return ArmResult(subject.name, arm.name, "failed",
                             baseline_stage="reconstructed")
        return ArmResult(subject.name, arm.name, "modernized")

    monkeypatch.setattr(runner_mod, "run_arm", fake_run_arm)
    arms = [Arm("baseline", "baseline"), Arm("pip_upgrade", "baseline"),
            Arm("aura_full_qwen14b", "aura")]
    run_subject(Subject(name="s"), arms)
    assert ran == ["baseline", "pip_upgrade", "aura_full_qwen14b"]

    # Nothing could be established: that subject really has no baseline.
    ran.clear()

    def no_baseline(subject, arm, **kw):
        ran.append(arm.name)
        return ArmResult(subject.name, arm.name, "failed", baseline_stage="")

    monkeypatch.setattr(runner_mod, "run_arm", no_baseline)
    run_subject(Subject(name="s"), arms)
    assert ran == ["baseline"]


# ------------------------------------------------- the sample's shape by age

def test_a_subject_belongs_to_the_stratum_of_its_pinned_commit():
    """Dependency rot is a function of time: a result from one era is a result
    about that era, so which era a subject represents cannot be implicit."""
    from evaluation.subjects import Subject, era_of

    assert era_of("2025-06-01") == "current"
    assert era_of("2022-09-26") == "recent"
    assert era_of("2020-04-14") == "stale"
    assert era_of("2016-03-02") == "old"
    assert era_of("2013-11-30") == "ancient"
    assert era_of("") == "", "an unknown date is not quietly filed somewhere"
    assert Subject(name="s", commit_date="2020-01-28").era == "stale"


def test_the_shape_of_the_sample_is_reported_including_its_gaps():
    """A stratum with no subjects is a gap to fill, not a stratum to drop."""
    from evaluation.subjects import Registry, Subject

    registry = Registry([
        Subject(name="a", commit_date="2020-04-14"),
        Subject(name="b", commit_date="2020-08-16"),
        Subject(name="c", commit_date="2022-09-26"),
    ])
    counts = registry.era_counts()
    assert counts["stale"] == 2 and counts["recent"] == 1
    assert counts["ancient"] == 0 and counts["old"] == 0 and counts["current"] == 0


def test_pytests_summary_is_found_after_pips_chatter():
    """Screening merges install and test into one container, and stderr is
    appended after stdout: pip's warnings come last, so reading the final line
    parses every suite as zero tests passed and rejects the repository."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "screen_docker", Path(__file__).resolve().parent.parent
        / "scripts" / "screen_docker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = (
        "collected 152 items\n"
        "======================== 150 passed, 2 failed in 3.21s ================\n"
        "WARNING: Running pip as the 'root' user can result in broken permissions\n"
        "[notice] A new release of pip is available: 23.0.1 -> 26.0.1\n")
    assert module.summary_line(output).startswith("150 passed, 2 failed")
    assert module.summary_line("no tests ran in 0.01s\nWARNING: pip\n") == \
        "no tests ran in 0.01s"
    assert module.summary_line("nothing useful here") == ""


def test_a_rejected_environment_is_not_reported_as_what_the_arm_reached(
        tmp_path, monkeypatch):
    """An arm whose environment failed validation ends at its baseline.

    test-flask's baseline arm failed validation, logged "no package advanced;
    the original pins are kept", and then reported "0.0% -> 57.14%" -- the
    version age of the environment it had just thrown away. The baseline row is
    what every other arm is compared against, so it read as though doing
    nothing modernised more than half the project.
    """
    from evaluation.arms import Arm
    from evaluation.runner import run_baseline_arm
    from evaluation.subjects import Subject
    import evaluation.runner as runner_mod

    work = tmp_path / "ws"
    work.mkdir()
    (work / "requirements.txt").write_text("click==7.0\n")
    (work / "primary_requirements.txt").write_text("click\n")

    class Rejected:
        install_ok = True
        accepted = False                    # the tests failed on it
        validation = None
        error_class = "validation_failed"
        stderr = ""
        duration_s = 1.0
        frozen = {"click": "8.1.8"}         # resolved, installed, thrown away

    ages = {"click==7.0": {"at_latest_pct": 0.0, "per_package": {"click": 11}},
            "click==8.1.8": {"at_latest_pct": 100.0, "per_package": {"click": 0}}}
    monkeypatch.setattr(runner_mod, "probe", lambda *a, **k: Rejected())
    monkeypatch.setattr(runner_mod, "version_age_report",
                        lambda index, pins: ages[
                            "".join(f"{n}=={v}" for n, v in pins.items())])
    monkeypatch.setattr(runner_mod, "_interpreter", lambda subject: "python")

    result = run_baseline_arm(
        Subject(name="s", python_version="3.11"),
        Arm("baseline", "baseline", upgrade_source="primary"),
        work, tmp_path / "log.jsonl", index=None, timeout_s=60)

    assert result.version_age_after == result.version_age_before, (
        "the arm kept its baseline, so that is the environment it ended with")
    assert result.version_age_after["at_latest_pct"] == 0.0
