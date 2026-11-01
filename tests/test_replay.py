"""Tests for the backend replay harness."""

import json
import pytest

from aura.llm import LLMUsage, NullLLM, ScriptedLLM
from aura.planner import DecisionPoint, Planner
from aura.telemetry import Outcome, RunConfig, RunLog, Tier
from evaluation.replay import (
    extract_decisions, render_markdown, replay_backend, summarize,
)
from tests.conftest import FakeIndex, history, llm_reply

INDEX = FakeIndex({
    "torch": history("torch", ["1.12.0", "1.12.1", "1.13.0", "2.0.0"]),
    "torchtext": history("torchtext", ["0.13.0", "0.13.1", "0.14.0"]),
})
CURRENT = {"torch": "1.12.0", "torchtext": "0.13.0"}
AVAILABLE = {"torch": "2.0.0", "torchtext": "0.14.0"}


@pytest.fixture
def results(tmp_path):
    """A reference run that escalated to the LLM and eventually succeeded."""
    root = tmp_path / "results"
    log = RunLog(root / "r1" / "aura_full.jsonl",
                 RunConfig(repo="r1", arm="aura_full", llm_model="gemini-2.5-flash"))
    for iteration, proposed in ((1, ["torch==2.0.0"]), (2, ["torch==1.12.1",
                                                            "torchtext==0.13.1"])):
        log.llm_call(
            LLMUsage(model="gemini-2.5-flash", prompt_tokens=5000,
                     completion_tokens=200).finalize(),
            purpose="co_resolution", package="torch", iteration=iteration,
            proposed=proposed, rejected=[],
            inputs=DecisionPoint(target="torch", error_log="ResolutionImpossible",
                                 current=CURRENT, available=AVAILABLE,
                                 iteration=iteration).as_dict())
    log.attempt(package="torch", tier=Tier.LLM,
                plan={"torch": "1.12.1", "torchtext": "0.13.1"},
                install_ok=True, validate_ok=True, duration_s=30.0, accepted=True)
    log.finish(Outcome.MODERNIZED, updated={"torch": ("1.12.0", "1.12.1")})
    return root


def test_decision_points_are_extracted_with_their_inputs(results):
    decisions = extract_decisions(results)
    assert len(decisions) == 2
    first = decisions[0]
    assert first.point.target == "torch"
    assert first.point.current == CURRENT and first.point.available == AVAILABLE
    assert first.point.error_log == "ResolutionImpossible"
    assert first.key == "r1/torch#1"


def test_the_plan_that_actually_worked_is_attached(results):
    """Scoring needs ground truth: what the reference run proved installable."""
    assert extract_decisions(results)[0].accepted_plan == {
        "torch": "1.12.1", "torchtext": "0.13.1"}


def test_logs_without_input_capture_are_skipped(tmp_path):
    """Older runs are not replayable and must not be silently half-used."""
    log = RunLog(tmp_path / "r0" / "old.jsonl", RunConfig(repo="r0", arm="aura_full"))
    log.llm_call(LLMUsage(model="m").finalize(), purpose="co_resolution",
                 package="torch", iteration=1, proposed=["torch==2.0.0"])
    log.finish(Outcome.MODERNIZED)
    assert extract_decisions(tmp_path) == []


def test_arm_filter_is_honoured(results):
    assert extract_decisions(results, arms={"nope"}) == []
    assert len(extract_decisions(results, arms={"aura_full"})) == 2


def test_replay_scores_a_backend_that_reproduces_the_working_plan(results, monkeypatch):
    decisions = extract_decisions(results)[:1]
    good = llm_reply(["torch==1.12.1", "torchtext==0.13.1"])
    monkeypatch.setattr("evaluation.replay.build_client",
                        lambda spec, temperature=0.0: ScriptedLLM([good]))
    rows = replay_backend(decisions, "fake:good", index=INDEX, progress=False)
    assert rows[0]["parsed"] and rows[0]["n_accepted"] == 2
    assert rows[0]["hallucinated"] == 0
    # Agreement is descriptive, not a grade: this backend happened to land on
    # the same plan the reference run accepted.
    assert rows[0]["agrees_with_reference"] is True


def test_replay_scores_a_hallucinating_backend(results, monkeypatch):
    decisions = extract_decisions(results)[:1]
    bad = llm_reply(["torch==1.12.9", "torchtext==0.13.1"])   # 1.12.9 is invented
    monkeypatch.setattr("evaluation.replay.build_client",
                        lambda spec, temperature=0.0: ScriptedLLM([bad]))
    rows = replay_backend(decisions, "fake:bad", index=INDEX, progress=False)
    assert rows[0]["hallucinated"] == 1
    assert rows[0]["rejection_reasons"]["not_on_index"] == 1
    assert rows[0]["agrees_with_reference"] is False


def test_replay_scores_an_unparseable_backend(results, monkeypatch):
    decisions = extract_decisions(results)[:1]
    monkeypatch.setattr("evaluation.replay.build_client",
                        lambda spec, temperature=0.0: ScriptedLLM(["sorry, no."]))
    rows = replay_backend(decisions, "fake:chatty", index=INDEX, progress=False)
    assert rows[0]["parsed"] is False and rows[0]["n_accepted"] == 0


def test_a_different_but_valid_plan_is_not_marked_wrong(results, monkeypatch):
    """The metric fix: AURA has no target version, so disagreeing != being wrong.

    torch==1.13.0 is a real release above the current pin and under the ceiling.
    It differs from what the reference run accepted, and nothing about it may be
    reported as an error -- only executing it can settle whether it is good.
    """
    decisions = extract_decisions(results)[:1]
    other = llm_reply(["torch==1.13.0", "torchtext==0.14.0"])
    monkeypatch.setattr("evaluation.replay.build_client",
                        lambda spec, temperature=0.0: ScriptedLLM([other]))
    row = replay_backend(decisions, "fake:other", index=INDEX, progress=False)[0]

    assert row["n_accepted"] == 2 and row["n_rejected"] == 0
    assert row["hallucinated"] == 0
    assert row["ceiling_violations"] == 0 and row["downgrades"] == 0
    assert row["agrees_with_reference"] is False      # differs...
    assert row.get("plan_works") is None              # ...and is not yet judged


def test_summary_computes_per_backend_rates(results, monkeypatch):
    decisions = extract_decisions(results)
    good = llm_reply(["torch==1.12.1", "torchtext==0.13.1"])
    bad = llm_reply(["torch==9.9.9"])
    monkeypatch.setattr("evaluation.replay.build_client",
                        lambda spec, temperature=0.0: ScriptedLLM(
                            [good, good] if spec == "a" else [bad, bad]))
    rows = replay_backend(decisions, "a", index=INDEX, progress=False)
    rows += replay_backend(decisions, "b", index=INDEX, progress=False)

    summary = {s["backend"]: s for s in summarize(rows)}
    # Without execution, utility is explicitly "not measured" rather than 0
    assert summary["a"]["plan_works_rate"] == "not executed"
    assert summary["a"]["hallucination_rate"] == 0.0
    assert summary["a"]["decisions"] == 2
    assert summary["b"]["wrong_version_rate"] == 1.0
    # 9.9.9 exceeds the ceiling, so it is caught before the index is consulted
    assert summary["b"]["ceiling_violation_rate"] == 1.0
    assert "100.0%" in summary["a"]["schema_valid"]


def test_markdown_table_renders(results, monkeypatch):
    monkeypatch.setattr("evaluation.replay.build_client",
                        lambda spec, temperature=0.0: ScriptedLLM(
                            [llm_reply(["torch==1.12.1"])] * 2))
    rows = replay_backend(extract_decisions(results), "x", index=INDEX, progress=False)
    table = render_markdown(summarize(rows))
    assert "| Backend |" in table and "`x`" in table


def test_planner_replay_reconstructs_the_same_prompt(results):
    """Replay must ask the identical question, not a paraphrase of it."""
    decision = extract_decisions(results)[0]
    llm = ScriptedLLM([llm_reply(["torch==1.12.1"])])
    Planner(llm, index=INDEX).replay(decision.point)
    prompt = llm.prompts[0]
    assert "torch" in prompt
    assert '"torch": "1.12.0"' in prompt        # the floor
    assert '"torch": "2.0.0"' in prompt         # the ceiling
    assert "ResolutionImpossible" in prompt     # the log


def test_decision_point_round_trips_through_json():
    point = DecisionPoint(target="torch", error_log="boom", current=CURRENT,
                          available=AVAILABLE, history=[("a", "b")], iteration=2)
    restored = DecisionPoint.from_dict(json.loads(json.dumps(point.as_dict())))
    assert restored.target == "torch" and restored.current == CURRENT
    assert restored.history == [("a", "b")]


# ------------------------------------------------ execution settles utility

def _subject(tmp_path):
    from evaluation.subjects import Subject
    return Subject(name="r1", url=str(tmp_path), commit="", screened_date="2026-09-07",
                   validation={"type": "script", "smoke_test_script": "validate.py"},
                   validation_provenance="repo_tests")


def test_execution_verdicts_are_deduplicated_across_backends(results, tmp_path, monkeypatch):
    """Two backends proposing the same plan must cost one install, not two."""
    from evaluation import replay as R

    installs = []

    def fake_probe(reqfile, *, config, validator, spec=None, env_path=None,
                   record_baseline=False):
        from aura.environment import ProbeResult
        from aura.validation import ValidationResult
        pins = reqfile.pins()
        installs.append(("base" if record_baseline else "cand", tuple(sorted(pins.items()))))
        if record_baseline:
            validator.baseline = ValidationResult(passed=True, collected=3)
            return ProbeResult(install_ok=True, accepted=True,
                               validation=validator.baseline, frozen=pins)
        works = pins.get("torch") == "1.12.1"
        return ProbeResult(install_ok=True, accepted=works,
                           validation=ValidationResult(passed=works, collected=3),
                           frozen=pins)

    monkeypatch.setattr("aura.environment.probe", fake_probe)
    monkeypatch.setattr("evaluation.runner.prepare_workspace",
                        lambda subject, dest, cache=None: tmp_path)

    decisions = R.extract_decisions(results)[:1]
    same = llm_reply(["torch==1.12.1", "torchtext==0.13.1"])
    monkeypatch.setattr("evaluation.replay.build_client",
                        lambda spec, temperature=0.0: ScriptedLLM([same]))
    rows = (R.replay_backend(decisions, "a", index=INDEX, progress=False)
            + R.replay_backend(decisions, "b", index=INDEX, progress=False))

    rows = R.execute_proposals(rows, decisions, subjects={"r1": _subject(tmp_path)},
                               workspaces_dir=tmp_path, index=INDEX, progress=False)

    assert all(r["plan_works"] is True for r in rows)
    candidate_installs = [i for i in installs if i[0] == "cand"]
    assert len(candidate_installs) == 1, "identical plans must be executed once"


def test_a_different_working_plan_scores_as_working(results, tmp_path, monkeypatch):
    """The whole point of the fix: disagreeing with the reference is not failing."""
    from evaluation import replay as R

    def fake_probe(reqfile, *, config, validator, spec=None, env_path=None,
                   record_baseline=False):
        from aura.environment import ProbeResult
        from aura.validation import ValidationResult
        pins = reqfile.pins()
        if record_baseline:
            validator.baseline = ValidationResult(passed=True, collected=3)
            return ProbeResult(install_ok=True, accepted=True,
                               validation=validator.baseline, frozen=pins)
        # Both 1.12.1 and the newer 1.13.0 are genuinely fine here.
        works = pins.get("torch") in {"1.12.1", "1.13.0"}
        return ProbeResult(install_ok=True, accepted=works,
                           validation=ValidationResult(passed=works, collected=3),
                           frozen=pins)

    monkeypatch.setattr("aura.environment.probe", fake_probe)
    monkeypatch.setattr("evaluation.runner.prepare_workspace",
                        lambda subject, dest, cache=None: tmp_path)

    decisions = R.extract_decisions(results)[:1]
    monkeypatch.setattr("evaluation.replay.build_client",
                        lambda spec, temperature=0.0: ScriptedLLM(
                            [llm_reply(["torch==1.13.0", "torchtext==0.14.0"])]))
    rows = R.replay_backend(decisions, "other", index=INDEX, progress=False)
    rows = R.execute_proposals(rows, decisions, subjects={"r1": _subject(tmp_path)},
                               workspaces_dir=tmp_path, index=INDEX, progress=False)

    row = rows[0]
    assert row["agrees_with_reference"] is False   # a different answer...
    assert row["plan_works"] is True               # ...that works
    # and it is more modern than the reference plan (1.13.0 > 1.12.1)
    assert row["releases_recovered"] > 0

    summary = R.summarize(rows)[0]
    assert "100.0%" in summary["plan_works_rate"]
    assert "0.0%" in summary["agreement_with_reference"]
