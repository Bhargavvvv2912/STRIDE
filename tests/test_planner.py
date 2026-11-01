"""Tests for conflict extraction, plan validation, and the planner loop."""

import json
import pytest

from aura.index import ReleaseHistory
from aura.llm import NullLLM, LLMUnavailable, ScriptedLLM
from aura.planner import (
    Planner, RejectionReason, build_prompt, extract_conflicting_packages, validate_plan,
)
from tests.test_index import FakeIndex, hist

RESOLVER_LOG = """\
ERROR: Cannot install torch==2.1.0 and torchvision==0.13.0 because these
package versions have conflicting dependencies.

The conflict is caused by:
    torchvision 0.13.0 depends on torch==1.12.0
    The user requested torch==2.1.0
"""

CURRENT = {"torch": "1.12.0", "torchvision": "0.13.0", "numpy": "1.22.0"}
AVAILABLE = {"torch": "2.1.0", "torchvision": "0.16.0", "numpy": "1.26.4"}
INDEX = FakeIndex({
    "torch": hist("torch", ["1.12.0", "1.12.1", "1.13.0", "2.0.0", "2.1.0"]),
    "torchvision": hist("torchvision", ["0.13.0", "0.13.1", "0.14.0", "0.16.0"]),
    "numpy": hist("numpy", ["1.22.0", "1.24.0", "1.26.4"]),
})


# --------------------------------------------------------- conflict extraction

def test_extraction_finds_the_real_blockers():
    found = extract_conflicting_packages(RESOLVER_LOG)
    assert "torch" in found and "torchvision" in found


def test_restricting_to_known_packages_removes_prose_noise():
    unrestricted = extract_conflicting_packages(RESOLVER_LOG)
    restricted = extract_conflicting_packages(RESOLVER_LOG, known=set(CURRENT))
    assert set(restricted) <= set(unrestricted)
    assert set(restricted) == {"torch", "torchvision"}
    assert len(restricted) <= len(unrestricted)


def test_extraction_drops_english_noise_words():
    found = extract_conflicting_packages("conflict between the version and package for 1.0")
    assert not ({"the", "and", "version", "package", "for", "conflict"} & set(found))


def test_extraction_canonicalizes_names():
    assert "ruamel-yaml" in extract_conflicting_packages("ruamel.yaml==0.17.0 depends on x")


def test_extraction_of_empty_log_is_empty():
    assert extract_conflicting_packages("") == []


# ------------------------------------------------------------ plan validation

def test_hallucinated_version_is_rejected():
    """v1 never checked the index, so invented versions cost a full install cycle."""
    ok, bad = validate_plan(["torch==1.12.5"], current=CURRENT,
                            ceilings=AVAILABLE, index=INDEX)
    assert ok == []
    assert bad[0]["reason"] == RejectionReason.NOT_ON_INDEX


def test_downgrade_is_rejected():
    ok, bad = validate_plan(["torch==1.11.0"], current=CURRENT,
                            ceilings=AVAILABLE, index=INDEX)
    assert ok == [] and bad[0]["reason"] == RejectionReason.NOT_AN_UPGRADE


def test_version_above_the_ceiling_is_rejected():
    """v1 computed this ceiling and then never compared against it."""
    ok, bad = validate_plan(["torch==3.0.0"], current=CURRENT,
                            ceilings=AVAILABLE, index=INDEX)
    assert ok == [] and bad[0]["reason"] == RejectionReason.EXCEEDS_CEILING


def test_package_outside_the_project_is_rejected():
    ok, bad = validate_plan(["tensorflow==2.15.0"], current=CURRENT,
                            ceilings=AVAILABLE, index=INDEX, allowed=set(CURRENT))
    assert ok == [] and bad[0]["reason"] == RejectionReason.NOT_IN_ENV


@pytest.mark.parametrize("junk", ["torch", "torch>=2.0", "", "==1.0", "torch==banana"])
def test_unparseable_entries_are_rejected(junk):
    ok, bad = validate_plan([junk], current=CURRENT, ceilings=AVAILABLE, index=INDEX)
    assert ok == [] and bad[0]["reason"] in {
        RejectionReason.UNPARSEABLE, RejectionReason.NOT_IN_ENV}


def test_valid_intermediate_versions_are_accepted():
    ok, bad = validate_plan(["torch==1.12.1", "torchvision==0.13.1"],
                            current=CURRENT, ceilings=AVAILABLE, index=INDEX)
    assert ok == ["torch==1.12.1", "torchvision==0.13.1"] and bad == []


def test_validation_is_resilient_when_the_index_is_unreachable():
    idx = FakeIndex({}, fail={"torch"})
    ok, bad = validate_plan(["torch==1.12.1"], current=CURRENT,
                            ceilings=AVAILABLE, index=idx)
    assert ok == ["torch==1.12.1"]     # fall through rather than block the run


def test_a_mixed_plan_keeps_the_good_half():
    ok, bad = validate_plan(
        ["torch==1.12.1", "torch==99.0.0", "numpy==1.20.0"],
        current=CURRENT, ceilings=AVAILABLE, index=INDEX)
    assert ok == ["torch==1.12.1"]
    assert {r["reason"] for r in bad} == {
        RejectionReason.EXCEEDS_CEILING, RejectionReason.NOT_AN_UPGRADE}


# ------------------------------------------------------------------- prompting

def test_prompt_contains_the_floor_the_ceiling_and_the_log():
    p = build_prompt(target="torch", error_log=RESOLVER_LOG,
                     current=CURRENT, available=AVAILABLE)
    assert '"torch": "1.12.0"' in p and '"torch": "2.1.0"' in p
    assert "conflicting dependencies" in p and "FORWARD ONLY" in p


def test_explanation_first_toggles_the_schema():
    kw = dict(target="torch", error_log="", current=CURRENT, available=AVAILABLE)
    assert "explanation" in build_prompt(**kw, explanation_first=True)
    assert '"explanation"' not in build_prompt(**kw, explanation_first=False)


def test_history_is_included_so_the_model_does_not_repeat_itself():
    p = build_prompt(target="torch", error_log="", current=CURRENT, available=AVAILABLE,
                     history=[("torch==2.1.0", "ResolutionImpossible")])
    assert "PREVIOUS FAILED ATTEMPTS" in p and "ResolutionImpossible" in p


# --------------------------------------------------------------- planner loop

def _planner(responses, **kw):
    return Planner(ScriptedLLM(responses), index=INDEX, **kw)


def test_planner_returns_a_validated_plan():
    llm_json = json.dumps({"explanation": "torchvision 0.13 pins torch 1.12",
                           "resolvable": True, "plausible": True,
                           "proposed_plan": ["torch==1.12.1", "torchvision==0.13.1"]})
    plan = _planner([llm_json]).propose(
        target="torch", error_log=RESOLVER_LOG, current=CURRENT, available=AVAILABLE)
    assert plan.requirements == ["torch==1.12.1", "torchvision==0.13.1"]
    assert plan.as_pins() == {"torch": "1.12.1", "torchvision": "0.13.1"}
    assert "torchvision" in plan.explanation and plan.rejected == []


def test_planner_records_hallucinations_rather_than_executing_them():
    """A hallucinated version for the *target* voids the whole plan.

    torchvision==0.13.1 is fine on its own, but with torch's version rejected the
    remainder does not touch the package that caused the escalation, so it cannot
    resolve anything. It is rejected rather than executed.
    """
    llm_json = json.dumps({"resolvable": True, "plausible": True,
                           "proposed_plan": ["torch==1.12.9", "torchvision==0.13.1"]})
    plan = _planner([llm_json]).propose(
        target="torch", error_log="", current=CURRENT, available=AVAILABLE)
    reasons = {r["reason"] for r in plan.rejected}
    assert reasons == {RejectionReason.NOT_ON_INDEX, RejectionReason.MISSING_TARGET}
    assert plan.requirements == []


def test_a_hallucinated_non_target_version_is_dropped_but_the_plan_survives():
    llm_json = json.dumps({"resolvable": True, "plausible": True,
                           # under the 0.16.0 ceiling, but no such release exists
                           "proposed_plan": ["torch==1.12.1", "torchvision==0.13.5"]})
    plan = _planner([llm_json]).propose(
        target="torch", error_log="", current=CURRENT, available=AVAILABLE)
    assert plan.requirements == ["torch==1.12.1"]
    assert plan.rejected[0]["reason"] == RejectionReason.NOT_ON_INDEX


def test_a_plan_that_omits_the_target_package_is_rejected():
    """Observed with llama3.1:8b on a real conflict: it upgraded the *other*
    package and dropped the target entirely.

    The prompt's SCOPE rule requires the target. Unenforced, such a plan gets
    executed, leaves the escalating package untouched, and burns a full
    install-and-validate cycle for nothing.
    """
    llm_json = json.dumps({"resolvable": True, "plausible": True,
                           "proposed_plan": ["torchvision==0.14.0"]})
    plan = _planner([llm_json]).propose(
        target="torch", error_log="", current=CURRENT, available=AVAILABLE)
    assert plan.requirements == []
    assert plan.rejected[0]["reason"] == RejectionReason.MISSING_TARGET
    assert "torch" in plan.reason


def test_planner_honours_the_early_exit_signal():
    llm_json = json.dumps({"resolvable": False, "reason": "pandas 3 needs Python >= 3.11"})
    plan = _planner([llm_json]).propose(
        target="pandas", error_log="", current=CURRENT, available=AVAILABLE)
    assert plan.resolvable is False and "3.11" in plan.reason and plan.is_empty


def test_planner_survives_a_non_json_response():
    plan = _planner(["I'm afraid I can't do that."]).propose(
        target="torch", error_log="", current=CURRENT, available=AVAILABLE)
    assert plan.is_empty and plan.reason == "unparseable_response"


def test_planner_records_token_usage():
    llm_json = json.dumps({"resolvable": True, "plausible": True,
                           "proposed_plan": ["torch==1.12.1"]})
    plan = _planner([llm_json]).propose(
        target="torch", error_log=RESOLVER_LOG, current=CURRENT, available=AVAILABLE)
    assert plan.usage.prompt_tokens > 0 and plan.usage.completion_tokens > 0


def test_planner_with_the_null_client_raises_rather_than_calling_out():
    """A deterministic-only arm must not be able to reach a model by accident."""
    with pytest.raises(LLMUnavailable):
        Planner(NullLLM(), index=INDEX).propose(
            target="torch", error_log="", current=CURRENT, available=AVAILABLE)


# ----------------------------------------------- compact prompt (small models)

BIG_CURRENT = {**CURRENT, **{f"pkg{i}": "1.0.0" for i in range(30)}}
BIG_AVAILABLE = {**AVAILABLE, **{f"pkg{i}": "2.0.0" for i in range(30)}}


def _compact(**kw):
    from aura.planner import build_compact_prompt
    base = dict(target="torch", error_log=RESOLVER_LOG,
                current=CURRENT, available=AVAILABLE)
    return build_compact_prompt(**{**base, **kw})


def test_compact_prompt_is_selected_by_style():
    p = build_prompt(target="torch", error_log=RESOLVER_LOG, current=CURRENT,
                     available=AVAILABLE, style="compact")
    assert "MUST APPEAR IN YOUR ANSWER" in p
    assert "CONSTRAINT OPTIMIZATION" not in p


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError, match="unknown prompt style"):
        build_prompt(target="torch", error_log="", current=CURRENT,
                     available=AVAILABLE, style="terse")


def test_compact_prompt_restates_the_target_at_the_end():
    """llama3.1:8b dropped the target under the full prompt; this is the fix."""
    p = _compact()
    assert p.rstrip().endswith("Reply with one JSON object and nothing else.")
    tail = p[-300:]
    assert "MUST contain torch" in tail


def test_compact_prompt_narrows_early_exit():
    """qwen2.5-coder:14b took the early exit on a solvable coupling."""
    p = _compact()
    assert "is NOT such a case" in p
    assert "newer Python" in p


def test_compact_prompt_marks_the_target_row():
    p = _compact()
    target_line = [l for l in p.splitlines() if l.strip().startswith("torch ")][0]
    assert "MUST APPEAR IN YOUR ANSWER" in target_line


def test_compact_prompt_caps_the_package_table():
    """A fifty-package dump buries the two lines that matter."""
    from aura.planner import COMPACT_MAX_PACKAGES
    p = _compact(current=BIG_CURRENT, available=BIG_AVAILABLE)
    table = p.split("PACKAGES YOU MAY CHANGE")[1].split("RULES")[0]
    rows = [l for l in table.splitlines() if l.strip()]
    assert len(rows) <= COMPACT_MAX_PACKAGES


def test_compact_prompt_keeps_the_conflicting_packages_when_truncating():
    p = _compact(current=BIG_CURRENT, available=BIG_AVAILABLE)
    table = p.split("PACKAGES YOU MAY CHANGE")[1].split("RULES")[0]
    assert "torch " in table and "torchvision" in table


def test_compact_prompt_includes_a_worked_example():
    assert '"proposed_plan": ["examplea==1.4.2"' in _compact()


def test_compact_prompt_includes_failed_history():
    p = _compact(history=[("torch==2.0.0", "ResolutionImpossible")])
    assert "ALREADY TRIED" in p and "ResolutionImpossible" in p


def test_compact_prompt_is_substantially_shorter():
    full = build_prompt(target="torch", error_log=RESOLVER_LOG, current=BIG_CURRENT,
                        available=BIG_AVAILABLE, style="full")
    assert len(_compact(current=BIG_CURRENT, available=BIG_AVAILABLE)) < len(full) * 0.7


def test_planner_honours_the_prompt_style():
    llm = ScriptedLLM([json.dumps({"resolvable": True, "plausible": True,
                                   "proposed_plan": ["torch==1.12.1"]})])
    Planner(llm, index=INDEX, prompt_style="compact").propose(
        target="torch", error_log=RESOLVER_LOG, current=CURRENT, available=AVAILABLE)
    assert "MUST APPEAR IN YOUR ANSWER" in llm.prompts[0]
