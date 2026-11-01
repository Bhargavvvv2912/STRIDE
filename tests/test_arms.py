"""Guards on the experimental design itself.

These have caught real mistakes. The arm once named "full AURA" was the top of
the ablation lattice, which ran with routing disabled -- so the headline system
was a configuration nobody would ship. Later, with routing enabled everywhere, a
three-tier ablation arm became a byte-identical duplicate of the full arm. And a
default LLM that needed an SDK nobody had installed crashed the LLM arms outright.
"""

import pytest

from evaluation.arms import (
    CORE_ARMS, ESSENTIAL_ARMS, FULL_ARM, MODELS, PRIMARY_MODEL, full_suite,
)


@pytest.fixture
def arms():
    return {a.name: a for a in full_suite()}


def test_every_llm_arm_uses_a_local_open_model(arms):
    """No arm depends on a hosted API or its SDK unless one is added explicitly."""
    specs = {a.llm_spec for a in arms.values() if a.uses_llm}
    assert specs <= set(MODELS.values())
    assert all(spec.startswith("ollama:") for spec in specs)


def test_the_shipped_system_exists_once_per_model(arms):
    for label in MODELS:
        full = arms[f"aura_full_{label}"]
        assert full.tiers == ("linear", "greedy", "llm")
        assert full.constraint_graph and full.routing
        assert full.llm_spec == MODELS[label]
    assert FULL_ARM == f"aura_full_{PRIMARY_MODEL}"


def test_routing_is_on_in_every_arm_that_measures_the_system(arms):
    off = [a.name for a in arms.values() if a.kind == "aura" and not a.routing]
    assert set(off) == {"aura_no_routing", "aura_no_graph"}


def test_the_lattice_does_not_duplicate_the_shipped_system(arms):
    lattice = [a for a in arms.values() if a.group == "rq_ablation"]
    assert all(len(a.tiers) < 3 for a in lattice)
    # 3 model-free subsets + 3 LLM subsets per model
    assert len(lattice) == 3 + 3 * len(MODELS)
    fulls = {(arms[f"aura_full_{m}"].tiers, arms[f"aura_full_{m}"].llm_spec)
             for m in MODELS}
    assert not any((a.tiers, a.llm_spec) in fulls for a in lattice)


def test_model_free_arms_are_not_duplicated_per_model(arms):
    """Linear and greedy never call a model, so one arm each is enough."""
    assert arms["aura_linear_greedy"].llm_spec == "none"
    assert not any(n.startswith("aura_linear_greedy_") for n in arms)


def test_each_cost_comparison_isolates_exactly_one_thing(arms):
    full = arms[FULL_ARM]
    no_routing, no_graph, serial = (arms["aura_no_routing"], arms["aura_no_graph"],
                                    arms["aura_serial"])
    for control in (no_routing, no_graph, serial):
        assert control.llm_spec == full.llm_spec       # same model as its reference
        assert control.tiers == full.tiers
    assert full.routing != no_routing.routing
    assert full.constraint_graph == no_routing.constraint_graph
    assert full.constraint_graph != no_graph.constraint_graph
    assert serial.batch_k == 1 and full.batch_k == 0
    assert (serial.routing, serial.constraint_graph) == \
        (full.routing, full.constraint_graph)


def test_the_essential_set(arms):
    assert set(ESSENTIAL_ARMS) <= set(arms)
    assert list(ESSENTIAL_ARMS[:3]) == [
        "baseline", "pip_upgrade", "aura_linear_greedy"]
    assert "dependabot" not in ESSENTIAL_ARMS      # runs on GitHub, per subject
    # Every model runs end to end: the cost and the quality of a whole run are
    # what the comparison is about, and replay cannot produce either.
    for label in MODELS:
        assert arms[f"aura_llm_{label}"].tiers == ("llm",)
        assert f"aura_llm_{label}" in ESSENTIAL_ARMS
        assert f"aura_full_{label}" in ESSENTIAL_ARMS
    assert len(ESSENTIAL_ARMS) == 3 + 2 * len(MODELS)


def test_only_the_coverage_arm_pays_for_instrumentation(arms):
    assert [a.name for a in arms.values() if a.measure_coverage] == ["aura_coverage"]


def test_the_core_set_contains_the_shipped_system(arms):
    assert FULL_ARM in CORE_ARMS
    assert all(name in arms for name in CORE_ARMS)


def test_an_extra_model_adds_its_own_arms():
    extra = {**MODELS, "gemini": "gemini:gemini-2.5-flash"}
    names = {a.name for a in full_suite(extra)}
    assert {"aura_full_gemini", "aura_llm_gemini"} <= names


def test_arm_names_are_unique():
    names = [a.name for a in full_suite()]
    assert len(names) == len(set(names))


def test_every_arm_compared_across_models_uses_the_same_prompt(arms):
    """The LLM-alone and full-system arms must share a prompt, or a difference
    between them could be the prompt rather than the search."""
    compared = [a for a in arms.values()
                if a.name.startswith(("aura_llm_", "aura_full_")) and a.uses_llm]
    assert {a.prompt_style for a in compared} == {"compact"}
    assert arms["aura_long_prompt"].prompt_style == "full"


def test_the_model_set_is_one_family_at_three_sizes_plus_one_other():
    from evaluation.arms import ALL_MODELS
    assert list(ALL_MODELS) == ["qwen3b", "llama8b", "qwen14b", "qwen32b"]
    qwen = [s for s in ALL_MODELS.values() if "qwen2.5-coder" in s]
    assert [s.rsplit(":", 1)[1] for s in qwen] == ["3b", "14b", "32b"]


def test_a_machine_can_run_a_subset_of_the_models(monkeypatch):
    """A 24 GB laptop cannot hold the 32b model; the cluster runs all four."""
    from evaluation.arms import active_models

    monkeypatch.setenv("AURA_MODELS", "qwen14b, llama8b")
    assert list(active_models()) == ["llama8b", "qwen14b"]

    monkeypatch.setenv("AURA_MODELS", "qwen99b")
    with pytest.raises(ValueError, match="unknown models"):
        active_models()

    monkeypatch.delenv("AURA_MODELS")
    assert len(active_models()) == 4


def test_the_committed_registry_loads():
    """Every cluster job loads subjects.json before running anything, so a field
    the Subject class does not know fails the whole batch in seconds -- which is
    how an unknown "exclusion_reason" failed five negativei2-server arms."""
    from pathlib import Path

    from evaluation.subjects import Registry

    registry = Registry.load(Path(__file__).resolve().parent.parent / "subjects.json")
    assert registry.subjects
