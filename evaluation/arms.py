"""Experimental arms: the evaluation design, expressed as data.

Two things the reviews demanded that v1's design could not express:

* **An isolated ablation.** Reviewer A asked for "resolution strategies
  separately rather than stacked". v1 reported only the nested chain
  (full / no-LLM / linear-only), which cannot tell you whether greedy or the
  LLM is useful *on its own*. We enumerate all seven non-empty subsets of the
  three tiers, so every tier and every pair is measured directly.
* **Baselines other than pip.** ``install_only`` is the stand-in for UPCY's
  regime -- accept a configuration when the resolver is satisfied, without
  running the project's tests. UpCy itself is a Java/Maven tool and cannot be
  run on Python subjects, so we reproduce the *verification regime* it
  represents rather than claiming to have run the tool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from itertools import combinations

TIERS = ("linear", "greedy", "llm")

# Every arm that uses an LLM runs once per model. Both models are open-weight
# and served locally by Ollama: no API key, no per-token cost, and anyone with a
# laptop can rerun the study. The model is part of the arm's name so a result
# can never be attributed to the wrong one.
#
# Three sizes of one family (Qwen2.5-Coder 3b / 14b / 32b) isolate the effect of
# model size; Llama 3.1 8b is the different-family comparison at a small size.
ALL_MODELS = {
    "qwen3b": "ollama:qwen2.5-coder:3b",
    "llama8b": "ollama:llama3.1:8b",
    "qwen14b": "ollama:qwen2.5-coder:14b",
    "qwen32b": "ollama:qwen2.5-coder:32b",
}
# Opt-in only (AURA_MODELS=qwen14b,qwen14bft): Qwen2.5-Coder-14B QLoRA + KTO-tuned
# on AURA's own LLM-tier logs, good and bad cases (scripts/ft/train_kto.py), at
# qwen14b's Q4_K_M quantisation.
# Not part of the main study's model set.
EXTRA_MODELS = {
    "qwen14bft": "ollama:qwen14b-aura-kto:latest",
}


def active_models() -> dict[str, str]:
    """The models this run uses: all of them, or the subset in ``AURA_MODELS``.

    A laptop cannot hold the 32b model, so ``AURA_MODELS=qwen14b,llama8b``
    restricts a local run without editing the study design.
    """
    wanted = [m.strip() for m in os.environ.get("AURA_MODELS", "").split(",") if m.strip()]
    if not wanted:
        return dict(ALL_MODELS)
    known = {**ALL_MODELS, **EXTRA_MODELS}
    unknown = [m for m in wanted if m not in known]
    if unknown:
        raise ValueError(f"AURA_MODELS names unknown models {unknown}; "
                         f"choose from {sorted(known)}")
    return {m: known[m] for m in known if m in wanted}


MODELS = active_models()
# The model the single-model controls (cost, prompting, ordering, oracle) use.
PRIMARY_MODEL = "qwen14b"
FULL_ARM = f"aura_full_{PRIMARY_MODEL}"


def _primary(models: dict[str, str]) -> str:
    return models.get(PRIMARY_MODEL) or next(iter(models.values()))


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str                       # "baseline" | "aura"
    tiers: tuple[str, ...] = ()
    ordering: str = "spread"
    ordering_seed: int = 0
    llm_spec: str = "none"          # see aura.llm.build_client
    explanation_first: bool = True
    # The compact prompt is the default: with the small local models this study
    # uses, the full prompt made them declare ordinary co-update conflicts
    # "unresolvable". The full prompt is kept as a named comparison arm.
    prompt_style: str = "compact"
    installer: str = "pip"
    require_metric_parity: bool = True
    validate: bool = True           # False = install-only (the UPCY regime)
    batching: bool = True           # probe k candidates concurrently
    batch_k: int = 0                # >0 pins k; 1 is the determinism control
    measure_coverage: bool = False  # instrument validation to size the oracle
    constraint_graph: bool = True   # read declared Requires-Dist coupling
    routing: bool = True            # let the failure class choose the next tier
    upgrade_source: str = "requirements"   # or "primary" for the relaxed spec
    description: str = ""
    group: str = ""                 # which research question this serves

    @property
    def uses_llm(self) -> bool:
        return "llm" in self.tiers


# --------------------------------------------------------------- baselines

def baseline_arms(llm_spec: str = "none") -> list[Arm]:
    return [
        Arm("baseline", "baseline", validate=True, upgrade_source="requirements",
            group="inclusion",
            description="Install the repo's own pinned requirements and validate. "
                        "Subjects that fail here are excluded from the study."),
        Arm("pip_upgrade", "baseline", upgrade_source="primary", group="rq1",
            description="pip install --upgrade against the relaxed spec: the "
                        "naive 'upgrade everything' floor."),
        Arm("uv_upgrade", "baseline", upgrade_source="primary", installer="uv",
            group="rq1",
            description="The same one-shot upgrade through uv's resolver, to "
                        "separate resolver quality from the upgrade strategy."),
        Arm("install_only", "aura", tiers=TIERS, llm_spec=llm_spec, validate=False,
            group="rq_baselines",
            description="AURA's search accepting on installation success alone, "
                        "with no test execution. Stand-in for UPCY's regime: "
                        "structurally valid but unverified."),
        # Real Dependabot, run on GitHub on a private copy of the subject: its
        # PRs, the subject's tests as CI, and auto-merge when they pass. Not
        # run by this harness -- scripts/real_dependabot.py collects the
        # outcome into the same result files every other arm writes.
        Arm("dependabot", "baseline", upgrade_source="github",
            group="rq_baselines",
            description="Real Dependabot on GitHub: one PR per dependency, the "
                        "subject's tests as CI, PRs merged when they pass, "
                        "conflicting PRs rebased by Dependabot. Collected with "
                        "scripts/real_dependabot.py."),
        # Real Renovate, the same way, with its recommended configuration:
        # packages released together are grouped into one PR.
        Arm("renovate", "baseline", upgrade_source="github",
            group="rq_baselines",
            description="Real Renovate on GitHub, recommended config (grouped "
                        "monorepo updates), the subject's tests as CI, PRs merged "
                        "when they pass. Collected with scripts/real_renovate.py."),
        # Renovate's real proposals (renovate_proposals.py, the Renovate CLI run
        # offline) tested and merged in rounds by this harness, as on GitHub.
        Arm("renovate_replay", "baseline", upgrade_source="renovate_replay",
            group="rq_baselines",
            description="Renovate (recommended config): its proposals from the "
                        "Renovate CLI, each tested with the subject's tests and "
                        "merged when they pass, in rounds until none passes."),
        Arm("pip_compile", "baseline", upgrade_source="compiled",
            group="rq_baselines",
            description="pip-tools resolving the unpinned specification to a "
                        "coherent pinned set, validated once. A stronger "
                        "resolver baseline than bare pip install --upgrade."),
    ]


# ------------------------------------------------------- ablation (isolated)

def ablation_arms(models: dict[str, str]) -> list[Arm]:
    """Every proper subset of the three tiers, with routing on throughout.

    Routing is part of the system under test, not a variable the ablation holds
    out: it is enabled in every arm, so each comparison isolates a *tier* while
    everything else stays fixed. That keeps the subtraction meaningful --
    ``aura_linear_greedy`` minus ``aura_linear`` is what greedy adds, both
    measured in the world the tool actually runs in.

    Only the one- and two-tier subsets are generated. The three-tier subset is
    ``aura_full`` itself, and with routing on everywhere the two would be
    byte-identical configurations; running both would spend a second full pass
    over every subject to reproduce a result already in hand.

    The cost of enabling routing everywhere is that no arm here can measure
    routing's own contribution. ``aura_no_routing`` exists for that and is meant
    for a subset of subjects rather than the whole campaign.
    """
    arms = []
    for size in (1, 2):
        for combo in combinations(TIERS, size):
            base = "aura_" + "_".join(combo)
            variants = ([(f"{base}_{label}", spec, f" using {label}")
                         for label, spec in models.items()]
                        if "llm" in combo else [(base, "none", "")])
            for name, spec, using in variants:
                arms.append(Arm(
                    name=name, kind="aura", tiers=combo, llm_spec=spec,
                    routing=True, group="rq_ablation",
                    description="AURA with tiers: " + ", ".join(combo) + using
                                + " (routing on, as in the shipped system)",
                ))
    return arms


# ------------------------------------------------------------- sensitivity

def ordering_arms(llm_spec: str, seeds: tuple[int, ...] = (1, 2, 3)) -> list[Arm]:
    """Order sensitivity: does the answer depend on which package we try first?"""
    arms = [
        Arm(f"aura_order_{name}", "aura", tiers=TIERS, ordering=name,
            llm_spec=llm_spec, group="rq_order",
            description=f"Full AURA, packages ordered by {name}.")
        for name in ("version_jump", "import_statements", "fanout", "alphabetical")
    ]
    arms += [
        Arm(f"aura_order_random_{seed}", "aura", tiers=TIERS, ordering="random",
            ordering_seed=seed, llm_spec=llm_spec, group="rq_order",
            description=f"Full AURA, random package order (seed {seed}).")
        for seed in seeds
    ]
    return arms


def main_arm(models: dict[str, str]) -> list[Arm]:
    """AURA as proposed: every tier, declared constraints read, routing on.

    This is the three-tier arm; the ablation lattice generates only the proper
    subsets, because with routing enabled everywhere a three-tier lattice arm
    would be this same configuration under another name.
    """
    return [
        Arm(f"aura_full_{label}", "aura", tiers=TIERS, llm_spec=spec,
            constraint_graph=True, routing=True, group="rq_main",
            description=f"AURA as proposed, using {label}: all three tiers, "
                        "declared-constraint reading, and failure-aware routing.")
        for label, spec in models.items()
    ]


def search_cost_arms(llm_spec: str) -> list[Arm]:
    """Do the search-cost reductions change the *answer*, or only the runtime?

    Both features exist to avoid probes, and either could in principle lose a
    solution: the graph by declaring a scan futile, routing by acting on that
    declaration. The claim to defend is that the final configurations match
    while the probe counts do not -- the same shape as the ``k=1`` vs
    ``k=production`` determinism check for batched probing. A saving nobody
    verified is a saving nobody should believe.

    Each control differs from ``aura_full`` in exactly one setting, so the
    comparison attributes the difference to that setting and nothing else:

    * ``aura_no_routing``  routing off   -- what failure-aware routing buys
    * ``aura_no_graph``    graph off     -- what reading declared metadata buys
    * ``aura_serial``      k = 1         -- what batching buys

    These are controls, not part of the per-subject set: run them on a subset
    large enough to support the claim. At hours per subject-arm, the campaign
    budget is better spent on full coverage of the arms that carry the results.
    """
    return [
        Arm("aura_no_graph", "aura", tiers=TIERS, llm_spec=llm_spec,
            constraint_graph=False, routing=False, group="rq_cost",
            description="No declared-constraint reading at all: coupling is "
                        "discovered only by failing, as in v1."),
        Arm("aura_no_routing", "aura", tiers=TIERS, llm_spec=llm_spec,
            routing=False, group="rq_cost",
            description="Every tier, with failure-aware routing disabled. The "
                        "control for routing's own effect: run on a subset to "
                        "show the final configurations match while the probe "
                        "count falls."),
        Arm("aura_serial", "aura", tiers=TIERS, llm_spec=llm_spec,
            batch_k=1, group="rq_cost",
            description="k=1: one candidate probed at a time. The reference "
                        "semantics batching must reproduce exactly."),
    ]


def coverage_arm(llm_spec: str) -> list[Arm]:
    """Size the oracle for each subject: what does 'validation passed' cover?"""
    return [
        Arm("aura_coverage", "aura", tiers=TIERS, llm_spec=llm_spec,
            measure_coverage=True, group="rq_oracle",
            description="Full AURA, instrumenting validation at baseline and at "
                        "the accepted result to measure repo and per-dependency "
                        "coverage."),
    ]


def prompting_arms(llm_spec: str) -> list[Arm]:
    """Does prompt design change the answer, and by how much relative to model size?"""
    return [
        Arm("aura_no_explanation", "aura", tiers=TIERS, llm_spec=llm_spec,
            explanation_first=False, group="rq_prompting",
            description="Full AURA asking for a plan directly, with no explanation."),
        Arm("aura_long_prompt", "aura", tiers=TIERS, llm_spec=llm_spec,
            prompt_style="full", group="rq_prompting",
            description="Full AURA with the original long prompt instead of the "
                        "compact default. On small models it gives up early on "
                        "conflicts a co-update would fix; this arm measures how much."),
    ]


def oracle_arms(llm_spec: str) -> list[Arm]:
    """Isolate what the behavioural check costs us in apparent success."""
    return [
        Arm("aura_tests_only", "aura", tiers=TIERS, llm_spec=llm_spec,
            require_metric_parity=False, group="rq_oracle",
            description="Full AURA accepting on test outcome alone, without the "
                        "numeric-drift check. The difference against aura_full is "
                        "the number of 'successes' that silently changed results."),
    ]


def oneshot_arms(models: dict[str, str]) -> list[Arm]:
    """The model asked once for the whole upgraded file, with no feedback.

    The alternative a reader will raise first: why not just ask the model? Run
    against the same models as the LLM tier, it separates what a model knows
    from what iterating against test results contributes. See
    evaluation/oneshot.py.
    """
    return [
        Arm(f"llm_oneshot_{label}", "baseline", upgrade_source="llm_oneshot",
            llm_spec=spec, group="rq_baselines",
            description=f"{spec} proposes every upgrade in a single answer, "
                        "given the releases that install on the project's "
                        "Python; kept only if the project's tests still pass.")
        for label, spec in models.items()
    ]


def full_suite(models: dict[str, str] | None = None) -> list[Arm]:
    """Every arm, deduplicated by name, in execution order."""
    models = dict(models or MODELS)
    primary = _primary(models)
    arms = (
        baseline_arms(primary)
        + oneshot_arms(models)
        + main_arm(models)
        + ablation_arms(models)
        + oracle_arms(primary)
        + prompting_arms(primary)
        + ordering_arms(primary)
        + search_cost_arms(primary)
        + coverage_arm(primary)
    )
    seen, unique = set(), []
    for arm in arms:
        if arm.name in seen:
            continue
        seen.add(arm.name)
        unique.append(arm)
    return unique


# The set run on *every* subject. Paired tests need the same subjects on both
# sides, so full coverage of a reduced set beats partial coverage of a larger
# one. Routing is on throughout, because it is part of the tool rather than a
# variable to hold out.
#
# The dependabot arm is not in this set: it runs on GitHub, per subject:
#   python scripts/real_dependabot.py setup|collect <subject>
#
#   baseline              the inclusion gate and the reference
#   pip_upgrade           "just upgrade everything"
#   aura_linear_greedy    deterministic search only (no model)
#   aura_llm_<model>      the model with no deterministic search: is the search
#                         necessary at all? A reviewer will ask.
#   aura_full_<model>     the shipped system; minus aura_linear_greedy is what
#                         the model adds, minus aura_llm_<model> is what the
#                         search adds, and the models side by side show how much
#                         of the result depends on which model was used
#
# Every model runs end to end, because the findings that matter are properties
# of a whole run rather than of a single answer: a model that advances as many
# packages as the best one while ending far from the newest releases, and the
# ten-fold difference in model calls between the full system and the model
# alone. Replaying recorded decision points (python -m evaluation.replay) asks
# every model the identical question and is a complement to these arms, not a
# replacement -- it has no trajectory and therefore no cost or quality of one.
#
# The cost is affordable because arms run as separate jobs: one subject is
# ~5 h of compute, or ~2 h of wall clock at four tasks in parallel.
ESSENTIAL_ARMS = (
    "baseline",
    "pip_upgrade",
    "aura_linear_greedy",
    *(f"aura_llm_{label}" for label in MODELS),
    *(f"aura_full_{label}" for label in MODELS),
)

# The one-shot arms, runnable on their own so they can be added to subjects
# already finished without re-running anything else (ARMS=oneshot).
ONESHOT_ARMS = tuple(f"llm_oneshot_{label}" for label in MODELS)

CORE_ARMS = (
    "baseline", "pip_upgrade", "install_only",
    *(f"aura_full_{label}" for label in MODELS),
    "aura_linear", "aura_greedy", "aura_linear_greedy",
    *(f"aura_{combo}_{label}" for combo in ("llm", "linear_llm", "greedy_llm")
      for label in MODELS),
)


def by_name(name: str, models: dict[str, str] | None = None) -> Arm:
    for arm in full_suite(models):
        if arm.name == name:
            return arm
    raise KeyError(f"unknown arm {name!r}")
