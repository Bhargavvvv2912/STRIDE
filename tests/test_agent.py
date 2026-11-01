"""Agent tests: outcomes, tier gating, truncation, and ordering effects."""

from contextlib import nullcontext

import pytest

from aura.agent import AuraAgent
from aura.llm import LLMUnavailable, NullLLM, ScriptedLLM
from aura.reqfile import safe_parse_version
from aura.telemetry import Outcome, RunConfig, RunLog, read_log
from aura.environment import EnvSpec, ErrorClass
from tests.conftest import FakeWorld, llm_reply

RELEASES = {
    "torch": ["1.12.0", "1.12.1", "1.13.0", "2.0.0"],
    "torchvision": ["0.13.0", "0.13.1", "0.14.0"],
    "numpy": ["1.22.0", "1.24.0", "1.26.4"],
}


# ``batch_k=1`` by default: most fixtures use a stateful predicate whose answer
# depends on what was probed before, which concurrent probing races. Batching is
# exercised deliberately in the tests that target it.
def make_agent(monkeypatch, tmp_path, world, *, tiers, llm=None, config=None,
               ordering="spread", max_passes=3, timeout_s=3600,
               python_version="", batch_k=1):
    monkeypatch.setattr("aura.agent.probe", world.probe)
    monkeypatch.setattr("aura.agent.dry_run_session",
                        lambda **k: nullcontext(lambda reqfile: (True, "")))
    rc = RunConfig(repo="demo", arm="test", tiers=tuple(tiers), ordering=ordering,
                   max_passes=max_passes, timeout_s=timeout_s,
                   python_version=python_version, batch_k=batch_k)
    return AuraAgent(
        config, run_config=rc, llm=llm or NullLLM(), index=world.index(),
        run_log=RunLog(tmp_path / "run.jsonl", rc),
        env_spec=EnvSpec(), workdir=tmp_path,
    )


# ------------------------------------------------------------------- outcomes

def test_modernized_when_a_package_advances(monkeypatch, project, config, tmp_path):
    project("torch==1.12.0\nnumpy==1.22.0\n")
    world = FakeWorld(releases=RELEASES, accepts=lambda pins: True)
    result = make_agent(monkeypatch, tmp_path, world, tiers=["linear"],
                        config=config).run()
    assert result.outcome is Outcome.MODERNIZED
    assert result.updated["torch"] == ("1.12.0", "2.0.0")
    assert result.updated["numpy"] == ("1.22.0", "1.26.4")


def test_kept_baseline_when_nothing_can_advance(monkeypatch, project, config, tmp_path):
    """The Lasagne case: baseline preserved, no regression, and *not* called success."""
    project("torch==1.12.0\n")
    world = FakeWorld(releases=RELEASES,
                      accepts=lambda pins: pins["torch"] == "1.12.0" or "breaks")
    result = make_agent(monkeypatch, tmp_path, world, tiers=["linear", "greedy"],
                        config=config).run()
    assert result.outcome is Outcome.KEPT_BASELINE
    assert result.updated == {} and result.unchanged == ["torch"]


def test_failed_when_the_baseline_does_not_validate(monkeypatch, project, config, tmp_path):
    """v1 had no reachable failure path at all; this outcome could not occur."""
    project("torch==1.12.0\n")
    world = FakeWorld(releases=RELEASES, baseline_ok=False)
    result = make_agent(monkeypatch, tmp_path, world, tiers=["linear"],
                        config=config).run()
    assert result.outcome is Outcome.FAILED
    assert "baseline" in result.reason


def test_broken_baseline_is_not_silently_repaired_by_default(monkeypatch, project,
                                                             config, tmp_path):
    """v1 stripped all constraints and continued, producing an AURA-made baseline."""
    project("torch==1.12.0\n")
    world = FakeWorld(releases=RELEASES, baseline_ok=False)
    make_agent(monkeypatch, tmp_path, world, tiers=["linear"], config=config).run()
    assert (tmp_path / "requirements.txt").read_text() == "torch==1.12.0\n"


def test_truncated_passes_is_distinguished_from_convergence(monkeypatch, project,
                                                            config, tmp_path):
    """The cap binds while changes are still landing, so this is not convergence.

    The oracle is a pure function of the configuration, as a real one is: the
    same pin set installed twice gives the same answer. Truncation is forced by
    the pass budget rather than by an oracle that changes its mind.
    """
    project("torch==1.12.0\nnumpy==1.22.0\n")
    world = FakeWorld(releases=RELEASES, accepts=lambda pins: True)
    result = make_agent(monkeypatch, tmp_path, world, tiers=["linear"],
                        config=config, max_passes=1).run()

    # Something advanced, and the budget ran out before a pass proved there was
    # nothing left -- which is a different claim from "converged".
    assert result.outcome is Outcome.TRUNCATED_PASSES
    assert result.passes_used == 1 and result.updated


def test_convergence_is_a_pass_that_changed_nothing(monkeypatch, project,
                                                    config, tmp_path):
    """The contrast: the budget was not the reason the run stopped."""
    project("torch==1.12.0\n")
    world = FakeWorld(releases=RELEASES, accepts=lambda pins: True)
    result = make_agent(monkeypatch, tmp_path, world, tiers=["linear"],
                        config=config, max_passes=5).run()

    assert result.outcome is Outcome.MODERNIZED
    assert result.passes_used < 5          # stopped early, on its own


def test_truncated_timeout_is_its_own_outcome(monkeypatch, project, config, tmp_path):
    project("torch==1.12.0\n")
    world = FakeWorld(releases=RELEASES)
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear"], config=config)
    agent.deadline = 0  # already out of budget
    result = agent.run()
    assert result.outcome is Outcome.TRUNCATED_TIMEOUT


def test_all_six_outcomes_are_reachable():
    assert len(set(Outcome)) == 6


# --------------------------------------------------------------- tier gating

def test_deterministic_arm_never_reaches_the_llm(monkeypatch, project, config, tmp_path):
    """The v1 ablation still called the API; here it cannot, by construction."""
    project("torch==1.12.0\n")
    world = FakeWorld(releases=RELEASES, accepts=lambda pins: "always fails")
    null = NullLLM()
    result = make_agent(monkeypatch, tmp_path, world, tiers=["linear", "greedy"],
                        llm=null, config=config).run()
    assert result.outcome is Outcome.KEPT_BASELINE
    assert null.calls_attempted == 0


def test_llm_tier_resolves_what_deterministic_tiers_cannot(monkeypatch, project,
                                                           config, tmp_path):
    project("torch==1.12.0\ntorchvision==0.13.0\n")

    def accepts(pins):
        # Only the coupled intermediate pair works.
        if pins["torch"] == "1.12.1" and pins["torchvision"] == "0.13.1":
            return True
        return "torchvision 0.13.0 depends on torch==1.12.0"

    world = FakeWorld(releases=RELEASES, accepts=accepts)
    llm = ScriptedLLM([llm_reply(["torch==1.12.1", "torchvision==0.13.1"])])
    result = make_agent(monkeypatch, tmp_path, world,
                        tiers=["linear", "greedy", "llm"], llm=llm, config=config).run()
    assert result.outcome is Outcome.MODERNIZED
    assert result.updated["torch"][1] == "1.12.1"
    assert result.updated["torchvision"][1] == "0.13.1"


def test_same_case_fails_without_the_llm_tier(monkeypatch, project, config, tmp_path):
    """The paired half of the ablation, on identical inputs."""
    project("torch==1.12.0\ntorchvision==0.13.0\n")

    def accepts(pins):
        if pins["torch"] == "1.12.1" and pins["torchvision"] == "0.13.1":
            return True
        return "torchvision 0.13.0 depends on torch==1.12.0"

    world = FakeWorld(releases=RELEASES, accepts=accepts)
    result = make_agent(monkeypatch, tmp_path, world, tiers=["linear", "greedy"],
                        config=config).run()
    assert result.outcome is Outcome.KEPT_BASELINE


def test_greedy_tier_co_resolves_a_coupled_pair(monkeypatch, project, config, tmp_path):
    project("torch==1.12.0\ntorchvision==0.13.0\n")

    def accepts(pins):
        if pins["torch"] == "2.0.0" and pins["torchvision"] == "0.14.0":
            return True
        return "torchvision 0.13.0 depends on torch==1.12.0"

    world = FakeWorld(releases=RELEASES, accepts=accepts)
    result = make_agent(monkeypatch, tmp_path, world, tiers=["greedy"],
                        config=config).run()
    assert result.outcome is Outcome.MODERNIZED
    assert result.updated["torchvision"][1] == "0.14.0"


def test_llm_only_arm_runs_without_the_deterministic_tiers(monkeypatch, project,
                                                           config, tmp_path):
    project("torch==1.12.0\n")
    world = FakeWorld(releases=RELEASES,
                      accepts=lambda pins: pins["torch"] == "1.13.0" or "nope")
    llm = ScriptedLLM([llm_reply(["torch==1.13.0"])])
    result = make_agent(monkeypatch, tmp_path, world, tiers=["llm"],
                        llm=llm, config=config).run()
    assert result.outcome is Outcome.MODERNIZED
    assert result.updated["torch"][1] == "1.13.0"


def test_an_llm_only_arm_installs_nothing_before_the_model_is_asked(
        monkeypatch, project, config, tmp_path):
    """No "try the newest" step: where to start is the model's decision, so
    its first request carries no error and nothing but the baseline has been
    installed."""
    project("torch==1.12.0\n")
    probed = []

    def accepts(pins):
        probed.append(dict(pins))
        return pins["torch"] in ("1.12.0", "1.13.0") or "nope"

    world = FakeWorld(releases=RELEASES, accepts=accepts)
    llm = ScriptedLLM([llm_reply(["torch==1.13.0"])])
    result = make_agent(monkeypatch, tmp_path, world, tiers=["llm"], llm=llm,
                        config=config, max_passes=1).run()
    assert result.updated["torch"][1] == "1.13.0"
    assert [p["torch"] for p in probed] == ["1.13.0"]    # only the model's plan
    assert "Nothing has been tried yet" in llm.prompts[0]
    assert "ERROR" not in llm.prompts[0].split("PACKAGES")[0]


def test_the_linear_scan_starts_at_the_newest_version(monkeypatch, project,
                                                      config, tmp_path):
    """The newest version is the scan's first step, logged as linear."""
    project("torch==1.12.0\n")
    world = FakeWorld(releases=RELEASES, accepts=lambda pins: True)
    make_agent(monkeypatch, tmp_path, world, tiers=["linear"], config=config,
               max_passes=1).run()
    attempts = [e for e in read_log(tmp_path / "run.jsonl")
                if e.get("event") == "attempt" and e.get("package") == "torch"]
    assert attempts[0]["tier"] == "linear"
    assert attempts[0]["plan"] == {"torch": "2.0.0"}
    assert not any(e.get("tier") == "direct" for e in read_log(tmp_path / "run.jsonl"))


# ------------------------------------------------------------------- ordering

def test_ordering_choice_is_recorded_and_changes_the_queue(monkeypatch, project,
                                                           config, tmp_path):
    from aura.telemetry import read_log
    project("torch==1.12.0\nnumpy==1.22.0\n",
            source="import torch\nimport torch.nn\nimport numpy\n")
    world = FakeWorld(releases=RELEASES, accepts=lambda pins: "no")
    make_agent(monkeypatch, tmp_path, world, tiers=["linear"], config=config,
               ordering="spread").run()
    events = [e for e in read_log(tmp_path / "run.jsonl")
              if e["event"] == "prioritization"]
    assert events and events[0]["ordering"] == "spread"
    assert {q["package"] for q in events[0]["queue"]} == {"torch", "numpy"}


def test_spread_is_computed_from_the_repo_not_the_version_jump(monkeypatch, project,
                                                              config, tmp_path):
    from aura.telemetry import read_log
    # numpy in two files, torch in one -> numpy must outrank torch under spread
    root = project("torch==1.12.0\nnumpy==1.22.0\n", source="import torch\nimport numpy\n")
    (root / "extra.py").write_text("import numpy\n")
    world = FakeWorld(releases=RELEASES, accepts=lambda pins: "no")
    make_agent(monkeypatch, tmp_path, world, tiers=["linear"], config=config).run()
    queue = [e for e in read_log(tmp_path / "run.jsonl")
             if e["event"] == "prioritization"][0]["queue"]
    assert queue[0]["package"] == "numpy"
    assert queue[0]["spread"] == 2 and queue[1]["spread"] == 1


# ------------------------------------------------------------------ telemetry

def test_run_log_captures_attempts_costs_and_version_age(monkeypatch, project,
                                                         config, tmp_path):
    from aura.telemetry import read_log
    project("torch==1.12.0\n")
    world = FakeWorld(releases=RELEASES)
    result = make_agent(monkeypatch, tmp_path, world, tiers=["linear"],
                        config=config).run()
    events = read_log(tmp_path / "run.jsonl")
    assert [e for e in events if e["event"] == "attempt"]
    end = [e for e in events if e["event"] == "run_end"][0]
    assert end["outcome"] == "modernized"
    assert end["version_age"]["before"]["mean_releases_behind"] > \
           end["version_age"]["after"]["mean_releases_behind"]
    assert result.version_age_after["at_latest"] == 1


def test_requirements_file_keeps_its_shape_through_a_run(monkeypatch, project,
                                                         config, tmp_path):
    """Comments, extras and markers must survive the rewrite loop."""
    project(
        "# core\n"
        "torch==1.12.0  # deep learning\n"
        'numpy==1.22.0 ; python_version >= "3.8"\n'
    )
    world = FakeWorld(releases=RELEASES)
    make_agent(monkeypatch, tmp_path, world, tiers=["linear"], config=config).run()
    out = (tmp_path / "requirements.txt").read_text()
    assert out.startswith("# core\n")
    assert "torch==2.0.0  # deep learning" in out
    assert 'numpy==1.26.4 ; python_version >= "3.8"' in out


# ------------------------------------------------------- metadata pre-filtering

def _torch_with_requires_python():
    """torch 2.0.0 declares >=3.10; the rest are open."""
    from aura.index import ReleaseHistory
    from tests.conftest import FakeIndex
    return FakeIndex({"torch": ReleaseHistory(
        name="torch",
        versions=("1.12.0", "1.12.1", "1.13.0", "2.0.0"),
        metadata={"2.0.0": {"requires_python": ">=3.10", "yanked": False}},
    )})


def test_prefilter_skips_versions_the_runtime_cannot_install(monkeypatch, project,
                                                             config, tmp_path):
    """requires-python is free metadata; spending an install to learn it is not."""
    project("torch==1.12.0\n")
    probed = []

    def accepts(pins):
        probed.append(pins["torch"])
        return pins["torch"] == "1.13.0"

    world = FakeWorld(releases={"torch": ["1.12.0", "1.12.1", "1.13.0", "2.0.0"]},
                      accepts=accepts)
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear"],
                       config=config, python_version="3.9")
    agent.index = _torch_with_requires_python()
    result = agent.run()

    assert result.updated["torch"] == ("1.12.0", "1.13.0")
    assert "2.0.0" not in probed          # eliminated without an install
    assert probed[0] == "1.13.0"          # scan still starts from the newest usable


def test_prefilter_keeps_versions_the_runtime_can_install(monkeypatch, project,
                                                          config, tmp_path):
    """The same package on a newer interpreter must lose nothing."""
    project("torch==1.12.0\n")
    probed = []

    def accepts(pins):
        probed.append(pins["torch"])
        return True

    world = FakeWorld(releases={"torch": ["1.12.0", "1.12.1", "1.13.0", "2.0.0"]},
                      accepts=accepts)
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear"],
                       config=config, python_version="3.11")
    agent.index = _torch_with_requires_python()
    result = agent.run()

    assert result.updated["torch"] == ("1.12.0", "2.0.0")
    assert probed[0] == "2.0.0"


def test_prefilter_reports_what_it_eliminated(monkeypatch, project, config, tmp_path):
    """The saving has to be measurable, or it cannot go in the paper.

    One pass, so the count is unambiguous: a later pass scans a different range
    and its eliminations are separate saved installs, not a double charge.
    """
    project("torch==1.12.0\n")
    world = FakeWorld(releases={"torch": ["1.12.0", "1.12.1", "1.13.0", "2.0.0"]},
                      accepts=lambda pins: True)
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear"],
                       config=config, python_version="3.9", max_passes=1)
    agent.index = _torch_with_requires_python()
    result = agent.run()

    assert result.counters.get("prefiltered_versions") == 1
    notes = [e for e in read_log(tmp_path / "run.jsonl")
             if e.get("note") == "metadata_prefilter"]
    assert notes and notes[0]["eliminated"] == 1 and notes[0]["remaining"] == 2


# ----------------------------------------------------- failure-aware routing

TV_RELEASES = {"torch": ["1.12.0", "1.12.1", "1.13.0", "2.0.0"],
               "torchvision": ["0.13.0", "0.14.0", "0.15.1"]}

# The real declared pins: torchvision fixes torch exactly.
TV_REQUIRES = {("torchvision", "0.13.0"): ["torch (==1.12.0)"],
               ("torchvision", "0.14.0"): ["torch (==1.13.0)"],
               ("torchvision", "0.15.1"): ["torch (==2.0.0)"]}


def _coupled_agent(monkeypatch, tmp_path, config, accepts, *, requires=TV_REQUIRES,
                   routing=True, tiers=("linear", "greedy")):
    from tests.conftest import FakeIndex, history
    world = FakeWorld(releases=TV_RELEASES, accepts=accepts)
    agent = make_agent(monkeypatch, tmp_path, world, tiers=list(tiers),
                       config=config, ordering="alphabetical")
    agent.rc.routing = routing
    agent.index = FakeIndex(
        {n: history(n, v) for n, v in TV_RELEASES.items()}, requires=requires)
    return agent


def _conflict_unless_paired(probed):
    """torchvision 0.13.0 pins torch==1.12.0, so torch alone always conflicts."""
    def accepts(pins):
        probed.append(dict(pins))
        tv = pins.get("torchvision", "0.13.0")
        want = {"0.13.0": "1.12.0", "0.14.0": "1.13.0", "0.15.1": "2.0.0"}[tv]
        if pins.get("torch", "1.12.0") != want:
            return f"torchvision {tv} depends on torch=={want}"
        return True
    return accepts


def test_exact_pin_routes_past_a_futile_scan(monkeypatch, project, config, tmp_path):
    """Every torch above the pin fails identically; scanning them is pure waste."""
    project("torch==1.12.0\ntorchvision==0.13.0\n")
    probed = []
    agent = _coupled_agent(monkeypatch, tmp_path, config,
                           _conflict_unless_paired(probed))
    result = agent.run()

    routed = [e for e in read_log(tmp_path / "run.jsonl")
              if e.get("event") == "routing" and e.get("chose") == "greedy"]
    assert routed, "routing never fired on the conflict"
    assert routed[0]["permitted"] == 0

    # The scan was skipped: with torchvision still at 0.13.0, no intermediate
    # torch was ever probed. Those are exactly the installs that would have
    # failed identically on the same declared pin.
    scan_probes = [p for p in probed
                   if p.get("torchvision") == "0.13.0"
                   and p.get("torch") in ("1.12.1", "1.13.0")]
    assert scan_probes == [], scan_probes
    assert result.updated["torch"][1] == "2.0.0"
    assert result.updated["torchvision"][1] == "0.15.1"


def test_a_ceiling_does_not_route_because_the_scan_can_still_win(monkeypatch, project,
                                                                 config, tmp_path):
    """`torch<1.14` leaves 1.13.0 reachable; skipping linear would lose it."""
    project("torch==1.12.0\ntorchvision==0.13.0\n")
    probed = []

    def accepts(pins):
        probed.append(dict(pins))
        if "torchvision" in pins and pins["torchvision"] != "0.13.0":
            return "torchvision upgrade unsupported here"
        return safe_parse_version(pins.get("torch", "1.12.0")) < safe_parse_version("1.14")

    agent = _coupled_agent(
        monkeypatch, tmp_path, config, accepts,
        requires={("torchvision", "0.13.0"): ["torch (<1.14)"]})
    result = agent.run()

    routed = [e for e in read_log(tmp_path / "run.jsonl")
              if e.get("event") == "routing" and e.get("chose") == "greedy"]
    assert not routed, "routed away from a scan that had a solution"
    assert result.updated["torch"][1] == "1.13.0"


def test_routing_off_reproduces_the_unrouted_search(monkeypatch, project,
                                                    config, tmp_path):
    """The ablation arm: same answer, more probes."""
    project("torch==1.12.0\ntorchvision==0.13.0\n")
    on, off = [], []
    a = _coupled_agent(monkeypatch, tmp_path, config, _conflict_unless_paired(on))
    result_on = a.run()

    project("torch==1.12.0\ntorchvision==0.13.0\n")
    b = _coupled_agent(monkeypatch, tmp_path, config, _conflict_unless_paired(off),
                       routing=False)
    result_off = b.run()

    assert result_on.updated == result_off.updated      # same answer
    assert len(on) < len(off)                            # fewer probes


def test_unknown_metadata_never_routes_away_from_the_scan(monkeypatch, project,
                                                          config, tmp_path):
    """No metadata means no information, so the scan must still run."""
    project("torch==1.12.0\ntorchvision==0.13.0\n")
    probed = []
    agent = _coupled_agent(monkeypatch, tmp_path, config,
                           _conflict_unless_paired(probed), requires={})
    agent.run()
    routed = [e for e in read_log(tmp_path / "run.jsonl")
              if e.get("event") == "routing" and e.get("chose") == "greedy"]
    assert not routed


# ------------------------------------------------- not re-learning known answers

def test_a_configuration_is_never_probed_twice(monkeypatch, project, config,
                                               tmp_path):
    """Tiers escalate by widening the plan; when one cannot widen it would
    otherwise re-run the exact pin set the linear scan already rejected."""
    project("torch==1.12.0\n")
    seen = []

    def accepts(pins):
        seen.append(tuple(sorted(pins.items())))
        return False                      # nothing works, so every tier is tried

    world = FakeWorld(releases=RELEASES, accepts=accepts)
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear", "greedy"],
                       config=config, max_passes=1)
    agent.run()

    assert len(seen) == len(set(seen)), "the same configuration was installed twice"


def test_greedy_does_not_run_when_it_has_no_blocker_to_co_resolve(monkeypatch,
                                                                  project, config,
                                                                  tmp_path):
    """A test failure names no packages, so greedy has nothing to widen with and
    its 'group' is the single-package plan the linear scan already rejected."""
    project("torch==1.12.0\n")
    world = FakeWorld(releases=RELEASES,
                      accepts=lambda pins: ("tests failed",
                                            ErrorClass.VALIDATION_FAILED))
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear", "greedy"],
                       config=config, max_passes=1)
    result = agent.run()

    notes = [e for e in read_log(tmp_path / "run.jsonl")
             if e.get("note") == "greedy_stopped" and e.get("rounds") == 0]
    assert notes, "greedy should have reported that it had nothing to do"
    assert "tests fail" in notes[0]["reason"]
    assert result.counters.get("greedy_skipped_no_group")
    # The target alone was already installed by the scan; nothing new ran.
    assert result.counters.get("attempts_greedy") is None


# ----------------------------------------- greedy picks versions that install

def _report(requested, source, source_ver, spec):
    return ("ERROR: Cannot install because these package versions have "
            "conflicting dependencies.\n\nThe conflict is caused by:\n"
            f"    The user requested {requested}\n"
            f"    {source} {source_ver} depends on {spec}\n\n"
            "ERROR: ResolutionImpossible\n", ErrorClass.RESOLUTION_CONFLICT)


def test_greedy_moves_a_blocker_only_to_a_version_this_python_can_install(
        monkeypatch, project, config, tmp_path):
    """click 8.5 needs Python 3.10. On 3.9, greedy must propose 8.1.8 --
    proposing 8.5 spends an install learning what the index already said."""
    from aura.index import ReleaseHistory
    from tests.conftest import FakeIndex

    project("attrs==19.0\nclick==7.0\n", source="import attrs\nimport click\n")
    probed = []

    def accepts(pins):
        probed.append(dict(pins))
        if pins.get("click") == "8.5.0":
            return ("ERROR: No matching distribution found for click==8.5.0",
                    ErrorClass.PYTHON_VERSION)
        attrs_new = pins.get("attrs") == "26.0"
        click_new = pins.get("click") == "8.1.8"
        if attrs_new and not click_new:
            return _report("attrs==26.0", "click", "7.0", "attrs<20")
        if click_new and not attrs_new:
            return _report("click==8.1.8", "attrs", "19.0", "click<8")
        return True

    world = FakeWorld(releases={"attrs": ["19.0", "26.0"],
                                "click": ["7.0", "8.1.8", "8.5.0"]},
                      accepts=accepts)
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear", "greedy"],
                       config=config, ordering="alphabetical",
                       python_version="3.9")
    agent.index = FakeIndex({
        "attrs": ReleaseHistory("attrs", ("19.0", "26.0")),
        "click": ReleaseHistory(
            "click", ("7.0", "8.1.8", "8.5.0"),
            metadata={"8.5.0": {"requires_python": ">=3.10", "yanked": False}}),
    })
    result = agent.run()

    assert result.updated["attrs"][1] == "26.0"
    assert result.updated["click"][1] == "8.1.8"
    assert not any(p.get("click") == "8.5.0" for p in probed)


def test_a_constraint_added_mid_pass_reaches_the_graph(monkeypatch, project,
                                                       config, tmp_path):
    """flask 1.1.1 does not cap itsdangerous; 1.1.4 does. Once flask moves in
    this pass, the next package must be planned against the new cap."""
    from tests.conftest import FakeIndex, history

    project("flask==1.1.1\nitsdangerous==1.1.0\n",
            source="import flask\nimport itsdangerous\n")

    def accepts(pins):
        if pins.get("flask") == "1.1.4" and pins.get("itsdangerous", "1.1.0") >= "2":
            return _report(f"itsdangerous=={pins['itsdangerous']}",
                           "flask", "1.1.4", "itsdangerous<2.0")
        return True

    releases = {"flask": ["1.1.1", "1.1.4"],
                "itsdangerous": ["1.1.0", "2.0.0", "2.2.0"]}
    world = FakeWorld(releases=releases, accepts=accepts)
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear", "greedy"],
                       config=config, ordering="alphabetical", max_passes=1)
    agent.index = FakeIndex(
        {n: history(n, v) for n, v in releases.items()},
        requires={("flask", "1.1.1"): ["itsdangerous (>=0.24)"],
                  ("flask", "1.1.4"): ["itsdangerous (<2.0,>=0.24)"],
                  ("itsdangerous", "1.1.0"): []})
    agent.run()

    routed = [e for e in read_log(tmp_path / "run.jsonl")
              if e.get("event") == "routing" and e.get("package") == "itsdangerous"]
    assert routed and routed[0]["chose"] == "greedy", \
        "the graph did not learn flask 1.1.4's cap before itsdangerous was planned"


# ------------------------------------------------ one install, not one per blocker

FLASK_REQ = {
    ("flask", "1.1.4"): {"itsdangerous": "<2.0", "click": "<8", "jinja2": "<3",
                         "werkzeug": "<2"},
    ("flask", "3.1.3"): {"itsdangerous": ">=2.2", "click": ">=8.1.3",
                         "jinja2": ">=3.1.2", "werkzeug": ">=3.1"},
}
FLASK_RELEASES = {"flask": ["1.1.4", "3.1.3"], "itsdangerous": ["1.1.0", "2.2.0"],
                  "click": ["7.0", "8.1.8"], "jinja2": ["2.11.3", "3.1.6"],
                  "werkzeug": ["1.0.1", "3.1.8"]}


def _pip_like(table, pins):
    """Resolve like pip: stop at the first violated requirement and report it."""
    from packaging.specifiers import SpecifierSet
    for (pkg, ver), deps in table.items():
        if pins.get(pkg) != ver:
            continue
        for dep, spec in deps.items():
            if dep in pins and pins[dep] not in SpecifierSet(spec):
                return _report(f"{dep}=={pins[dep]}", pkg, ver, f"{dep}{spec}")
    return True


def _flask_agent(monkeypatch, tmp_path, config, table, declared, dry_run=None):
    from tests.conftest import FakeIndex, history
    world = FakeWorld(releases=FLASK_RELEASES, accepts=lambda p: _pip_like(table, p))
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear", "greedy"],
                       config=config, ordering="alphabetical", max_passes=1)
    agent.index = FakeIndex(
        {n: history(n, v) for n, v in FLASK_RELEASES.items()},
        requires={**{(p, v): [f"{d} ({s})" for d, s in deps.items()]
                     for (p, v), deps in declared.items()},
                  **{(n, v): [] for n, vs in FLASK_RELEASES.items() for v in vs
                     if n != "flask"}})
    if dry_run is not None:
        monkeypatch.setattr("aura.agent.dry_run_session",
                            lambda **k: nullcontext(dry_run))
    return agent


def test_greedy_adds_one_blocker_per_round_as_the_resolver_names_them(
        monkeypatch, project, config, tmp_path):
    """The original paper's procedure: each round, the resolver names the next
    blocker, which joins the group at its newest version, until it installs."""
    project("click==7.0\nflask==1.1.4\nitsdangerous==1.1.0\njinja2==2.11.3\n"
            "werkzeug==1.0.1\n",
            source="import click, flask, itsdangerous, jinja2, werkzeug\n")
    agent = _flask_agent(monkeypatch, tmp_path, config, FLASK_REQ, FLASK_REQ)
    result = agent.run()

    assert {n: v[1] for n, v in result.updated.items()} == {
        "click": "8.1.8", "flask": "3.1.3", "itsdangerous": "2.2.0",
        "jinja2": "3.1.6", "werkzeug": "3.1.8"}

    rounds = [e for e in read_log(tmp_path / "run.jsonl")
              if e.get("note") == "greedy_round"]
    assert [[a["package"] for a in r["added"]] for r in rounds] == [
        ["flask"], ["itsdangerous"], ["jinja2"], ["werkzeug"]]
    assert rounds[0]["added"][0]["because"] == "flask 1.1.4 depends on click<8"
    assert result.counters.get("attempts_greedy") == 4        # one per round


def test_a_round_that_only_reveals_a_blocker_costs_a_dry_run(monkeypatch,
                                                             project, config,
                                                             tmp_path):
    """Rounds still add one blocker at a time, but a plan the resolver rejects
    is caught by a dry run -- only the plan that resolves is installed."""
    project("click==7.0\nflask==1.1.4\nitsdangerous==1.1.0\njinja2==2.11.3\n"
            "werkzeug==1.0.1\n",
            source="import click, flask, itsdangerous, jinja2, werkzeug\n")
    declared = FLASK_REQ

    def resolve(reqfile):
        verdict = _pip_like(FLASK_REQ, reqfile.pins())
        return (True, "") if verdict is True else (False, verdict[0])

    agent = _flask_agent(monkeypatch, tmp_path, config, FLASK_REQ, declared,
                         dry_run=resolve)
    result = agent.run()

    assert result.updated["werkzeug"][1] == "3.1.8"
    # The target alone, then three rounds that only reveal the next blocker.
    assert result.counters.get("greedy_dry_run_rejections") == 4
    assert result.counters.get("attempts_greedy") == 1


def test_greedy_is_not_cut_off_after_five_rounds(monkeypatch, project, config,
                                                 tmp_path):
    """A seven-package chain needs seven rounds; the old limit of five would
    have stopped with a solution in reach."""
    names = [f"pkg{i}" for i in range(8)]
    table = {(names[i], "2.0"): {names[i + 1]: ">=2.0"} for i in range(7)}
    releases = {n: ["1.0", "2.0"] for n in names}
    project("".join(f"{n}==1.0\n" for n in names),
            source="".join(f"import {n}\n" for n in names))

    from tests.conftest import FakeIndex, history
    world = FakeWorld(releases=releases, accepts=lambda p: _pip_like(table, p))
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear", "greedy"],
                       config=config, ordering="alphabetical", max_passes=1)
    agent.index = FakeIndex({n: history(n, v) for n, v in releases.items()})
    result = agent.run()

    assert all(result.updated[n][1] == "2.0" for n in names)


# ------------------------------------------- baselines that no longer install

class _FakeEnv:
    """Stands in for a probe environment; answers only the closure question."""

    def __init__(self, closure):
        self.closure = closure

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def resolve_closure(self, reqfile, workdir):
        return dict(self.closure)


def _repairable_agent(monkeypatch, tmp_path, config, closure, baseline_ok):
    from tests.conftest import FakeIndex, history

    world = FakeWorld(releases={"flask": ["1.1.1", "1.1.4", "2.0.3"]},
                      accepts=lambda pins: True)
    world.baseline_ok = baseline_ok
    index = FakeIndex(
        {"flask": history("flask", ["1.1.1", "1.1.4", "2.0.3"]),
         "jinja2": history("jinja2", ["2.11.2", "3.1.6"])},
        dates={"jinja2": {"2.11.2": "2020-04-13", "3.1.6": "2025-03-05"},
               "flask": {"1.1.1": "2019-07-08", "2.0.3": "2022-02-14"}})
    monkeypatch.setattr("aura.agent.Environment",
                        lambda *a, **k: _FakeEnv(closure))
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear"],
                       config=config, max_passes=1)
    agent.index = index
    agent.rc.commit_date = "2020-06-01"
    return agent


def test_a_dead_baseline_is_rebuilt_as_of_the_projects_own_commit(
        monkeypatch, project, config, tmp_path):
    """A file that pins Flask and not Jinja2 installs Jinja2 3.x today and dies.
    The environment it had in 2020 is recoverable, and is the honest baseline."""
    project("flask==1.1.1\n")
    agent = _repairable_agent(
        monkeypatch, tmp_path, config,
        closure={"flask": "1.1.1", "jinja2": "3.1.6"},
        baseline_ok=lambda pins: pins.get("jinja2") == "2.11.2")
    result = agent.run()

    assert result.outcome is not Outcome.FAILED
    assert result.baseline_stage == "reconstructed"
    # The repair moves jinja2 *back* to the version the project had, and the
    # search then moves it forward from there. Two different events, recorded
    # in two different places: mixing them would let reconstruction count as
    # modernization, or hide modernization the search really did.
    assert result.baseline_repaired == {"jinja2": ["3.1.6", "2.11.2"]}
    assert result.updated["jinja2"] == ("2.11.2", "3.1.6")
    assert result.counters["baseline_packages_repaired"] == 1


def test_the_repair_is_identical_for_every_arm(monkeypatch, project, config,
                                               tmp_path):
    """Arms are only comparable if they start from the same file, so the
    digest of the repaired baseline must not depend on the arm."""
    digests = []
    for tiers in (["linear"], ["linear", "greedy"]):
        work = tmp_path / f"arm_{len(digests)}"
        work.mkdir()
        (work / "requirements.txt").write_text("flask==1.1.1\n")
        cfg = {**config, "PROJECT_ROOT": str(work),
               "REQUIREMENTS_FILE": "requirements.txt"}
        agent = _repairable_agent(
            monkeypatch, work, cfg,
            closure={"flask": "1.1.1", "jinja2": "3.1.6"},
            baseline_ok=lambda pins: pins.get("jinja2") == "2.11.2")
        agent.tiers = set(tiers)
        digests.append(agent.run().baseline_digest)
    assert digests[0] and digests[0] == digests[1]


def test_a_baseline_that_cannot_be_rebuilt_still_fails_the_run(
        monkeypatch, project, config, tmp_path):
    """Repair widens the sample; it must never manufacture a subject."""
    project("flask==1.1.1\n")
    agent = _repairable_agent(
        monkeypatch, tmp_path, config,
        closure={"flask": "1.1.1", "jinja2": "3.1.6"},
        baseline_ok=lambda pins: False)
    result = agent.run()
    assert result.outcome is Outcome.FAILED
    assert "baseline" in result.reason


# ------------------------------------------- trading a local maximum for a group

def _flask_click_world(monkeypatch, tmp_path, config, regroup=True):
    """The trap seen on both subjects: the newest Flask installable alone
    declares click<8, so taking it costs click. An older Flask does not."""
    from tests.conftest import FakeIndex, history

    # click has real headroom above the cap, so unlocking it is worth more than
    # the single release flask steps back -- which is the trade being tested.
    releases = {"flask": ["1.1.1", "1.1.2", "1.1.4"],
                "click": ["7.0", "7.1.2", "8.0.0", "8.1.0", "8.1.8"]}

    def accepts(pins):
        flask, click = pins.get("flask"), pins.get("click")
        # flask 1.1.4 declares click<8; 1.1.1 and 1.1.2 do not.
        if flask == "1.1.4" and click and click.startswith("8"):
            return ("ERROR: Cannot install because these package versions have "
                    f"conflicting dependencies.\n\nThe conflict is caused by:\n"
                    f"    The user requested click=={click}\n"
                    "    flask 1.1.4 depends on click<8\n")
        return True

    world = FakeWorld(releases=releases, accepts=accepts)
    index = FakeIndex({n: history(n, v) for n, v in releases.items()},
                      requires={("flask", "1.1.4"): ["click (<8,>=5.1)"],
                                ("flask", "1.1.2"): ["click (>=5.1)"],
                                ("flask", "1.1.1"): ["click (>=5.1)"]},
                      dates={"flask": {"1.1.1": "2019-07-08", "1.1.2": "2020-04-03",
                                       "1.1.4": "2021-05-13"},
                             "click": {"7.0": "2018-09-25", "7.1.2": "2020-04-27",
                                       "8.0.0": "2021-05-11", "8.1.0": "2022-10-07",
                                       "8.1.8": "2024-12-21"}})
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear", "greedy"],
                       config=config, max_passes=3)
    agent.index = index
    agent.rc.regroup = regroup
    return agent


def test_a_local_maximum_is_traded_when_it_blocks_more_than_it_gains(
        monkeypatch, project, config, tmp_path):
    """Escalation never revisits a package the scan advanced, so the pass that
    stops at flask 1.1.4 keeps click at 7.1.2 forever. The regroup asks whether
    stepping flask back -- but not below where the project started -- buys more."""
    project("flask==1.1.1\nclick==7.0\n", source="import flask, click\n")
    agent = _flask_click_world(monkeypatch, tmp_path, config)
    result = agent.run()

    assert result.updated["click"][1] == "8.1.8", "the point of the trade"
    assert result.updated["flask"][0] == "1.1.1"
    # Traded back from the scan's pick, still forward from the project's own pin.
    assert result.updated["flask"][1] in ("1.1.1", "1.1.2")
    assert result.counters.get("regroups_accepted") == 1


def test_without_the_regroup_the_local_maximum_stands(monkeypatch, project,
                                                      config, tmp_path):
    """The behaviour this pass exists to change, kept as the control."""
    project("flask==1.1.1\nclick==7.0\n", source="import flask, click\n")
    agent = _flask_click_world(monkeypatch, tmp_path, config, regroup=False)
    result = agent.run()
    assert result.updated["flask"][1] == "1.1.4"
    assert result.updated["click"][1] == "7.1.2"


def test_a_trade_is_refused_when_it_does_not_modernise_the_whole_set(
        monkeypatch, project, config, tmp_path):
    """No churn: stepping a package back has to buy more than it costs."""
    project("flask==1.1.1\nclick==7.0\n", source="import flask, click\n")
    agent = _flask_click_world(monkeypatch, tmp_path, config)
    # Nothing can reach click 8 here, so no trade can pay for itself.
    agent.index.requires[("flask", "1.1.2")] = ["click (<8,>=5.1)"]
    agent.index.requires[("flask", "1.1.1")] = ["click (<8,>=5.1)"]
    result = agent.run()
    assert not result.counters.get("regroups_accepted")
    assert result.updated["flask"][1] == "1.1.4"


def test_a_trade_is_found_even_when_a_newer_unlocking_version_exists(
        monkeypatch, project, config, tmp_path):
    """The case both real subjects hit. Flask 3.1.3 unlocks click 8 and breaks
    the tests -- greedy has already tried exactly that -- while Flask 1.1.2
    unlocks it by never declaring click<8. Answering with the newest unlocking
    version means no trade is ever attempted and the regroup does nothing."""
    from tests.conftest import FakeIndex, history

    releases = {"flask": ["1.1.1", "1.1.2", "1.1.4", "3.1.3"],
                "click": ["7.0", "7.1.2", "8.0.0", "8.1.8"]}

    def accepts(pins):
        flask, click = pins.get("flask"), pins.get("click")
        if flask == "1.1.4" and click and click.startswith("8"):
            return ("ERROR: Cannot install because these package versions have "
                    "conflicting dependencies.\n\nThe conflict is caused by:\n"
                    f"    The user requested click=={click}\n"
                    "    flask 1.1.4 depends on click<8\n")
        if flask == "3.1.3":
            return "tests fail with flask 3"       # what greedy already found
        return True

    world = FakeWorld(releases=releases, accepts=accepts)
    index = FakeIndex(
        {n: history(n, v) for n, v in releases.items()},
        requires={("flask", "1.1.4"): ["click (<8,>=5.1)"],
                  ("flask", "1.1.2"): ["click (>=5.1)"],
                  ("flask", "1.1.1"): ["click (>=5.1)"],
                  ("flask", "3.1.3"): ["click (>=8.1.3)"]},
        # Real dates, so the trade is judged the way it is in a run: stepping
        # flask back a few months to bring click forward by years.
        dates={"flask": {"1.1.1": "2019-07-08", "1.1.2": "2020-04-03",
                         "1.1.4": "2021-05-13", "3.1.3": "2025-08-19"},
               "click": {"7.0": "2018-09-25", "7.1.2": "2020-04-27",
                         "8.0.0": "2021-05-11", "8.1.8": "2024-12-21"}})
    project("flask==1.1.1\nclick==7.0\n", source="import flask, click\n")
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear", "greedy"],
                       config=config, max_passes=3)
    agent.index = index
    agent.rc.regroup = True          # off by default; this is the ablation
    result = agent.run()

    assert result.updated["click"][1] == "8.1.8", "the trade is the point"
    assert result.updated["flask"][1] == "1.1.2", "stepped back, not forward"
    assert result.counters.get("regroups_accepted") == 1


def test_a_trade_is_refused_when_it_gives_up_a_version_line(monkeypatch, project,
                                                            config, tmp_path):
    """Counting releases makes fifteen given up look cheaper than sixteen
    gained. It is not: a package dragged back across a major version loses a
    line of development, and the releases of two projects are not the same
    unit -- sixteen of one can span a year and fifteen of another five."""
    from tests.conftest import FakeIndex, history

    releases = {"lib": ["1.0", "2.0", "2.1"], "app": ["1.0", "1.1"]}

    def accepts(pins):
        # app 1.1 needs the old lib: the only way to advance app is to drag
        # lib back across its major version.
        if pins.get("app") == "1.1" and pins.get("lib", "").startswith("2"):
            return ("ERROR: Cannot install because these package versions have "
                    "conflicting dependencies.\n\nThe conflict is caused by:\n"
                    "    The user requested app==1.1\n"
                    "    app 1.1 depends on lib<2\n")
        return True

    world = FakeWorld(releases=releases, accepts=accepts)
    index = FakeIndex(
        {n: history(n, v) for n, v in releases.items()},
        requires={("app", "1.1"): ["lib (<2)"], ("app", "1.0"): ["lib"]},
        dates={"lib": {"1.0": "2018-01-01", "2.0": "2021-01-01",
                       "2.1": "2024-01-01"},
               "app": {"1.0": "2020-01-01", "1.1": "2020-06-01"}})
    project("app==1.0\nlib==1.0\n", source="import app, lib\n")
    agent = make_agent(monkeypatch, tmp_path, world, tiers=["linear", "greedy"],
                       config=config, max_passes=2)
    agent.index = index
    result = agent.run()

    # lib reached 2.1; nothing drags it back to 1.x to gain one app release.
    assert result.updated["lib"][1] == "2.1"
    assert not result.counters.get("regroups_accepted")


def test_each_probe_runs_its_tests_in_a_fresh_copy_of_the_project(tmp_path):
    """hk-flask-restful's tests create migrations/ and example.db in the
    working directory; parallel probes sharing one checkout made a test that
    passes at baseline fail in 307 of 405 probes."""
    from aura.agent import isolated_project
    from aura.validation import Validator
    project = tmp_path / "proj"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_x.py").write_text("def test_x(): pass\n")
    (project / ".git").mkdir()
    (project / ".aura_probe_0").mkdir()
    (project / "leftover.db").write_text("from an earlier probe\n")
    config = {"VALIDATION_CONFIG": {"type": "pytest", "pytest_target": "tests",
                                    "project_dir": str(project)}}
    validator = Validator(config)
    dest = tmp_path / "slot_src"
    cfg, val = isolated_project(config, validator, dest)
    assert val is not validator and val.cwd == str(dest)
    assert cfg["VALIDATION_CONFIG"]["project_dir"] == str(dest)
    assert config["VALIDATION_CONFIG"]["project_dir"] == str(project), "the shared config is untouched"
    assert (dest / "tests" / "test_x.py").exists()
    assert not (dest / ".git").exists() and not (dest / ".aura_probe_0").exists()
    (dest / "written_by_the_suite").write_text("x")
    isolated_project(config, validator, dest)
    assert not (dest / "written_by_the_suite").exists(), "each probe starts clean"
