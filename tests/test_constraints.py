"""Tests for the declared-constraint graph.

The fixture mirrors the real torch/torchvision relationship, which is the case
the whole feature exists for: torchvision pins torch exactly, so every torch
above the current pin is forbidden until torchvision moves too.
"""

import pytest

from aura.constraints import (
    Constraint, ConstraintGraph, build_graph, close_group, co_update_plan,
    marker_environment,
    parse_requires_dist, unlocking_versions,
)
from aura.index import ReleaseHistory

# Real pins, taken from PyPI metadata.
TORCHVISION_PINS = {
    "0.13.0": "1.12.0",
    "0.14.0": "1.13.0",
    "0.15.1": "2.0.0",
    "0.15.2": "2.0.1",
}


class FakeIndex:
    """Just enough index for the graph: declared deps and release histories."""

    def __init__(self, requires: dict, histories: dict, unknown=()):
        self.requires, self.histories = requires, histories
        self.unknown = set(unknown)
        self.calls = 0

    def requires_dist(self, name, version):
        self.calls += 1
        if name in self.unknown:
            return None
        return self.requires.get((name, version))

    def history(self, name):
        return self.histories.get(name)


@pytest.fixture
def index():
    requires = {("torchvision", v): [f"torch (=={t})", "numpy", "requests"]
                for v, t in TORCHVISION_PINS.items()}
    requires[("torch", "1.12.0")] = ["typing-extensions"]
    histories = {
        "torchvision": ReleaseHistory("torchvision", tuple(TORCHVISION_PINS)),
        "torch": ReleaseHistory("torch", ("1.12.0", "1.12.1", "1.13.0",
                                          "2.0.0", "2.0.1")),
    }
    return FakeIndex(requires, histories)


# ------------------------------------------------------------------- parsing

def test_extra_gated_requirements_do_not_constrain_the_base_install():
    """`scipy ; extra == 'scipy'` binds nothing unless that extra is requested."""
    out = parse_requires_dist("torchvision", "0.13.0",
                              ["torch (==1.12.0)", "scipy ; extra == 'scipy'"])
    assert [c.target for c in out] == ["torch"]


def test_a_requested_extra_does_constrain():
    out = parse_requires_dist("torchvision", "0.13.0",
                              ["scipy (>=1.5) ; extra == 'scipy'"],
                              active_extras=frozenset({"scipy"}))
    assert [c.target for c in out] == ["scipy"]


def test_markers_are_evaluated_against_the_target_interpreter():
    raw = ["importlib-metadata ; python_version < '3.10'"]
    on_39 = parse_requires_dist("x", "1.0", raw, env=marker_environment("3.9"))
    on_311 = parse_requires_dist("x", "1.0", raw, env=marker_environment("3.11"))
    assert [c.target for c in on_39] == ["importlib-metadata"]
    assert on_311 == []


def test_unparseable_requirements_are_dropped_not_guessed():
    out = parse_requires_dist("x", "1.0", ["this is not a requirement!!", "numpy"])
    assert [c.target for c in out] == ["numpy"]


# --------------------------------------------------------------------- graph

def test_graph_reads_the_pin_that_v1_had_to_fail_to_discover(index):
    graph = build_graph({"torchvision": "0.13.0", "torch": "1.12.0"}, index)
    edges = graph.constraining("torch")
    assert [c.describe() for c in edges] == [
        "torchvision==0.13.0 requires torch==1.12.0"]


def test_exact_pin_makes_the_whole_scan_range_infeasible(index):
    """The routing case: every candidate above the current pin is forbidden."""
    graph = build_graph({"torchvision": "0.13.0", "torch": "1.12.0"}, index)
    keep, blocked = graph.feasible("torch", ["1.12.1", "1.13.0", "2.0.0"])
    assert keep == []
    assert len(blocked) == 3
    assert "torchvision==0.13.0" in blocked[0]["blockers"][0]


def test_a_ceiling_leaves_part_of_the_range_feasible(index):
    """`torch<1.14` must not be routed like `torch==1.12.0`; linear still works."""
    index.requires["torchvision", "0.13.0"] = ["torch (<1.14)"]
    graph = build_graph({"torchvision": "0.13.0"}, index)
    keep, _ = graph.feasible("torch", ["1.12.1", "1.13.0", "2.0.0"])
    assert keep == ["1.12.1", "1.13.0"]


def test_packages_free_to_move_do_not_block(index):
    """Inside a co-resolution group the blocker's own pin is not binding."""
    graph = build_graph({"torchvision": "0.13.0"}, index)
    keep, _ = graph.feasible("torch", ["2.0.0"], exclude={"torchvision"})
    assert keep == ["2.0.0"]


def test_unreadable_metadata_means_no_information_not_no_constraints(index):
    """Absent metadata must never license a futility claim."""
    index.unknown.add("torchvision")
    graph = build_graph({"torchvision": "0.13.0"}, index)
    assert graph.constraining("torch") == []
    assert "torchvision" in graph.unknown
    keep, _ = graph.feasible("torch", ["2.0.0"])
    assert keep == ["2.0.0"]          # not blocked: we simply do not know


# ----------------------------------------------------------------- inversion

def test_inversion_finds_the_unlocking_version_below_the_newest(index):
    """The answer is not at the top: newer torchvision pins newer torch."""
    search = unlocking_versions("torchvision", list(TORCHVISION_PINS),
                                "torch", "2.0.0", index)
    assert search.versions == ["0.15.1"]
    assert search.complete


def test_an_exhausted_budget_is_reported_not_mistaken_for_no_solution(index):
    """A cap reached before the answer must not read as 'unsolvable'."""
    search = unlocking_versions("torchvision", list(TORCHVISION_PINS),
                                "torch", "2.0.0", index, budget=1)
    assert search.versions == [] and search.complete is False


def test_co_update_plan_solves_the_pair(index):
    graph = build_graph({"torchvision": "0.13.0", "torch": "1.12.0"}, index)
    plan, reason = co_update_plan("torch", "2.0.0", graph,
                                  {"torchvision": "0.13.0"}, index)
    assert plan == {"torch": "2.0.0", "torchvision": "0.15.1"}
    assert reason == "solved"


def test_co_update_plan_separates_proof_from_giving_up(index):
    """'no version unlocks this' is evidence; 'I stopped looking' is not."""
    graph = build_graph({"torchvision": "0.13.0", "torch": "1.12.0"}, index)
    pins = {"torchvision": "0.13.0"}

    plan, reason = co_update_plan("torch", "9.9.9", graph, pins, index)
    assert plan is None and reason == "no_unlocking_version:torchvision"

    plan, reason = co_update_plan("torch", "2.0.0", graph, pins, index, budget=1)
    assert plan is None and reason == "search_incomplete:torchvision"


def test_unconstrained_target_needs_no_group(index):
    graph = build_graph({"torch": "1.12.0"}, index)
    plan, reason = co_update_plan("numpy", "1.26.4", graph, {}, index)
    assert plan == {"numpy": "1.26.4"} and reason == "no_blockers"


# -------------------------------------------------------------- group closure

@pytest.fixture
def flask_index():
    requires = {
        ("flask", "3.1.3"): ["click (>=8.1.3)", "jinja2 (>=3.1.2)",
                             "werkzeug (>=3.1)", "itsdangerous (>=2.2)",
                             "importlib-metadata (>=3.6) ; python_version < '3.10'",
                             "blinker (>=1.9)"],
        ("importlib-metadata", "8.7.1"): ["zipp (>=3.20)"],
        ("jinja2", "3.1.6"): ["markupsafe (>=2.0)"],
    }
    histories = {
        "click": ReleaseHistory("click", ("7.0", "8.1.8", "8.5.0"), metadata={
            "8.5.0": {"requires_python": ">=3.10", "yanked": False}}),
        "jinja2": ReleaseHistory("jinja2", ("2.10.3", "3.1.6")),
        "werkzeug": ReleaseHistory("werkzeug", ("0.16.0", "3.1.8")),
        "itsdangerous": ReleaseHistory("itsdangerous", ("1.1.0", "2.2.0")),
        "importlib-metadata": ReleaseHistory("importlib-metadata", ("1.1.0", "8.7.1")),
        "zipp": ReleaseHistory("zipp", ("0.6.0", "3.23.1")),
        "markupsafe": ReleaseHistory("markupsafe", ("1.1.1", "3.0.3")),
    }
    return FakeIndex(requires, histories)


PINS = {"flask": "1.1.4", "click": "7.0", "jinja2": "2.10.3", "werkzeug": "0.16.0",
        "itsdangerous": "1.1.0", "importlib-metadata": "1.1.0", "zipp": "0.6.0",
        "markupsafe": "1.1.1"}


def test_closure_collects_every_declared_requirement_in_one_step(flask_index):
    """pip names one conflict per failed install; the metadata names them all."""
    group, notes = close_group({"itsdangerous": "2.2.0", "flask": "3.1.3"}, PINS,
                               flask_index, python_version="3.9")
    assert group == {"itsdangerous": "2.2.0", "flask": "3.1.3", "click": "8.1.8",
                     "jinja2": "3.1.6", "werkzeug": "3.1.8",
                     "importlib-metadata": "8.7.1", "zipp": "3.23.1",
                     "markupsafe": "3.0.3"}
    reasons = {n["package"]: n["required_by"] for n in notes}
    assert reasons["zipp"] == "importlib-metadata==8.7.1"      # second-level


def test_closure_respects_the_interpreter(flask_index):
    """click 8.5 needs 3.10: on 3.9 the closure must stop at 8.1.8, and a
    marker-gated requirement must only apply where its marker holds."""
    on_39, _ = close_group({"flask": "3.1.3"}, PINS, flask_index, python_version="3.9")
    on_311, _ = close_group({"flask": "3.1.3"}, PINS, flask_index, python_version="3.11")
    assert on_39["click"] == "8.1.8"
    assert on_311["click"] == "8.5.0"
    assert "importlib-metadata" in on_39
    assert "importlib-metadata" not in on_311


def test_closure_leaves_unpinned_dependencies_to_pip(flask_index):
    group, _ = close_group({"flask": "3.1.3"}, PINS, flask_index, python_version="3.9")
    assert "blinker" not in group


def test_closure_never_moves_a_package_down(flask_index):
    """A requirement that only an older version meets is reported, not applied."""
    flask_index.requires[("flask", "3.1.3")] = ["click (<7)"]
    group, notes = close_group({"flask": "3.1.3"}, PINS, flask_index)
    assert "click" not in group
    assert notes and notes[0]["resolved"] is None
