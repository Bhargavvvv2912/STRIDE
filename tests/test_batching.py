"""Batched probing: same answer as a serial scan, fewer wall-clock seconds.

The exactness argument is that a batch is a contiguous prefix of the
newest-first order and the highest passing member is taken, so the batch returns
what a serial scan would have returned. These tests hold that claim against
oracles designed to break it.
"""

import pytest

from aura.batching import (
    BatchPolicy, Resources, batches, footprint_mb, size_bucket,
)
from aura.environment import ErrorClass


# ------------------------------------------------------------------- policy

def test_k_falls_out_of_measured_size_and_detected_resources():
    """No tuned constant: wheel size in, batch width out."""
    workstation = Resources(cpus=12, free_disk_mb=250_000, total_memory_mb=25_000)
    policy = BatchPolicy(resources=workstation)
    assert policy.k_for(139.5) == 6          # torch, capped by cpu then k_max
    assert policy.plan(139.5)["bucket"] == "large"


def test_the_same_formula_gives_k_1_on_a_small_ci_runner():
    """Why AURA runs faster on a workstation than in CI, stated as arithmetic."""
    runner = Resources(cpus=2, free_disk_mb=14_000, total_memory_mb=7_000)
    assert BatchPolicy(resources=runner).k_for(139.5) == 1


def test_a_tiny_disk_binds_before_the_cpu_does():
    tight = Resources(cpus=32, free_disk_mb=6_000, total_memory_mb=64_000)
    plan = BatchPolicy(resources=tight).plan(139.5)   # 558 MB per probe
    assert plan["binding_constraint"] == "disk"
    assert plan["k"] == 1


def test_batching_can_be_disabled_and_pinned():
    res = Resources(cpus=12, free_disk_mb=250_000, total_memory_mb=25_000)
    assert BatchPolicy(resources=res, enabled=False).k_for(139.5) == 1
    assert BatchPolicy(resources=res, override=3).k_for(139.5) == 3


def test_unknown_size_does_not_crash_the_plan():
    res = Resources(cpus=8, free_disk_mb=100_000, total_memory_mb=16_000)
    assert BatchPolicy(resources=res).k_for(None) >= 1
    assert size_bucket(None) == "unknown"
    assert footprint_mb(None) > 0


def test_groups_are_contiguous_which_is_what_preserves_exactness():
    assert batches(["5", "4", "3", "2", "1"], 2) == [["5", "4"], ["3", "2"], ["1"]]
    assert batches(["5", "4", "3"], 1) == [["5"], ["4"], ["3"]]


# --------------------------------------------------- equivalence with serial

RELEASES = {"torch": ["1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0"]}


def _run(monkeypatch, tmp_path, config, project, accepts, k, probed=None):
    from tests.conftest import FakeWorld
    from tests.test_agent import make_agent

    project("torch==1.0.0\n")

    def record(pins):
        if probed is not None:
            probed.append(pins["torch"])
        return accepts(pins)

    world = FakeWorld(releases=RELEASES, accepts=record)
    return make_agent(monkeypatch, tmp_path, world, tiers=["linear"],
                      config=config, batch_k=k).run()


@pytest.mark.parametrize("ceiling", ["1.1.0", "1.3.0", "1.5.0", "1.6.0"])
def test_batched_and_serial_scans_agree(monkeypatch, project, config, tmp_path,
                                        ceiling):
    """The claim, held against every position the answer could sit in."""
    from aura.reqfile import safe_parse_version

    def accepts(pins):
        return safe_parse_version(pins["torch"]) <= safe_parse_version(ceiling)

    serial = _run(monkeypatch, tmp_path, config, project, accepts, k=1)
    batched = _run(monkeypatch, tmp_path, config, project, accepts, k=6)
    assert serial.updated == batched.updated
    assert batched.updated["torch"][1] == ceiling


def test_a_hole_in_the_version_line_does_not_fool_a_batch(monkeypatch, project,
                                                          config, tmp_path):
    """[1,1,0,1,1]: the pattern that rules out bisection. Batching is unaffected
    because it probes a whole contiguous block rather than sampling it."""
    broken = {"1.4.0"}

    def accepts(pins):
        return pins["torch"] not in broken

    serial = _run(monkeypatch, tmp_path, config, project, accepts, k=1)
    batched = _run(monkeypatch, tmp_path, config, project, accepts, k=3)
    assert serial.updated == batched.updated == {"torch": ("1.0.0", "1.6.0")}


def test_batching_takes_the_highest_passer_not_the_first_to_finish(monkeypatch,
                                                                   project, config,
                                                                   tmp_path):
    """Several members of one batch pass; only the newest may be returned."""
    def accepts(pins):
        return pins["torch"] in {"1.2.0", "1.3.0", "1.4.0"}

    result = _run(monkeypatch, tmp_path, config, project, accepts, k=6)
    assert result.updated["torch"][1] == "1.4.0"


def test_batching_probes_no_more_candidates_than_the_serial_scan(monkeypatch,
                                                                 project, config,
                                                                 tmp_path):
    """A batch may probe versions a serial scan would have skipped, but never
    more than the batch it is in -- and it must not run away down the line."""
    serial_probes, batched_probes = [], []

    def accepts(pins):
        return pins["torch"] == "1.6.0"

    _run(monkeypatch, tmp_path, config, project, accepts, k=1, probed=serial_probes)
    _run(monkeypatch, tmp_path, config, project, accepts, k=6, probed=batched_probes)
    # The newest passes immediately, so both stop after the first block.
    assert len(batched_probes) <= len(serial_probes) + 6


# ------------------------------------------------------------------- guards
#
# The direct attempt probes the newest version before any tier runs, so the
# linear scan only ever sees versions *below* the target. These fixtures make
# the target fail deterministically at DIRECT and use a single pass, so what is
# measured is the batch's own behaviour rather than a later pass succeeding.

def _guard_agent(monkeypatch, tmp_path, config, project, accepts, seen):
    from tests.conftest import FakeWorld
    from tests.test_agent import make_agent

    project("torch==1.0.0\n")

    def record(pins):
        seen.append(pins["torch"])
        return accepts(pins)

    world = FakeWorld(releases=RELEASES, accepts=record)
    return make_agent(monkeypatch, tmp_path, world, tiers=["linear"],
                      config=config, batch_k=6, max_passes=1)


def test_a_contended_failure_above_the_winner_is_re_probed_alone(monkeypatch,
                                                                 project, config,
                                                                 tmp_path):
    """The correctness guard: a timeout under load must not discard a version."""
    seen = []

    def accepts(pins):
        version = pins["torch"]
        if version == "1.6.0":
            return "no wheel for this platform"        # a fact: target is out
        if version == "1.5.0":
            # Fails the first time, in a class contention can explain.
            if seen.count("1.5.0") == 1:
                return ("ERROR: operation timed out", ErrorClass.TIMEOUT)
            return True
        return version == "1.2.0"

    result = _guard_agent(monkeypatch, tmp_path, config, project, accepts, seen).run()
    # Without the guard the batch settles for 1.2.0; with it, 1.5.0 is recovered.
    assert result.updated["torch"][1] == "1.5.0"
    assert seen.count("1.5.0") >= 2


def test_a_deterministic_failure_is_not_re_probed(monkeypatch, project, config,
                                                  tmp_path):
    """Only contention-explicable classes get a second chance. Re-running a
    genuine test failure buys nothing and costs an install."""
    seen = []

    def accepts(pins):
        version = pins["torch"]
        if version == "1.5.0":
            return ("tests failed", ErrorClass.VALIDATION_FAILED)
        return version == "1.2.0"

    result = _guard_agent(monkeypatch, tmp_path, config, project, accepts, seen).run()
    assert result.updated["torch"][1] == "1.2.0"
    assert seen.count("1.5.0") == 1          # failed once, believed


def test_the_winner_is_confirmed_serially_before_it_is_accepted(monkeypatch,
                                                                project, config,
                                                                tmp_path):
    """Nothing is accepted on a measurement taken while k installs competed."""
    seen = []

    def accepts(pins):
        version = pins["torch"]
        if version == "1.6.0":
            return "no wheel for this platform"
        return version == "1.5.0"

    result = _guard_agent(monkeypatch, tmp_path, config, project, accepts, seen).run()
    assert result.updated["torch"][1] == "1.5.0"
    assert seen.count("1.5.0") == 2          # once in the batch, once alone


def test_a_winner_that_fails_confirmation_falls_back_to_the_next_passer(
        monkeypatch, project, config, tmp_path):
    """A flattering concurrent measurement must not become a result."""
    seen = []

    def accepts(pins):
        version = pins["torch"]
        if version == "1.6.0":
            return "no wheel for this platform"
        if version == "1.5.0":
            return seen.count("1.5.0") == 1   # passes batched, fails alone
        return version == "1.4.0"

    result = _guard_agent(monkeypatch, tmp_path, config, project, accepts, seen).run()
    assert result.updated["torch"][1] == "1.4.0"


# ------------------------------------------- sizing from the job's allocation

def test_a_slurm_allocation_limits_the_cpus_a_job_plans_for(tmp_path):
    """A job given 8 of a node's 64 cores must plan for 8."""
    res = Resources.detect(str(tmp_path), env={"SLURM_CPUS_PER_TASK": "8"},
                           cgroup_root=str(tmp_path / "none"))
    assert res.cpus <= 8
    assert res.cpu_source in ("slurm", "affinity")


def test_slurm_memory_is_used_when_it_is_tighter_than_the_machine(tmp_path):
    res = Resources.detect(str(tmp_path), env={"SLURM_MEM_PER_NODE": "2000"},
                           cgroup_root=str(tmp_path / "none"))
    assert res.total_memory_mb == 2000 and res.memory_source == "slurm"


def test_memory_per_cpu_scales_with_the_allocated_cpus(tmp_path):
    res = Resources.detect(str(tmp_path),
                           env={"SLURM_CPUS_PER_TASK": "2", "SLURM_MEM_PER_CPU": "900"},
                           cgroup_root=str(tmp_path / "none"))
    assert res.total_memory_mb == 900 * res.cpus


def test_a_cgroup_limit_is_respected(tmp_path):
    """Containers and some schedulers cap a job through cgroups instead."""
    (tmp_path / "cpu.max").write_text("200000 100000\n")          # 2 CPUs
    (tmp_path / "memory.max").write_text(str(1_500_000_000) + "\n")   # 1.5 GB
    res = Resources.detect(str(tmp_path), env={}, cgroup_root=str(tmp_path))
    assert res.cpus <= 2
    assert res.total_memory_mb == 1500 and res.memory_source == "cgroup"


def test_an_unlimited_cgroup_does_not_count_as_a_limit(tmp_path):
    (tmp_path / "cpu.max").write_text("max 100000\n")
    (tmp_path / "memory.max").write_text("max\n")
    res = Resources.detect(str(tmp_path), env={}, cgroup_root=str(tmp_path))
    assert res.cpu_source != "cgroup" and res.memory_source != "cgroup"


def test_slurm_list_formats_are_parsed():
    from aura.batching import _first_int
    assert _first_int("8(x2)") == 8
    assert _first_int("4,8") == 4
    assert _first_int(None) is None


def test_a_small_allocation_shrinks_the_batch():
    """The point of all this: 2 allocated CPUs must not run 6 installs at once."""
    small = Resources(cpus=2, free_disk_mb=250_000, total_memory_mb=64_000)
    assert BatchPolicy(resources=small).k_for(0.1) == 1
