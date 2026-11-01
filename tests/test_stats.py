"""Tests for the paired-comparison statistics."""

import pytest

from evaluation.stats import (
    bootstrap_ci, describe, mcnemar_exact, paired_outcomes, wilson_interval,
)


def test_wilson_handles_a_perfect_score_without_a_degenerate_interval():
    """14/14 must not produce a zero-width interval, which a normal CI would."""
    p = wilson_interval(14, 14)
    assert p.value == 1.0
    assert p.low < 1.0 and p.high == pytest.approx(1.0)
    assert p.low > 0.75


def test_wilson_handles_zero_successes():
    p = wilson_interval(0, 32)
    assert p.value == 0.0 and p.low == 0.0 and 0 < p.high < 0.2


def test_wilson_on_an_empty_sample_is_not_an_error():
    assert wilson_interval(0, 0).value == 0.0


def test_mcnemar_ignores_concordant_pairs():
    """Only subjects where the arms disagree carry information."""
    assert mcnemar_exact(0, 0).p_value == 1.0
    assert mcnemar_exact(5, 5).p_value == 1.0


def test_mcnemar_detects_a_one_sided_difference():
    result = mcnemar_exact(8, 0)
    assert result.n_discordant == 8
    assert result.p_value == pytest.approx(2 * 0.5 ** 8)
    assert result.p_value < 0.01


def test_mcnemar_is_symmetric_in_its_p_value():
    assert mcnemar_exact(7, 2).p_value == mcnemar_exact(2, 7).p_value


def test_mcnemar_small_discordant_counts_are_not_significant():
    """A 3-vs-0 split cannot reach significance -- worth knowing before claiming it."""
    assert mcnemar_exact(3, 0).p_value == pytest.approx(0.25)


def test_paired_outcomes_uses_only_shared_subjects():
    a = {"r1": True, "r2": True, "r3": False, "only_in_a": True}
    b = {"r1": False, "r2": True, "r3": False}
    result = paired_outcomes(a, b)
    assert result.b == 1 and result.c == 0 and result.n_discordant == 1


def test_bootstrap_ci_brackets_the_point_estimate():
    values = [10.0, 12.0, 11.0, 30.0, 9.0, 14.0, 13.0, 11.5]
    point, low, high = bootstrap_ci(values, iterations=2000, seed=1)
    assert low <= point <= high
    assert point == pytest.approx(sum(values) / len(values))


def test_bootstrap_is_reproducible_for_a_seed():
    values = [1.0, 5.0, 2.0, 8.0]
    assert bootstrap_ci(values, iterations=500, seed=3) == \
           bootstrap_ci(values, iterations=500, seed=3)


def test_describe_summarizes_runtimes():
    d = describe([60.0, 120.0, 300.0])
    assert d["n"] == 3 and d["median"] == 120.0 and d["max"] == 300.0


def test_describe_of_nothing_is_zeroed_not_an_error():
    assert describe([])["n"] == 0
