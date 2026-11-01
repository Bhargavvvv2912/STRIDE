"""Tests for oracle-strength measurement."""

import pytest

from aura.telemetry import Outcome, RunConfig, RunLog, Tier
from evaluation.oracle import (
    HARD_FLOOR, Mutation, MutationResult, OracleReport, observed_discrimination,
    probe_oracle, summarize_reports,
)
from evaluation.report import Analysis, load_runs


def report(subject, **detected):
    return OracleReport(subject=subject, mutations=[
        MutationResult(subject=subject, mutation=name, detected=value)
        for name, value in detected.items()])


def test_hard_floor_is_only_the_guaranteed_break():
    """Only a mutation that is broken independently of the oracle can be a floor."""
    assert HARD_FLOOR == {Mutation.MISSING}


def test_an_oracle_that_misses_a_missing_dependency_fails_the_floor():
    r = report("weak", missing_dependency=False, stale_dependency=True)
    assert r.passes_floor is False


def test_an_oracle_that_catches_a_missing_dependency_passes_the_floor():
    r = report("ok", missing_dependency=True, stale_dependency=False)
    assert r.passes_floor is True


def test_a_report_with_no_mutations_does_not_pass_by_default():
    assert OracleReport(subject="none").passes_floor is False


def test_detection_rate_is_reported():
    r = report("x", missing_dependency=True, stale_dependency=True, major_jump=False)
    assert r.detection_rate == pytest.approx(2 / 3)


def test_summary_names_the_subjects_whose_results_must_be_discarded():
    summary = summarize_reports([
        report("good", missing_dependency=True, stale_dependency=True),
        report("weak", missing_dependency=False, stale_dependency=False),
    ])
    assert summary["failed_hard_floor"] == ["weak"]
    assert "100.0%" not in summary["detection_rate_by_mutation"]["missing_dependency"]
    assert summary["subjects"] == 2


# ------------------------------------------- observed discrimination from logs

def _run(root, subject, *, installs_rejected_by_oracle, install_failures):
    log = RunLog(root / subject / "aura_full_qwen14b.jsonl",
                 RunConfig(repo=subject, arm="aura_full_qwen14b"))
    for _ in range(installs_rejected_by_oracle):
        log.attempt(package="torch", tier=Tier.LINEAR, plan={}, install_ok=True,
                    validate_ok=False, duration_s=1.0)
    for _ in range(install_failures):
        log.attempt(package="torch", tier=Tier.LINEAR, plan={}, install_ok=False,
                    validate_ok=False, duration_s=1.0)
    log.attempt(package="torch", tier=Tier.LINEAR, plan={}, install_ok=True,
                validate_ok=True, duration_s=1.0, accepted=True)
    log.finish(Outcome.MODERNIZED, updated={"torch": ("1.0", "2.0")})


def test_a_silent_oracle_is_identified(tmp_path):
    """If every rejection came from the resolver, the oracle contributed nothing."""
    root = tmp_path / "results"
    _run(root, "discriminating", installs_rejected_by_oracle=4, install_failures=1)
    _run(root, "silent", installs_rejected_by_oracle=0, install_failures=6)

    result = observed_discrimination(Analysis(load_runs(root)))
    assert result["silent_oracles"] == ["silent"]
    assert result["per_subject"]["discriminating"]["rejected_by_oracle"] == 4
    assert result["per_subject"]["silent"]["rejected_by_oracle"] == 0
    assert "50.0%" in result["oracle_fired_at_least_once"]


def test_discrimination_is_free_of_extra_runs(tmp_path):
    """It reads only what the search already logged."""
    root = tmp_path / "results"
    _run(root, "r1", installs_rejected_by_oracle=2, install_failures=0)
    result = observed_discrimination(Analysis(load_runs(root)))
    assert result["per_subject"]["r1"]["reached_the_oracle"] == 3   # 2 rejects + 1 accept


# --------------------------------------------------- real end-to-end mutation

@pytest.mark.slow
def test_probe_oracle_catches_a_missing_dependency(tmp_path):
    """A real environment, really damaged, with a real oracle."""
    from evaluation.subjects import Subject

    (tmp_path / "requirements.txt").write_text("packaging==25.0\n")
    (tmp_path / "main.py").write_text("import packaging\n")
    (tmp_path / "validate.py").write_text(
        "import packaging\nfrom packaging.version import parse\n"
        "assert parse('2.0') > parse('1.0')\n")

    subject = Subject(
        name="oracle-demo", commit="a" * 40, screened_date="2026-09-07",
        validation={"type": "script", "smoke_test_script": str(tmp_path / "validate.py")},
        validation_provenance="repo_tests")

    result = probe_oracle(subject, tmp_path, mutations=(Mutation.MISSING,))
    assert result.error == ""
    assert len(result.mutations) == 1
    assert result.mutations[0].detected is True, "oracle failed to notice a missing import"
    assert result.passes_floor


@pytest.mark.slow
def test_probe_oracle_exposes_a_vacuous_oracle(tmp_path):
    """A script that asserts nothing about the project must fail the floor."""
    from evaluation.subjects import Subject

    (tmp_path / "requirements.txt").write_text("packaging==25.0\n")
    (tmp_path / "main.py").write_text("import packaging\n")
    (tmp_path / "validate.py").write_text("print('looks fine to me')\n")

    subject = Subject(
        name="vacuous-demo", commit="a" * 40, screened_date="2026-09-07",
        validation={"type": "script", "smoke_test_script": str(tmp_path / "validate.py")},
        validation_provenance="authored", validation_reviewed_by="me")

    result = probe_oracle(subject, tmp_path, mutations=(Mutation.MISSING,))
    assert result.mutations[0].detected is False
    assert result.passes_floor is False
