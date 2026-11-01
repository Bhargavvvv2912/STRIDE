"""Tests for oracle sizing.

The case that motivates the whole module, measured for real against click 8.1.7:
a validation script covering 90% of the repo exercised 32% of click. Reporting
those two numbers as one "tests passed" is what Reviewer B objected to.
"""

import json

import pytest

from aura.coverage import (
    CoverageReport, PackageCoverage, build_report, compare, measure_validation,
)


def payload(files: dict) -> dict:
    return {"files": {name: {"summary": {"num_statements": stmts,
                                         "covered_lines": covered}}
                      for name, (stmts, covered) in files.items()}}


@pytest.fixture
def tree(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    site = tmp_path / "site"
    (site / "click").mkdir(parents=True)
    (site / "fastnum").mkdir(parents=True)
    (site / "fastnum" / "_core.so").write_bytes(b"\x00")
    return repo, site


def test_repo_and_dependency_are_measured_separately(tree):
    repo, site = tree
    report = build_report(
        payload({str(repo / "pkg" / "a.py"): (10, 9),
                 str(site / "click" / "core.py"): (4000, 1280),
                 str(site / "click" / "types.py"): (136, 44)}),
        repo_root=repo, dependency_dirs={"click": [site / "click"]})

    assert report.repo.statements == 10 and report.repo.pct == 0.9
    click = report.dependencies["click"]
    assert click.statements == 4136 and click.files == 2
    assert click.pct == round(1324 / 4136, 4)


def test_high_repo_coverage_does_not_imply_the_dependency_was_exercised(tree):
    """The finding the metric exists to expose."""
    repo, site = tree
    report = build_report(
        payload({str(repo / "pkg" / "a.py"): (100, 95),
                 str(site / "click" / "core.py"): (1000, 30)}),
        repo_root=repo, dependency_dirs={"click": [site / "click"]})
    assert report.repo.pct == 0.95
    assert report.dependencies["click"].pct == 0.03


def test_compiled_dependencies_are_flagged_so_the_number_is_read_correctly(tree):
    """coverage.py sees Python lines only; a C-heavy package understates."""
    repo, site = tree
    report = build_report(
        payload({str(site / "fastnum" / "__init__.py"): (20, 4)}),
        repo_root=repo, dependency_dirs={"fastnum": [site / "fastnum"]})
    assert report.dependencies["fastnum"].compiled is True


def test_files_outside_the_repo_and_tracked_dependencies_are_ignored(tree):
    repo, site = tree
    report = build_report(
        payload({"/usr/lib/python3.11/json/decoder.py": (200, 100),
                 str(repo / "pkg" / "a.py"): (10, 10)}),
        repo_root=repo, dependency_dirs={})
    assert report.repo.files == 1 and report.repo.statements == 10


def test_pct_is_none_rather_than_zero_when_nothing_was_measured():
    assert PackageCoverage(name="x").pct is None


# ------------------------------------------------------------------ deltas

def _report(repo_pct, deps):
    r = CoverageReport(ok=True, repo=PackageCoverage("repo", 100,
                                                     int(repo_pct * 100), files=1))
    r.dependencies = {n: PackageCoverage(n, 100, int(p * 100), files=1)
                      for n, p in deps.items()}
    return r


def test_a_coverage_drop_is_surfaced():
    """Less code running after an upgrade: tests pass, metrics match, silent."""
    delta = compare(_report(0.80, {"numpy": 0.50}), _report(0.80, {"numpy": 0.20}))
    assert [d["unit"] for d in delta["dropped"]] == ["numpy"]
    assert delta["dependency_deltas"]["numpy"] == -0.30


def test_a_coverage_rise_is_not_treated_as_a_problem():
    delta = compare(_report(0.80, {"numpy": 0.20}), _report(0.80, {"numpy": 0.50}))
    assert delta["dropped"] == []


def test_small_movements_stay_within_tolerance():
    delta = compare(_report(0.80, {"numpy": 0.50}), _report(0.80, {"numpy": 0.49}))
    assert delta["dropped"] == []


# ------------------------------------------------------------- install-only

class _NoCodeValidator:
    cwd = "."
    timeout_s = 60

    def command(self, python):
        return None


def test_install_only_reports_that_nothing_was_executed():
    """The regime where coverage is not low but undefined -- and saying so matters."""
    class _Env:
        python = "/nonexistent/python"

    report = measure_validation(_Env(), _NoCodeValidator(), repo_root=None,
                                dependencies=[], workdir=None)
    assert report.ok is False and "runs no code" in report.reason
