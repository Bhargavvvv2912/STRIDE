"""Tests for baseline-relative validation and behavioural drift detection."""

import sys
import pytest

from aura.validation import (
    ValidationResult, Validator, compare_metrics, extract_metrics, run_pytest, run_script,
)

PYTEST_OUTPUT_2_FAIL = """\
collected 10 tests

FAILED tests/test_a.py::test_one - AssertionError: nope
FAILED tests/test_b.py::test_two - ValueError
2 failed, 8 passed in 1.23s
"""


def _result(failing=(), metrics=None, passed=None, collected=10):
    failing = set(failing)
    return ValidationResult(
        passed=(not failing) if passed is None else passed,
        failing=failing, metrics=metrics or {}, collected=collected,
    )


def _validator(baseline, tolerance=0.01, require_metric_parity=True):
    v = Validator({"VALIDATION_CONFIG": {"type": "pytest", "pytest_target": "tests"}},
                  tolerance=tolerance, require_metric_parity=require_metric_parity)
    v.baseline = baseline
    return v


# --------------------------------------------------------- metric extraction

def test_explicit_metrics_line_wins():
    out = 'noise: 1.0\nAURA_METRICS: {"accuracy": 0.91, "loss": 0.13}\nmore: 2.0'
    assert extract_metrics(out) == {"accuracy": 0.91, "loss": 0.13}


def test_scrape_fallback_handles_common_shapes():
    out = "Final accuracy: 0.9123\nloss = 1.4e-3\nBLEU score:  27.4\nTop-1 Acc: 76 %"
    m = extract_metrics(out)
    assert m["final_accuracy"] == pytest.approx(0.9123)
    assert m["loss"] == pytest.approx(0.0014)
    assert m["bleu_score"] == pytest.approx(27.4)
    assert m["top_1_acc"] == pytest.approx(76.0)


def test_scrape_ignores_booleans_and_noise_keys():
    assert extract_metrics('AURA_METRICS: {"ok": true, "n": 3}') == {"n": 3.0}
    assert "version" not in extract_metrics("version: 3.11")


def test_metrics_from_a_repo_with_no_numbers_is_empty():
    assert extract_metrics("everything worked!\nDone.\n") == {}


# ------------------------------------------------------------- drift compare

def test_metric_drift_detected_beyond_tolerance():
    drifted, detail = compare_metrics({"acc": 0.91}, {"acc": 0.62}, tolerance=0.01)
    assert drifted == ["acc"]
    assert detail["acc"]["relative_delta"] == pytest.approx(0.3187, rel=1e-3)


def test_metric_within_tolerance_is_not_drift():
    assert compare_metrics({"acc": 0.91}, {"acc": 0.9105}, tolerance=0.01)[0] == []


def test_missing_key_is_not_treated_as_drift():
    assert compare_metrics({"acc": 0.9, "extra": 1.0}, {"acc": 0.9}, 0.01)[0] == []


def test_zero_baseline_does_not_divide_by_zero():
    drifted, _ = compare_metrics({"loss": 0.0}, {"loss": 0.5}, tolerance=0.01)
    assert drifted == ["loss"]


# ------------------------------------------------- baseline-relative judging

def test_preexisting_failure_does_not_block_a_candidate():
    """The soft-pass replacement: same failures as baseline is acceptable."""
    v = _validator(_result(failing={"tests/test_a.py::test_one"}))
    accepted, new, drifted, _ = v.judge(_result(failing={"tests/test_a.py::test_one"}))
    assert accepted and new == [] and drifted == []


def test_a_new_failure_is_rejected_even_at_the_same_count():
    """v1's absolute threshold accepted this; three different failures still 'passed'."""
    v = _validator(_result(failing={"tests/test_a.py::test_one"}))
    accepted, new, _, _ = v.judge(_result(failing={"tests/test_b.py::test_two"}))
    assert not accepted
    assert new == ["tests/test_b.py::test_two"]


def test_fixing_a_failure_is_still_accepted():
    v = _validator(_result(failing={"tests/test_a.py::test_one"}))
    assert v.judge(_result(failing=set()))[0] is True


def test_crash_with_no_node_id_is_rejected():
    v = _validator(_result(failing=set()))
    candidate = ValidationResult(passed=False, failing=set(), reason="ImportError", collected=10)
    accepted, new, _, _ = v.judge(candidate)
    assert not accepted and new == ["ImportError"]


def test_zero_collected_tests_is_rejected():
    v = _validator(_result(failing=set()))
    assert v.judge(_result(passed=True, collected=0))[0] is False


def test_metric_drift_rejects_a_test_passing_candidate():
    """Reviewer B's point: green tests do not imply preserved behaviour."""
    v = _validator(_result(metrics={"accuracy": 0.91}))
    accepted, new, drifted, detail = v.judge(_result(metrics={"accuracy": 0.62}))
    assert not accepted and new == [] and drifted == ["accuracy"]
    assert detail["accuracy"]["candidate"] == 0.62


def test_drift_is_reported_but_not_fatal_when_parity_not_required():
    v = _validator(_result(metrics={"accuracy": 0.91}), require_metric_parity=False)
    accepted, _, drifted, _ = v.judge(_result(metrics={"accuracy": 0.62}))
    assert accepted is True and drifted == ["accuracy"]


def test_without_a_baseline_it_falls_back_to_absolute_pass_fail():
    v = Validator({"VALIDATION_CONFIG": {"type": "pytest", "pytest_target": "t"}})
    assert v.judge(_result(failing=set()))[0] is True
    assert v.judge(_result(failing={"x"}))[0] is False


# ------------------------------------------------------------ runner parsing

def test_pytest_node_ids_are_recovered(tmp_path):
    from aura.validation import PYTEST_NODE_RE
    assert set(PYTEST_NODE_RE.findall(PYTEST_OUTPUT_2_FAIL)) == {
        "tests/test_a.py::test_one", "tests/test_b.py::test_two",
    }


def test_run_script_captures_metrics_and_exit_code(tmp_path):
    import sys
    script = tmp_path / "val.py"
    script.write_text('print(\'AURA_METRICS: {"accuracy": 0.5}\')\n')
    res = run_script(sys.executable, str(script), cwd=str(tmp_path))
    assert res.passed and res.metrics == {"accuracy": 0.5}


def test_run_script_reports_failure_exit_code(tmp_path):
    import sys
    script = tmp_path / "bad.py"
    script.write_text("import sys; sys.exit(3)\n")
    res = run_script(sys.executable, str(script), cwd=str(tmp_path))
    assert not res.passed and "exit 3" in res.reason


def test_install_only_mode_accepts_without_running_anything():
    """The UPCY stand-in regime: resolver satisfaction is the whole criterion."""
    v = Validator({"VALIDATION_CONFIG": {"type": "none"}})
    result = v.run("/nonexistent/python")     # never invoked
    assert result.passed and result.collected is None
    assert "no tests executed" in result.reason


def test_install_only_baseline_then_judge_still_accepts():
    v = Validator({"VALIDATION_CONFIG": {"type": "none"}})
    v.record_baseline("/nonexistent/python")
    assert v.judge(v.run("/nonexistent/python"))[0] is True


# --------------------------------------------- suites that never start

def _judge_against(baseline_failing, candidate):
    from aura.validation import ValidationResult, Validator
    v = Validator({"VALIDATION_CONFIG": {"type": "pytest", "pytest_target": "t"}})
    v.baseline = ValidationResult(passed=not baseline_failing,
                                  failing=set(baseline_failing), collected=19)
    return v.judge(candidate)


def test_a_suite_that_cannot_start_is_rejected_even_if_the_baseline_had_failures():
    """Nothing new appears in the failing set of a suite that never ran, so a
    failing-set comparison alone would accept an upgrade that breaks imports."""
    from aura.validation import ValidationResult
    crashed = ValidationResult(passed=False, failing=set(), collected=None,
                               reason="tests could not start (pytest exit 4): "
                                      "ImportError: cannot import name 'json'")
    accepted, new, _, _ = _judge_against({"tests/t.py::already_broken"}, crashed)
    assert accepted is False
    assert new == [crashed.reason]


def test_the_same_pre_existing_failures_are_still_forgiven():
    from aura.validation import ValidationResult
    same = ValidationResult(passed=False, failing={"tests/t.py::already_broken"},
                            collected=19)
    accepted, new, _, _ = _judge_against({"tests/t.py::already_broken"}, same)
    assert accepted is True and new == []


def test_the_reason_names_what_stopped_the_suite():
    from aura.validation import _startup_error
    output = ("ImportError while loading conftest '/x/tests/conftest.py'.\n"
              "E   ImportError: cannot import name 'json' from 'itsdangerous' "
              "(/x/site-packages/itsdangerous/__init__.py)\n")
    assert _startup_error(output) == \
        "ImportError: cannot import name 'json' from 'itsdangerous'"


# ------------------------------------------------- projects that predate pytest

def _unittest_project(tmp_path, body: str, name: str = "test_demo.py"):
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / name).write_text(body)
    return tmp_path


def test_a_unittest_suite_is_run_by_its_own_runner(tmp_path):
    """Installing pytest to run a unittest suite would add packages the project
    never declared into the environment whose resolution is being measured."""
    from aura.validation import run_unittest

    _unittest_project(tmp_path, """
import unittest

class T(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(1, 1)

    def test_broken(self):
        self.assertEqual(1, 2)
""")
    result = run_unittest(sys.executable, "tests", cwd=str(tmp_path))
    assert result.passed is False
    assert result.collected == 2
    assert result.counts["passed"] == 1
    # A named test, not just an exit code: the judge compares these to the
    # baseline's failures, so "which test" is the whole point.
    assert result.failing == {"test_demo.T.test_broken"}


def test_a_passing_unittest_suite_reports_no_failures(tmp_path):
    from aura.validation import run_unittest

    _unittest_project(tmp_path, """
import unittest

class T(unittest.TestCase):
    def test_ok(self):
        self.assertTrue(True)
""")
    result = run_unittest(sys.executable, "tests", cwd=str(tmp_path))
    assert result.passed and result.failing == set() and result.collected == 1


def test_a_unittest_suite_that_cannot_import_says_so(tmp_path):
    """The same rule as pytest: a suite that never ran is never an upgrade."""
    from aura.validation import run_unittest

    _unittest_project(tmp_path, "import nonexistent_module_xyz\n")
    result = run_unittest(sys.executable, "tests", cwd=str(tmp_path))
    assert not result.passed
    assert "could not start" in result.reason or result.failing


def test_several_pytest_targets_are_all_passed_to_pytest(tmp_path):
    """Some projects name their test files in ways pytest will not collect from
    a directory; naming the files keeps those projects usable as subjects."""
    (tmp_path / "a.py").write_text("def test_a():\n    assert True\n")
    (tmp_path / "b.py").write_text("def test_b():\n    assert True\n")
    result = run_pytest(sys.executable, ["a.py", "b.py"], cwd=str(tmp_path))
    assert result.passed and result.counts["passed"] == 2
    # The same thing written as one string, which is how a registry entry
    # carrying two targets round-trips through JSON.
    again = run_pytest(sys.executable, "a.py b.py", cwd=str(tmp_path))
    assert again.passed and again.counts["passed"] == 2
    # One target only: the single test in a.py, not both files.
    one = run_pytest(sys.executable, "a.py", cwd=str(tmp_path))
    assert one.passed and one.counts["passed"] == 1


def test_a_baseline_may_have_failing_tests_if_it_ran(tmp_path):
    """The judge rejects a candidate for tests that *newly* fail, so a suite
    with pre-existing failures still constrains an upgrade through the tests
    that pass. Demanding a perfect suite discards real projects for a property
    the method does not depend on."""
    from aura.environment import ErrorClass, ProbeResult

    result = ValidationResult(passed=False, failing={"test_b"},
                              counts={"passed": 4, "failed": 1}, collected=5)
    ran = result.counts.get("passed", 0) > 0
    started = "could not start" not in (result.reason or "")
    assert ran and started, "four tests passed: this is usable evidence"

    # A suite that never started is a different thing entirely.
    crashed = ValidationResult(passed=False, failing=set(), counts={},
                               reason="tests could not start (pytest exit 4)")
    assert crashed.counts.get("passed", 0) == 0


def test_a_subject_can_shorten_the_test_timeout_for_every_arm():
    """slicelabeler's 7-second suite spins under some upgraded combinations;
    at the default each such probe held its slot for an hour."""
    from aura.validation import Validator
    default = Validator({"VALIDATION_CONFIG": {"type": "pytest", "pytest_target": "tests"}})
    assert default.timeout_s == 3600
    short = Validator({"VALIDATION_CONFIG": {"type": "pytest", "pytest_target": "tests",
                                             "timeout_s": 300}})
    assert short.timeout_s == 300
