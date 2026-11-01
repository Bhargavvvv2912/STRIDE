"""Execution validation with baseline-relative judgement and behavioural checks.

Two things v1 got wrong here:

1. **The soft pass.** ``ACCEPTABLE_FAILURE_THRESHOLD`` compared an *absolute*
   count of failures, so a repo with 3 pre-existing failures accepted a
   candidate that introduced 3 completely different ones. We instead record the
   set of failing test ids in the baseline and reject a candidate only if a
   *new* id fails.
2. **Exit-code-only oracles.** For script-validated repos v1 checked the return
   code and parsed metrics with regexes hardcoded to one repository, so for 31
   of 32 subjects "passing" meant "did not crash". We extract numeric metrics
   generically and compare them against the baseline within a tolerance, which
   is what makes a claim of behavioural preservation checkable.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Projects can emit an explicit machine-readable record, which is always
# preferred over scraping. One line, anywhere in stdout:
#     AURA_METRICS: {"accuracy": 0.91, "loss": 0.13}
METRICS_LINE_RE = re.compile(r"^\s*AURA_METRICS:\s*(\{.*\})\s*$", re.MULTILINE)

# Fallback scrape: "Final accuracy: 0.9123", "loss = 1.4e-3", "BLEU  27.4"
METRIC_SCRAPE_RE = re.compile(
    r"^[^\S\n]*(?P<key>[A-Za-z][A-Za-z0-9 _.\-]{0,48}?)"
    r"\s*[:=]\s*"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*%?\s*$",
    re.MULTILINE,
)

# pytest short-summary lines: "FAILED tests/test_x.py::test_y - AssertionError"
PYTEST_NODE_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?)(?:\s+-.*)?$", re.MULTILINE)
PYTEST_COUNTS_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)")


@dataclass
class ValidationResult:
    passed: bool
    failing: set[str] = field(default_factory=set)
    metrics: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    output: str = ""
    duration_s: float = 0.0
    reason: str = ""
    collected: int | None = None

    def summary(self) -> str:
        if self.counts:
            return ", ".join(f"{v} {k}" for k, v in sorted(self.counts.items()))
        return "passed" if self.passed else self.reason or "failed"


def extract_metrics(output: str) -> dict[str, float]:
    """Pull numeric metrics out of validation stdout.

    An explicit ``AURA_METRICS:`` JSON line wins outright; otherwise we scrape
    ``key: value`` pairs. Scraped keys are normalised so cosmetic changes in
    log formatting do not look like a metric appearing or disappearing.
    """
    explicit: dict[str, float] = {}
    for match in METRICS_LINE_RE.finditer(output):
        try:
            for k, v in json.loads(match.group(1)).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    explicit[str(k).strip().lower()] = float(v)
        except (json.JSONDecodeError, AttributeError):
            continue
    if explicit:
        return explicit

    scraped: dict[str, float] = {}
    for match in METRIC_SCRAPE_RE.finditer(output):
        key = re.sub(r"[\s_.\-]+", "_", match.group("key").strip().lower()).strip("_")
        if not key or key in {"line", "file", "version", "python", "usage"}:
            continue
        try:
            scraped[key] = float(match.group("value"))
        except ValueError:
            continue
    return scraped


def compare_metrics(
    baseline: dict[str, float], candidate: dict[str, float], tolerance: float = 0.01
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Return the keys that drifted beyond ``tolerance`` (relative), with detail.

    Keys absent from either side are ignored rather than treated as drift: a run
    that simply prints less is not evidence of a behavioural regression.
    """
    drifted: list[str] = []
    detail: dict[str, dict[str, float]] = {}
    for key, base_value in baseline.items():
        if key not in candidate:
            continue
        cand_value = candidate[key]
        denominator = max(abs(base_value), 1e-12)
        relative = abs(cand_value - base_value) / denominator
        detail[key] = {"baseline": base_value, "candidate": cand_value,
                       "relative_delta": round(relative, 6)}
        if relative > tolerance:
            drifted.append(key)
    return drifted, detail


def _run(command: list[str], cwd: str | None, timeout_s: int) -> tuple[str, int]:
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              cwd=cwd, timeout=timeout_s)
        return f"{proc.stdout}\n{proc.stderr}", proc.returncode
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout_s}s", 124


# unittest -v prints "test_name (dotted.Class) ... ok". Python 3.11 adds a
# docstring or the method path after the class, hence the tolerant middle.
UNITTEST_RESULT_RE = re.compile(
    r"^(\w+) \(([\w.]+)\)[^.\n]*\.\.\. "
    r"(ok|FAIL|ERROR|skipped|expected failure|unexpected success)",
    re.M)


# Plugins the harness keeps out of every run. no:typeguard: newer setuptools
# vendors typeguard and registers it as a pytest plugin, which adds an ini
# option of type "string" -- an assertion error in pytest before 6.2, so a
# project pinning old pytest crashed before collecting anything
# (authors-of-leisure on the cluster; mfa-python-example-app in screening),
# for a reason in the installer, not the project.
PYTEST_PLUGIN_FLAGS = ("-p", "no:cacheprovider", "-p", "no:typeguard")


def run_pytest(python_executable: str, target: str, cwd: str | None = None,
               timeout_s: int = 3600, extra_args: list[str] | None = None) -> ValidationResult:
    """Run a pytest target and recover the *set of failing node ids*."""
    # A subject may need several targets named explicitly (tests that pytest
    # will not collect from a directory because of how they are named).
    targets = target if isinstance(target, (list, tuple)) else str(target).split()
    command = [python_executable, "-m", "pytest", *[str(t) for t in targets],
               "-q", "--tb=no", "-rfE", *PYTEST_PLUGIN_FLAGS]
    command += extra_args or []
    start = time.time()
    output, code = _run(command, cwd, timeout_s)
    duration = time.time() - start

    failing = set(PYTEST_NODE_RE.findall(output))
    counts: dict[str, int] = {}
    for value, label in PYTEST_COUNTS_RE.findall(output):
        counts[label.rstrip("s")] = counts.get(label.rstrip("s"), 0) + int(value)

    collected_match = re.search(r"(\d+)\s+tests? collected", output)
    collected = int(collected_match.group(1)) if collected_match else None
    if collected == 0 or (code == 5 and not failing):
        return ValidationResult(False, failing, {}, counts, output, duration,
                                "pytest collected 0 tests", collected)

    # Exit code 0 means everything passed; anything else is judged by the
    # caller against the baseline's failing set, not by the code alone.
    reason = ""
    if code != 0:
        reason = f"pytest exit {code}"
        if not failing:
            # No test was named, so the suite never really ran -- typically an
            # import error while loading conftest.py. Say what broke, since
            # "pytest exit 4" alone tells a reader nothing.
            reason = f"tests could not start (pytest exit {code})"
            cause = _startup_error(output)
            if cause:
                reason += f": {cause}"
    return ValidationResult(code == 0, failing, extract_metrics(output), counts,
                            output, duration, reason, collected)


def run_unittest(python_executable: str, target: str, cwd: str | None = None,
                 timeout_s: int = 3600) -> ValidationResult:
    """Run a project's unittest suite and recover the set of failing tests.

    Projects that predate pytest are a large part of the population this work is
    about, and their runner is part of the subject. Installing pytest to run
    them instead would add packages the project never declared -- new versions
    of pluggy, packaging and more -- into the very environment whose dependency
    resolution is being measured.

    ``-v`` is what makes this comparable to the pytest path: it prints one line
    per test, so a failure is a named test rather than an exit code, and the
    judge can tell a newly broken test from one that was already failing.
    """
    target = str(target or ".")
    root = Path(cwd or ".") / target
    if target.endswith(".py"):                       # one file
        command = [python_executable, "-m", "unittest", "discover", "-v",
                   "-s", str(Path(target).parent or "."), "-p", Path(target).name]
    elif root.is_dir():                              # a directory of tests
        command = [python_executable, "-m", "unittest", "discover", "-v",
                   "-s", target]
    else:                                            # a dotted module or class
        command = [python_executable, "-m", "unittest", "-v", target]
    start = time.time()
    output, code = _run(command, cwd, timeout_s)
    duration = time.time() - start

    def name(match) -> str:
        # Python 3.11 prints "test_x (pkg.Class.test_x)"; older versions print
        # "test_x (pkg.Class)". Either way the id is wanted exactly once.
        path, method = match.group(2), match.group(1)
        return path if path.endswith("." + method) else f"{path}.{method}"

    failing = {name(m) for m in UNITTEST_RESULT_RE.finditer(output)
               if m.group(3) in ("FAIL", "ERROR")}
    counts: dict[str, int] = {}
    for m in UNITTEST_RESULT_RE.finditer(output):
        label = {"ok": "passed", "FAIL": "failed", "ERROR": "error",
                 "skipped": "skipped", "expected failure": "xfailed",
                 "unexpected success": "xpassed"}.get(m.group(3))
        if label:
            counts[label] = counts.get(label, 0) + 1
    ran = re.search(r"^Ran (\d+) tests?", output, re.M)
    collected = int(ran.group(1)) if ran else None

    if collected == 0:
        return ValidationResult(False, failing, {}, counts, output, duration,
                                "unittest collected 0 tests", collected)
    reason = ""
    if code != 0:
        reason = f"unittest exit {code}"
        if not failing:
            reason = f"tests could not start (unittest exit {code})"
            cause = _startup_error(output)
            if cause:
                reason += f": {cause}"
    return ValidationResult(code == 0, failing, extract_metrics(output), counts,
                            output, duration, reason, collected)


def _startup_error(output: str) -> str:
    """The error that stopped a test suite from starting, in one line."""
    lines = output.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("E ") and "Error" in stripped:
            text = stripped[1:].strip()
            return re.sub(r"\s*\((?:/[^)]*)\)\s*$", "", text)[:200]
    for line in lines:
        if "Error" in line and ("conftest" in line or "collecting" in line):
            return line.strip()[:200]
    return ""


def run_script(python_executable: str, script: str, cwd: str | None = None,
               timeout_s: int = 3600) -> ValidationResult:
    """Run a validation script, capturing both its exit code and its numbers."""
    start = time.time()
    output, code = _run([python_executable, str(script)], cwd, timeout_s)
    duration = time.time() - start
    failing = set() if code == 0 else {f"script:{Path(script).name}"}
    return ValidationResult(code == 0, failing, extract_metrics(output), {},
                            output, duration,
                            "" if code == 0 else f"script exit {code}")


class Validator:
    """Runs a project's validation and judges candidates relative to the baseline."""

    def __init__(self, config: dict, tolerance: float = 0.01,
                 timeout_s: int = 3600, require_metric_parity: bool = True):
        vc = config.get("VALIDATION_CONFIG", {})
        self.kind = vc.get("type", "script")
        self.script = vc.get("smoke_test_script")
        self.pytest_target = vc.get("pytest_target")
        # The project's own runner: pytest for most, unittest for those that
        # predate it. Its target is a directory, a file, or a dotted module.
        self.unittest_target = vc.get("unittest_target")
        self.cwd = vc.get("project_dir") or "."
        # Run the suite from a subdirectory: tensorgrad vendors tensorly as
        # ./tensorly/tensorly, so from the root `import tensorly` finds the
        # outer folder as an empty namespace package instead of the library.
        if vc.get("test_dir"):
            self.cwd = str(Path(self.cwd) / vc["test_dir"])
        self.tolerance = tolerance
        # A subject may set its own: slicelabeler's suite runs in 7 s, but some
        # upgraded combinations (a newer pyfakefs with a newer nibabel) make it
        # spin, and each such probe held a slot for the full hour before being
        # rejected. The limit is the subject's, so every arm gets the same one.
        self.timeout_s = int(vc.get("timeout_s", timeout_s))
        self.require_metric_parity = require_metric_parity
        self.baseline: ValidationResult | None = None

    # ------------------------------------------------------------------ run

    def command(self, python_executable: str) -> list[str] | None:
        """The argv this validator executes, so instrumentation can wrap it.

        ``None`` means there is nothing to measure -- the install-only regime
        runs no code, which is precisely the thing coverage is meant to expose.
        """
        if self.kind == "pytest":
            targets = (self.pytest_target
                       if isinstance(self.pytest_target, (list, tuple))
                       else str(self.pytest_target).split())
            return [python_executable, "-m", "pytest", *[str(t) for t in targets],
                    "-q", "--tb=no", "-rfE", *PYTEST_PLUGIN_FLAGS]
        if self.kind == "unittest":
            return [python_executable, "-m", "unittest", "discover", "-v",
                    "-s", str(self.unittest_target or ".")]
        if self.kind in ("script", "smoke_test_with_pytest_report"):
            return [python_executable, str(self.script)]
        return None

    def run(self, python_executable: str) -> ValidationResult:
        if self.kind == "none":
            # The install-only regime: a configuration is accepted as soon as it
            # installs. Used to reproduce what a purely structural safe-update
            # analysis would conclude, so it can be validated after the fact.
            return ValidationResult(True, set(), {}, {}, "", 0.0,
                                    "install-only: no tests executed", collected=None)
        if self.kind == "pytest":
            return run_pytest(python_executable, self.pytest_target, self.cwd, self.timeout_s)
        if self.kind == "unittest":
            return run_unittest(python_executable, self.unittest_target, self.cwd,
                                self.timeout_s)
        if self.kind == "script":
            return run_script(python_executable, self.script, self.cwd, self.timeout_s)
        if self.kind == "smoke_test_with_pytest_report":
            smoke = run_script(python_executable, self.script, self.cwd, self.timeout_s)
            if not smoke.passed:
                return smoke
            suite = run_pytest(python_executable, self.pytest_target, self.cwd, self.timeout_s)
            suite.metrics = {**smoke.metrics, **suite.metrics}
            suite.output = smoke.output + "\n" + suite.output
            return suite
        return ValidationResult(False, set(), {}, {}, "", 0.0,
                                f"unknown validation type {self.kind!r}")

    def record_baseline(self, python_executable: str) -> ValidationResult:
        """Establish the reference: which tests already fail, and what numbers we expect."""
        self.baseline = self.run(python_executable)
        return self.baseline

    # ---------------------------------------------------------------- judge

    def judge(self, candidate: ValidationResult) -> tuple[bool, list[str], list[str], dict]:
        """Accept a candidate iff it introduces no new failure and no metric drift.

        Returns ``(accepted, new_failures, drifted_metrics, metric_detail)``.
        """
        if self.baseline is None:
            return candidate.passed, sorted(candidate.failing), [], {}

        if candidate.collected == 0:
            return False, ["<no tests collected>"], [], {}

        new_failures = sorted(candidate.failing - self.baseline.failing)
        if new_failures:
            return False, new_failures, [], {}

        # A failed run that names no failing test never really ran: an import
        # error, a crash, a timeout. That is never acceptable -- including when
        # the baseline already had failing tests. Comparing failing sets alone
        # would let such a run through, because nothing new appears in a suite
        # that did not execute at all.
        if not candidate.passed and not candidate.failing:
            return False, [candidate.reason or "validation failed"], [], {}

        drifted, detail = compare_metrics(
            self.baseline.metrics, candidate.metrics, self.tolerance
        )
        if drifted and self.require_metric_parity:
            return False, [], drifted, detail
        return True, [], drifted, detail
