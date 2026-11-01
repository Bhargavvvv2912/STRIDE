"""Shared fixtures: a simulated package world for deterministic agent tests."""

import json
from dataclasses import dataclass, field

import pytest

from aura.environment import ErrorClass, ProbeResult
from aura.index import PackageIndex, ReleaseHistory
from aura.reqfile import safe_parse_version
from packaging.version import Version
from aura.validation import ValidationResult


class FakeIndex(PackageIndex):
    def __init__(self, table, fail=(), requires=None, dates=None):
        super().__init__()
        self.table, self.fail = table, set(fail)
        # (name, version) -> Requires-Dist strings. Absent means "unknown",
        # which the constraint graph must treat as no information.
        self.requires = requires or {}
        # name -> {version: YYYY-MM-DD}; only the baseline reconstruction asks.
        self.dates = dates or {}

    def upload_dates(self, name):
        return self.dates.get(name, {})

    def newest_at(self, name, on_or_before, python_version=None, specifier=None, prefer_wheels=False):
        eligible = [v for v, d in self.dates.get(name, {}).items()
                    if d <= on_or_before]
        if not eligible:
            return None
        return max(eligible, key=lambda v: safe_parse_version(v) or Version("0"))

    def _fetch(self, name):
        from aura.index import IndexError_
        if name in self.fail:
            raise IndexError_(f"boom: {name}")
        return self.table.get(name)

    def requires_dist(self, name, version):
        """Never touch the network from a test."""
        return self.requires.get((name, version))


def history(name, versions, pre=()):
    return ReleaseHistory(name=name, versions=tuple(versions), prereleases=tuple(pre))


@dataclass
class FakeWorld:
    """A package universe with an explicit rule for which pin sets are valid.

    ``accepts`` is a predicate over the full pin mapping. Everything the agent
    does reduces to asking this question, so tier behaviour can be tested
    exactly without building a single virtualenv.
    """

    releases: dict = field(default_factory=dict)
    accepts: callable = lambda pins: True
    baseline_ok: bool = True
    calls: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def index(self, fail=()):
        return FakeIndex({n: history(n, v) for n, v in self.releases.items()}, fail=fail)

    def probe(self, reqfile, *, config, validator, spec=None, env_path=None,
              record_baseline=False):
        pins = reqfile.pins()
        self.calls.append({"pins": dict(pins), "baseline": record_baseline})

        if record_baseline:
            # A predicate when the test is about which baseline works (the
            # repair ladder tries several); a flat verdict otherwise.
            ok = (self.baseline_ok(pins) if callable(self.baseline_ok)
                  else self.baseline_ok)
            validation = ValidationResult(passed=ok, metrics=dict(self.metrics), collected=5)
            validator.baseline = validation
            return ProbeResult(
                install_ok=ok, accepted=ok, validation=validation,
                frozen=dict(pins), duration_s=0.01,
                error_class=ErrorClass.NONE if ok else ErrorClass.VALIDATION_FAILED,
                stderr="" if ok else "baseline validation failed",
            )

        verdict = self.accepts(pins)
        # A verdict may be True, a message, or (message, error_class) when the
        # test needs to distinguish a fact about the candidate from an artefact
        # of running several probes at once.
        error_class = ErrorClass.RESOLUTION_CONFLICT
        if isinstance(verdict, tuple):
            verdict, error_class = verdict
        ok = verdict is True
        stderr = "" if ok else (verdict if isinstance(verdict, str) else "conflict")
        validation = ValidationResult(passed=ok, metrics=dict(self.metrics), collected=5)
        return ProbeResult(
            install_ok=ok, accepted=ok, validation=validation, frozen=dict(pins),
            duration_s=0.01, stderr=stderr,
            error_class=ErrorClass.NONE if ok else error_class,
        )


@pytest.fixture
def project(tmp_path):
    """A minimal repo: requirements.txt, a source file, a validation script."""
    def build(requirements, source="import torch\nimport numpy\n"):
        (tmp_path / "requirements.txt").write_text(requirements)
        (tmp_path / "main.py").write_text(source)
        (tmp_path / "validate.py").write_text("print('ok')\n")
        return tmp_path
    return build


@pytest.fixture
def config(tmp_path):
    return {
        "PROJECT_NAME": "demo",
        "IS_INSTALLABLE_PACKAGE": False,
        "REQUIREMENTS_FILE": "requirements.txt",
        "PRIMARY_REQUIREMENTS_FILE": "primary_requirements.txt",
        "VALIDATION_CONFIG": {"type": "script", "smoke_test_script": "validate.py",
                              "project_dir": str(tmp_path)},
        "ON_BROKEN_BASELINE": "fail",
    }


def llm_reply(plan, resolvable=True, explanation="because"):
    return json.dumps({"explanation": explanation, "resolvable": resolvable,
                       "plausible": True, "proposed_plan": plan})
