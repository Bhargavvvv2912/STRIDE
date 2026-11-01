"""Reading pip's failures correctly -- fixtures are real output from a run.

The itsdangerous case, from mean-review-collector: flask had already moved to
1.1.4, which caps itsdangerous below 2.0, so ``itsdangerous==2.2.0`` could not
resolve. Three things went wrong reading that output, each covered here.
"""

from pathlib import Path

import pytest

from aura.environment import ErrorClass, classify_error
from aura.planner import extract_conflicting_packages, parse_conflict_report

FIXTURES = Path(__file__).parent / "fixtures"
CONFLICT = (FIXTURES / "pip_conflict_itsdangerous.txt").read_text()
PY_VERSION = (FIXTURES / "pip_python_version_click.txt").read_text()

INSTALLED = {"attrs", "click", "coverage", "flask", "gunicorn", "importlib-metadata",
             "itsdangerous", "jinja2", "markupsafe", "more-itertools", "packaging",
             "pluggy", "py", "pyparsing", "pytest", "six", "wcwidth", "werkzeug", "zipp"}


# ------------------------------------------------------------- classification

def test_a_resolver_conflict_is_classified_as_one():
    """pip's conflict output also says "no matching distributions available";
    that phrase must not win, or routing never sees the conflict."""
    assert "no matching distributions" in CONFLICT
    assert classify_error(CONFLICT) == ErrorClass.RESOLUTION_CONFLICT


def test_a_version_excluded_by_requires_python_is_classified_as_such():
    assert classify_error(PY_VERSION) == ErrorClass.PYTHON_VERSION


def test_an_unrelated_python_note_does_not_decide_the_class():
    """pip lists skipped-for-Python versions on many failures; it only explains
    this one if the requested version is among them."""
    log = ("ERROR: Ignored the following versions that require a different python "
           "version: 8.2.0 Requires-Python >=3.10\n"
           "ERROR: Could not find a version that satisfies the requirement foo==9.9.9\n"
           "ERROR: No matching distribution found for foo==9.9.9\n")
    assert classify_error(log) == ErrorClass.NO_MATCHING_DIST


# ------------------------------------------------------------------ extraction

def test_pips_conflict_report_is_read_directly():
    report = parse_conflict_report(CONFLICT)
    assert report["requested"] == ["itsdangerous"]
    assert report["constraints"][0]["source"] == "flask"
    assert report["constraints"][0]["target"] == "itsdangerous"
    assert "itsdangerous<2.0" in report["constraints"][0]["text"]


def test_the_blocker_is_the_package_pip_named_not_the_whole_file():
    """The old regex returned all 19 requirement names, because pip echoes
    every line it collects -- so greedy upgraded 13 packages at once."""
    blockers = extract_conflicting_packages(CONFLICT, known=INSTALLED)
    assert blockers == ["flask", "itsdangerous"]


def test_echoed_requirement_lines_are_ignored_when_there_is_no_report():
    log = ("Collecting flask==1.1.4 (from -r reqs.txt (line 4))\n"
           "Collecting werkzeug==1.0.1 (from -r reqs.txt (line 18))\n"
           "ERROR: jinja2 3.1 is incompatible with markupsafe==1.1.1\n")
    assert extract_conflicting_packages(log, known=INSTALLED) == ["jinja2", "markupsafe"]


def test_no_report_means_none():
    assert parse_conflict_report("ERROR: something else entirely") is None
