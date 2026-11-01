"""The dependabot arm is collected from GitHub; check what gets written."""

import importlib.util
import json
from pathlib import Path

import pytest

from aura.telemetry import read_log

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def rd(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "real_dependabot", ROOT / "scripts" / "real_dependabot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(mod, "repo_name", lambda name: f"me/dependabot-check-{name}")
    monkeypatch.setattr(mod, "subject", lambda name: {
        "name": name, "commit": "abc", "python_version": "3.9",
        "requirements_file": "requirements.txt"})
    files = {"abc": "Click==7.0\nFlask==1.1.1\nsix==1.13.0\n",
             "main": "Click==7.0\nFlask==1.1.1\nsix==1.17.0\n"}
    monkeypatch.setattr(mod, "_file_at", lambda repo, path, ref: files[ref])
    monkeypatch.setattr(mod, "_test_main", lambda repo: ("success", "https://x/run/1"))
    monkeypatch.setattr(mod, "_failure_reason",
                        lambda repo, url: "flask 3.1.3 depends on click>=8.1.3")
    import aura.index
    monkeypatch.setattr(aura.index, "PackageIndex", lambda: None)
    monkeypatch.setattr(aura.index, "version_age_report", lambda index, pins: {})
    return mod


def _pr(n, pkg, old, new, state, ci, merged=None, hours_ago=10):
    return {"number": n, "url": f"https://x/pull/{n}", "package": pkg, "from": old,
            "to": new, "state": state, "created": "2026-01-01T00:00:00Z",
            "updated": "2026-01-01T00:00:00Z", "merged": merged,
            "ci_status": "completed", "ci": ci, "ci_url": "https://x/runs/5"}


def test_collect_writes_the_arm_from_what_github_did(rd, monkeypatch, tmp_path):
    monkeypatch.setattr(rd, "_prs", lambda repo: [
        _pr(1, "click", "7.0", "8.5.0", "open", "failure"),
        _pr(2, "flask", "1.1.1", "3.1.3", "open", "failure"),
        _pr(3, "six", "1.13.0", "1.17.0", "merged", "success",
            merged="2026-01-01T00:00:00Z")])
    rd.collect("demo", settle_hours=3, force=False)

    base = tmp_path / "results" / "demo" / "demo__dependabot"
    events = read_log(base.with_suffix(".jsonl"))
    end = events[-1]
    assert end["outcome"] == "modernized"
    assert end["updated"] == {"six": ["1.13.0", "1.17.0"]}
    assert sorted(end["unchanged"]) == ["click", "flask"]
    proposals = [e for e in events if e["event"] == "proposal"]
    assert [(p["package"], p["accepted"]) for p in proposals] == [
        ("click", False), ("flask", False), ("six", True)]
    assert (base.parent / "demo__dependabot.requirements.txt").read_text().endswith(
        "six==1.17.0\n")
    assert json.loads(base.with_suffix(".github.json").read_text())["main_tests"] == "success"
    text = base.with_suffix(".log").read_text()
    assert "PR #3" in text and "merged" in text
    assert "flask 3.1.3 depends on click>=8.1.3" in text


def test_collect_waits_until_dependabot_has_settled(rd, monkeypatch):
    monkeypatch.setattr(rd, "SETTLE_WAIT_S", 0)      # no real waiting in a test
    fresh = _pr(1, "six", "1.13.0", "1.17.0", "open", "")
    fresh["ci_status"] = "in_progress"
    monkeypatch.setattr(rd, "_prs", lambda repo: [fresh])
    with pytest.raises(SystemExit, match="not settled"):
        rd.collect("demo", settle_hours=3, force=False)


def test_a_broken_final_main_is_a_failure_not_a_modernisation(rd, monkeypatch):
    monkeypatch.setattr(rd, "_prs", lambda repo: [
        _pr(3, "six", "1.13.0", "1.17.0", "merged", "success",
            merged="2026-01-01T00:00:00Z")])
    monkeypatch.setattr(rd, "_test_main", lambda repo: ("failure", "u"))
    rd.collect("demo", settle_hours=3, force=False)
    end = read_log(rd.RESULTS_DIR / "demo" / "demo__dependabot.jsonl")[-1]
    assert end["outcome"] == "failed"


def test_poetry_tables_are_dropped_so_dependabot_bumps_the_measured_file(rd, tmp_path):
    """htb-api declared Flask and Werkzeug in requirements.txt and in Poetry's
    tables. Dependabot bumped the Poetry side, CI installed requirements.txt,
    and 13 merges "passed" against pins nothing had changed."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "pyproject.toml").write_text(
        '[tool.poetry]\nname = "x"\nversion = "1.0"\n\n'
        '[tool.poetry.dependencies]\npython = "^3.8"\nrequests = "^2.28"\n\n'
        '[tool.poetry.dev-dependencies]\nFlask = "^2.0"\n\n'
        '[tool.poetry.group.docs.dependencies]\nSphinx = "^4"\n\n'
        '[build-system]\nrequires = ["poetry-core"]\n'
        'build-backend = "poetry.core.masonry.api"\n')
    (work / "poetry.lock").write_text("# lock\n")

    assert rd._drop_poetry_dependencies(work)
    text = (work / "pyproject.toml").read_text()
    assert 'python = "^3.8"' in text, "the interpreter constraint is metadata"
    for gone in ("requests", "Flask", "Sphinx", "dev-dependencies", "group.docs"):
        assert gone not in text
    assert 'name = "x"' in text and "build-backend" in text, "it must still install"
    assert not (work / "poetry.lock").exists()


def test_a_pyproject_without_poetry_dependencies_is_left_alone(rd, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.black]\nline-length = 100\n")
    assert not rd._drop_poetry_dependencies(tmp_path)
    assert (tmp_path / "pyproject.toml").read_text() == "[tool.black]\nline-length = 100\n"
