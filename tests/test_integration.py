"""End-to-end integration: the real agent, real pip, real PyPI.

Small pure-Python packages are used so the test is fast, but nothing is mocked:
virtualenvs are built, wheels are downloaded, and validation is executed.
"""

import pytest

from aura.agent import AuraAgent
from aura.environment import EnvSpec
from aura.index import PackageIndex
from aura.llm import NullLLM
from aura.telemetry import Outcome, RunConfig, RunLog, read_log

pytestmark = pytest.mark.slow


def build_project(root, requirements, validation):
    (root / "requirements.txt").write_text(requirements)
    (root / "validate.py").write_text(validation)
    (root / "main.py").write_text("import packaging\nimport packaging.version\n")
    return {
        "PROJECT_NAME": "integration-demo",
        "IS_INSTALLABLE_PACKAGE": False,
        "REQUIREMENTS_FILE": "requirements.txt",
        "PRIMARY_REQUIREMENTS_FILE": "primary_requirements.txt",
        "VALIDATION_CONFIG": {"type": "script", "smoke_test_script": str(root / "validate.py"),
                              "project_dir": str(root)},
        "ON_BROKEN_BASELINE": "fail",
    }


def run_agent(root, config, tiers=("linear",), **kw):
    rc = RunConfig(repo="integration-demo", arm="test", tiers=tuple(tiers),
                   timeout_s=kw.pop("timeout_s", 900), **kw)
    agent = AuraAgent(config, run_config=rc, llm=NullLLM(), index=PackageIndex(),
                      run_log=RunLog(root / "run.jsonl", rc),
                      env_spec=EnvSpec(use_cache=True, build_isolation=True),
                      workdir=root)
    return agent.run()


def test_agent_modernizes_a_real_outdated_dependency(tmp_path):
    """packaging 21.0 -> latest, verified by an actual import and behaviour check."""
    config = build_project(
        tmp_path,
        "packaging==21.0\n",
        "import packaging\n"
        "from packaging.version import parse\n"
        "assert parse('2.0') > parse('1.0')\n"
        "print('AURA_METRICS: {\"ok\": 1}')\n",
    )
    result = run_agent(tmp_path, config)

    assert result.outcome is Outcome.MODERNIZED
    assert "packaging" in result.updated
    old, new = result.updated["packaging"]
    assert old == "21.0" and new != "21.0"
    # The requirements file really was rewritten
    assert f"packaging=={new}" in (tmp_path / "requirements.txt").read_text()
    # And it genuinely got newer
    assert result.version_age_before["mean_releases_behind"] > \
           result.version_age_after["mean_releases_behind"]


def test_agent_keeps_the_baseline_when_the_new_version_breaks_validation(tmp_path):
    """A validation that only passes on the old version must block the upgrade."""
    config = build_project(
        tmp_path,
        "packaging==21.0\n",
        "import packaging\n"
        "assert packaging.__version__ == '21.0', 'pinned behaviour required'\n",
    )
    result = run_agent(tmp_path, config)

    assert result.outcome is Outcome.KEPT_BASELINE
    assert result.updated == {}
    assert (tmp_path / "requirements.txt").read_text() == "packaging==21.0\n"


def test_agent_reports_failure_on_a_broken_baseline(tmp_path):
    """v1 could not express this outcome; it repaired the baseline and continued."""
    config = build_project(
        tmp_path, "packaging==21.0\n", "raise SystemExit('baseline is broken')\n")
    result = run_agent(tmp_path, config)

    assert result.outcome is Outcome.FAILED
    assert "baseline" in result.reason


def test_run_log_is_complete_and_machine_readable(tmp_path):
    config = build_project(
        tmp_path, "packaging==21.0\n",
        "import packaging\nprint('AURA_METRICS: {\"ok\": 1}')\n")
    run_agent(tmp_path, config)

    events = read_log(tmp_path / "run.jsonl")
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    for required in ("prioritization", "attempt", "validation", "pass_start", "pass_end"):
        assert required in kinds, f"missing {required} event"

    start = events[0]
    assert start["env"]["python"] and start["config"]["llm_temperature"] == 0.0
    end = events[-1]
    assert end["counters"]["attempts"] >= 1
    assert end["counters"]["install_seconds"] > 0
    assert "llm_calls" not in end["counters"]      # a deterministic arm made none


def test_comments_extras_and_markers_survive_a_real_run(tmp_path):
    config = build_project(
        tmp_path,
        "# pinned deps\n"
        "packaging==21.0  # version utilities\n",
        "import packaging\n",
    )
    run_agent(tmp_path, config)
    text = (tmp_path / "requirements.txt").read_text()
    assert text.startswith("# pinned deps\n")
    assert "# version utilities" in text
