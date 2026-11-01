"""The one-shot LLM arm: one proposal, installed and tested once."""

from __future__ import annotations

from aura.index import PackageIndex, ReleaseHistory
from aura.llm import ScriptedLLM
from evaluation.oneshot import (VERSIONS_SHOWN, build_prompt, nonexistent,
                                offered_versions, parse_proposal)


class FakeIndex(PackageIndex):
    def __init__(self, table):
        super().__init__()
        self.table = table

    def _fetch(self, name):
        return self.table.get(name)


def hist(name, versions, requires=None):
    return ReleaseHistory(name=name, versions=tuple(versions),
                          metadata={v: {"requires_python": (requires or {}).get(v)}
                                    for v in versions})


def test_the_model_is_shown_only_releases_that_install_on_the_project_python():
    """numpy 2.5.3 needs Python 3.12. Dependabot proposed it to a 3.10 project
    anyway; offering it here would hand the model the same mistake."""
    index = FakeIndex({"numpy": hist("numpy", ["1.23.0", "2.2.6", "2.5.3"],
                                     requires={"2.5.3": ">=3.12"})})
    assert offered_versions(index, "numpy", "1.23.0", "3.10") == ["1.23.0", "2.2.6"]


def test_the_offer_is_capped_to_the_newest_releases():
    many = [f"1.{i}.0" for i in range(40)]
    index = FakeIndex({"boto3": hist("boto3", many)})
    shown = offered_versions(index, "boto3", "1.0.0", "3.11")
    assert len(shown) == VERSIONS_SHOWN and shown[-1] == "1.39.0"


def test_every_package_and_its_options_reach_the_prompt():
    prompt = build_prompt({"flask": "1.1.1", "click": "7.0"},
                          {"flask": ["2.0.0", "3.1.3"], "click": []}, "3.9")
    assert "flask==1.1.1" in prompt and "2.0.0, 3.1.3" in prompt
    assert "click==7.0" in prompt and "no newer release installs" in prompt
    assert "Python 3.9" in prompt and "exactly one attempt" in prompt


def test_a_proposal_is_read_from_prose_and_invented_packages_are_left_out():
    text = ("Here is the upgraded file:\n```\nFlask==3.1.3\nclick==8.1.8\n"
            "gunicorn==23.0.0\n```\nThese are compatible.")
    proposed, invented = parse_proposal(text, known={"flask", "click"})
    assert proposed == {"flask": "3.1.3", "click": "8.1.8"}
    assert invented == ["gunicorn==23.0.0"], (
        "the arm upgrades what the project declares; it does not add packages")


def test_a_version_never_released_is_recorded_as_a_hallucination():
    index = FakeIndex({"flask": hist("flask", ["1.1.1", "3.1.3"])})
    assert nonexistent(index, {"flask": "3.9.9"}) == ["flask==3.9.9"]
    assert nonexistent(index, {"flask": "3.1.3"}) == []


# ----------------------------------------------------------- the arm end to end

def _run(tmp_path, monkeypatch, *, reply, accepted):
    """Drive run_oneshot_arm with a scripted model and a stubbed install."""
    import evaluation.runner as runner_mod
    from aura.reqfile import ReqFile
    from evaluation.arms import Arm
    from evaluation.subjects import Subject

    work = tmp_path / "ws"
    work.mkdir()
    (work / "requirements.txt").write_text("flask==1.1.1\nclick==7.0\n")
    baseline = ReqFile.from_text("flask==1.1.1\nclick==7.0\n")
    installed = {}

    class Ref:
        accepted = True
        error_class = "none"

    class Result:
        install_ok = True
        validation = None
        error_class = "none" if accepted else "validation_failed"
        stderr = ""
        duration_s = 1.0

        def __init__(self, candidate):
            self.accepted = accepted
            self.frozen = candidate.pins()

    def fake_probe(candidate, **_):
        installed["pins"] = candidate.pins()
        return Result(candidate)

    monkeypatch.setattr(runner_mod, "_reference_baseline",
                        lambda *a, **k: (Ref(), {"reqfile": baseline,
                                                 "pins": baseline.pins()}))
    monkeypatch.setattr(runner_mod, "probe", fake_probe)
    monkeypatch.setattr(runner_mod, "build_client",
                        lambda spec, temperature=0.0: ScriptedLLM([reply]))
    monkeypatch.setattr(runner_mod, "version_age_report",
                        lambda index, pins: {"at_latest_pct":
                                             100.0 if pins.get("flask") == "3.1.3" else 0.0})
    monkeypatch.setattr(runner_mod, "_interpreter", lambda subject: "python")
    monkeypatch.setattr("evaluation.oneshot.offered_versions",
                        lambda index, name, current, py: [current])
    monkeypatch.setattr("evaluation.oneshot.nonexistent", lambda index, p: [])

    result = runner_mod.run_oneshot_arm(
        Subject(name="s", python_version="3.11"),
        Arm("llm_oneshot_qwen3b", "baseline", upgrade_source="llm_oneshot",
            llm_spec="scripted"),
        work, tmp_path / "log.jsonl", index=None, timeout_s=60)
    return result, installed


def test_an_accepted_proposal_advances_exactly_what_the_model_changed(tmp_path, monkeypatch):
    result, installed = _run(tmp_path, monkeypatch,
                             reply="flask==3.1.3\nclick==8.1.8\n", accepted=True)
    assert installed["pins"] == {"flask": "3.1.3", "click": "8.1.8"}
    assert result.outcome == "modernized"
    assert result.updated == {"flask": ["1.1.1", "3.1.3"], "click": ["7.0", "8.1.8"]}
    assert result.version_age_after["at_latest_pct"] == 100.0


def test_a_failed_proposal_ends_at_the_baseline_it_started_from(tmp_path, monkeypatch):
    """All or nothing, as pip_upgrade is scored: an environment the tests
    rejected is installed nowhere, so it is not what the arm reached."""
    result, _ = _run(tmp_path, monkeypatch,
                     reply="flask==3.1.3\nclick==8.1.8\n", accepted=False)
    assert result.outcome == "failed" and result.updated == {}
    assert result.version_age_after == result.version_age_before


def test_a_package_the_model_leaves_out_keeps_its_current_pin(tmp_path, monkeypatch):
    result, installed = _run(tmp_path, monkeypatch,
                             reply="flask==3.1.3\n", accepted=True)
    assert installed["pins"] == {"flask": "3.1.3", "click": "7.0"}
    assert result.updated == {"flask": ["1.1.1", "3.1.3"]}


def test_a_subject_environment_is_set_for_the_arm_and_restored_after(monkeypatch):
    """Placeholders like a Django secret or an AWS region reach every test run
    of the arm, and do not leak into the next subject's arms."""
    import os
    from evaluation.runner import subject_environment
    from evaluation.subjects import Subject

    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.delenv("DJANGO_SECRET", raising=False)
    subject = Subject(name="s", environment={"AWS_DEFAULT_REGION": "us-east-1",
                                             "DJANGO_SECRET": "test-only"})
    with subject_environment(subject):
        assert os.environ["AWS_DEFAULT_REGION"] == "us-east-1"
        assert os.environ["DJANGO_SECRET"] == "test-only"
    assert os.environ["AWS_DEFAULT_REGION"] == "eu-west-1", "a prior value comes back"
    assert "DJANGO_SECRET" not in os.environ, "one that was unset is unset again"


def test_the_one_shot_call_is_counted_like_every_other_model_call(tmp_path, monkeypatch):
    """One answer against an iterated search is the comparison this arm is for,
    so its call and tokens land in the same counters the AURA arms use."""
    result, _ = _run(tmp_path, monkeypatch, reply="flask==3.1.3\n", accepted=True)
    assert result.counters.get("llm_calls") == 1
    assert "prompt_tokens" in result.counters and "completion_tokens" in result.counters
