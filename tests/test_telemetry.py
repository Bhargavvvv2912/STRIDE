"""Tests for the run log: the substrate every paper table is generated from."""

import json
import pytest

from aura.telemetry import (
    LLMUsage, Outcome, RunConfig, RunLog, Tier, estimate_cost, read_log, read_logs,
)


@pytest.fixture
def log(tmp_path):
    cfg = RunConfig(repo="demo", arm="aura_full", llm_model="gemini-2.5-flash")
    return RunLog(tmp_path / "run.jsonl", cfg), tmp_path / "run.jsonl"


def test_run_start_records_config_and_environment(log):
    rl, path = log
    rl.finish(Outcome.KEPT_BASELINE)
    events = read_log(path)
    start = events[0]
    assert start["event"] == "run_start"
    assert start["config"]["arm"] == "aura_full"
    assert start["config"]["llm_temperature"] == 0.0
    # The 'which Python / which machine' question Reviewer B asked
    assert start["env"]["python"]
    assert "platform" in start["env"]
    assert "aura_commit" in start["env"]


def test_every_outcome_is_representable(log):
    """v1 could only report success; all six states must round-trip."""
    rl, path = log
    rl.finish(Outcome.TRUNCATED_TIMEOUT, reason="6h budget")
    end = read_log(path)[-1]
    assert end["outcome"] == "truncated_timeout"
    assert end["reason"] == "6h budget"
    assert {o.value for o in Outcome} == {
        "modernized", "kept_baseline", "failed",
        "truncated_passes", "truncated_timeout", "error",
    }


def test_attempt_events_accumulate_counters(log):
    rl, path = log
    rl.attempt(package="torch", tier=Tier.LINEAR, plan={"torch": "2.0.1"},
               install_ok=True, validate_ok=False, duration_s=12.5)
    rl.attempt(package="torch", tier=Tier.GREEDY, plan={"torch": "2.0.0"},
               install_ok=True, validate_ok=True, duration_s=7.5, accepted=True)
    assert rl.counters["attempts"] == 2
    assert rl.counters["attempts_linear"] == 1
    assert rl.counters["attempts_greedy"] == 1
    assert rl.counters["install_seconds"] == 20.0
    rl.finish(Outcome.MODERNIZED, updated={"torch": ("1.9.0", "2.0.0")})
    events = read_log(path)
    attempts = [e for e in events if e["event"] == "attempt"]
    assert [a["accepted"] for a in attempts] == [False, True]
    assert events[-1]["updated"] == {"torch": ["1.9.0", "2.0.0"]}
    assert events[-1]["updated_n"] == 1


def test_llm_call_tracks_tokens_and_cost(log):
    rl, path = log
    usage = LLMUsage(model="gemini-2.5-flash", prompt_tokens=10_000,
                     completion_tokens=1_000, latency_s=2.2).finalize()
    rl.llm_call(usage, purpose="co_resolution", package="torch", iteration=1,
                proposed=["torch==2.0.1"],
                rejected=[{"req": "torch==99.0", "reason": "not_on_index"}])
    assert rl.counters["llm_calls"] == 1
    assert rl.counters["prompt_tokens"] == 10_000
    assert rl.counters["llm_rejected_versions"] == 1
    assert rl.counters["llm_cost_usd"] == pytest.approx(0.30 * 0.01 + 2.50 * 0.001)
    call = [e for e in read_log(path) if e["event"] == "llm_call"][0]
    assert call["rejected"][0]["reason"] == "not_on_index"


def test_cost_estimation_matches_published_rates():
    # 1M prompt + 1M completion tokens at the 2.5-flash rate
    assert estimate_cost("gemini-2.5-flash", 1_000_000, 1_000_000) == pytest.approx(2.80)
    # Self-hosted open models carry no per-token price
    assert estimate_cost("qwen2.5-coder-32b-instruct", 1_000_000, 1_000_000) == 0.0
    assert estimate_cost("unknown-model", 1_000_000, 0) == 0.0


def test_conflict_detection_computes_precision(log):
    rl, path = log
    rl.conflict_detection(
        package="torch",
        extracted=["torchvision", "numpy", "notapackage", "alsofake"],
        installed=["torch", "torchvision", "numpy"],
    )
    ev = [e for e in read_log(path) if e["event"] == "conflict_detection"][0]
    assert ev["in_env"] == ["numpy", "torchvision"]
    assert ev["precision"] == 0.5  # 2 of 4 extracted names are real


def test_validation_records_new_failures_not_just_a_count(log):
    rl, path = log
    rl.validation(phase="candidate", passed=False,
                  failing=["test_a", "test_c"], new_failures=["test_c"],
                  mode="baseline_relative")
    ev = [e for e in read_log(path) if e["event"] == "validation"][0]
    assert ev["new_failures"] == ["test_c"]
    assert ev["failing_n"] == 2 and ev["new_failures_n"] == 1


def test_behavior_check_is_recorded(log):
    rl, path = log
    rl.behavior_check(baseline={"accuracy": 0.91}, candidate={"accuracy": 0.62},
                      drifted=["accuracy"], tolerance=0.01, verdict="drift")
    ev = [e for e in read_log(path) if e["event"] == "behavior_check"][0]
    assert ev["verdict"] == "drift" and ev["drifted_n"] == 1


def test_timed_context_manager_records_duration(log):
    rl, path = log
    with rl.timed("note", label="slow thing") as extra:
        extra["detail"] = "x"
    ev = [e for e in read_log(path) if e["event"] == "note"][0]
    assert ev["duration_s"] >= 0 and ev["detail"] == "x"


def test_log_survives_a_truncated_final_line(tmp_path):
    """A run killed at the timeout must still leave a readable partial log."""
    p = tmp_path / "partial.jsonl"
    p.write_text('{"event":"run_start"}\n{"event":"attempt","pack')
    assert [e["event"] for e in read_log(p)] == ["run_start"]


def test_read_logs_aggregates_across_arms(tmp_path):
    for arm in ("aura_full", "llm_only"):
        rl = RunLog(tmp_path / arm / "run.jsonl", RunConfig(repo="demo", arm=arm))
        rl.finish(Outcome.MODERNIZED)
    events = read_logs(tmp_path)
    assert {e["arm"] for e in events} == {"aura_full", "llm_only"}
    assert len([e for e in events if e["event"] == "run_end"]) == 2


# ------------------------------------------------ re-running an arm safely

def test_rerunning_an_arm_does_not_merge_into_the_previous_log(tmp_path):
    """Appending would double every attempt and report two runs as one.

    The reader takes the first run_start and the last run_end, so a merged log
    looks valid and is silently wrong -- the worst outcome for a campaign where
    subjects get re-run after a crash or a timeout.
    """
    from aura.telemetry import Outcome, RunConfig, RunLog, Tier, read_log

    path = tmp_path / "arm.jsonl"
    first = RunLog(path, RunConfig(repo="s", arm="a"))
    first.attempt(package="torch", tier=Tier.LINEAR, plan={"torch": "1.0"},
                  install_ok=True, validate_ok=True, duration_s=1.0, accepted=True)
    first.finish(Outcome.MODERNIZED, updated={"torch": ["0.9", "1.0"]})

    second = RunLog(path, RunConfig(repo="s", arm="a"))
    second.attempt(package="torch", tier=Tier.LINEAR, plan={"torch": "2.0"},
                   install_ok=True, validate_ok=True, duration_s=1.0, accepted=True)
    second.finish(Outcome.MODERNIZED, updated={"torch": ["0.9", "2.0"]})

    events = read_log(path)
    assert sum(1 for e in events if e["event"] == "run_start") == 1
    assert sum(1 for e in events if e["event"] == "attempt") == 1
    assert [e for e in events if e["event"] == "attempt"][0]["plan"] == {"torch": "2.0"}


def test_the_previous_run_is_archived_not_destroyed(tmp_path):
    """Nothing collected over days of compute should be deleted by a re-run."""
    from aura.telemetry import Outcome, RunConfig, RunLog

    path = tmp_path / "arm.jsonl"
    RunLog(path, RunConfig(repo="s", arm="a")).finish(Outcome.MODERNIZED)
    RunLog(path, RunConfig(repo="s", arm="a")).finish(Outcome.FAILED)

    backups = sorted(b.name.split(".")[1] for b in tmp_path.glob("*.bak"))
    # Both the data file and the readable log are archived, so the new log
    # never mixes two runs.
    assert backups == ["jsonl", "log"]
    # And the archive must stay out of the results glob the report reads.
    assert list(tmp_path.glob("**/*.jsonl")) == [path]


# ------------------------------------------------- readable attempt lines

def _render(tmp_path, **attempt):
    from aura.telemetry import RunConfig, RunLog, Tier
    log = RunLog(tmp_path / "a.jsonl", RunConfig(repo="s", arm="a"),
                 console_path=tmp_path / "a.log")
    defaults = dict(package="flask", tier=Tier.LINEAR, plan={"flask": "2.0.0"},
                    install_ok=True, validate_ok=True, duration_s=8.0)
    log.attempt(**{**defaults, **attempt})
    return (tmp_path / "a.log").read_text()


def test_installation_and_validation_are_reported_on_separate_lines(tmp_path):
    text = _render(tmp_path, validate_ok=False, accepted=False,
                   error_class="validation_failed", validate_s=3.0,
                   tests={"passed": 13, "failed": 6},
                   failing=["t::a", "t::b"], new_failures=["t::a", "t::b"])
    assert "installation : ok" in text
    assert "validation   : FAILED  13 passed, 6 failed" in text
    assert "newly failing: t::a, t::b" in text
    assert "result       : REJECTED (tests fail)" in text


def test_an_install_failure_says_the_tests_never_ran(tmp_path):
    text = _render(tmp_path, install_ok=False, validate_ok=False,
                   error_class="resolution_conflict",
                   error_excerpt="ERROR: Cannot install flask==2.0.0 and werkzeug==0.16")
    assert "installation : FAILED (resolution_conflict)" in text
    assert "ERROR: Cannot install flask==2.0.0" in text
    assert "validation   : not run" in text


def test_pre_existing_failures_are_not_shown_as_a_rejection(tmp_path):
    """Baseline-relative judging: tests already failing do not count."""
    text = _render(tmp_path, validate_ok=False, accepted=True,
                   tests={"passed": 17, "failed": 2},
                   failing=["t::old1", "t::old2"], new_failures=[])
    assert "only failures already present at baseline" in text
    assert "result       : ACCEPTED" in text


def test_reference_and_upgrade_probes_are_labelled(tmp_path):
    text = _render(tmp_path, package="<reference>", accepted=True)
    assert "[original pins]" in text


def test_ollama_host_selects_the_server_a_job_started(monkeypatch):
    """Each cluster job serves Ollama on its own port; clients must follow it."""
    from aura.llm import build_client

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert build_client("ollama:llama3.1:8b").base_url == "http://localhost:11434/v1"
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:12401")
    assert build_client("ollama:llama3.1:8b").base_url == "http://127.0.0.1:12401/v1"
