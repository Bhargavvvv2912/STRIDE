"""Regression tests for walk screening's acceptance and evidence boundaries."""
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
from packaging.version import Version
from scripts import walk_support


@pytest.mark.parametrize("result,expected", [
    (None, False), ((0, 0, 0), False), ((11, 1, 0), False),
    ((11, 0, 1), False), ((11, 0, 2), False), ((11, 0, 124), False),
    ((11, 0, 0), True),
])
def test_only_clean_success_is_accepted(result, expected):
    assert walk_support.clean_result(result) is expected


def test_conflict_names_reads_pip_report():
    text = ("The conflict is caused by:\n    The user requested dask==2023.5.0\n"
            "    distributed 2021.8.1 depends on dask==2021.08.1\n")
    assert walk_support.conflict_names(text, {"dask", "distributed", "pandas"}) == {"dask", "distributed"}


def test_greedy_must_reach_every_target_version():
    target = {"a": "2.0", "b": "3.0"}
    assert not walk_support.reaches_target({"a": "4.0", "b": "2.0"}, target, Version)
    assert walk_support.reaches_target({"a": "4.0", "b": "3.0"}, target, Version)


def test_logger_preserves_output_exit_and_timeout(tmp_path):
    logger = walk_support.ProbeLogger(tmp_path / "logs", tmp_path)
    r = logger.run([sys.executable, "-c", "print('11 passed'); raise SystemExit(2)"])
    assert r.returncode == 2 and "11 passed" in r.stdout
    assert json.loads((tmp_path / "logs/00001-command.json").read_text())["returncode"] == 2
    r = logger.run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.05)
    assert r.returncode == 124 and "TIMEOUT" in r.stderr


@pytest.mark.parametrize("output,exit_code", [
    ("11 passed", 2), ("no tests ran", 5), ("2 failed", 1),
])
def test_checker_stops_at_invalid_baseline(tmp_path, monkeypatch, capsys, output, exit_code):
    src = tmp_path / "src"
    src.mkdir()
    (src / "requirements.txt").write_text("example==1.0\n")
    calls = []

    class FakeLogger:
        def __init__(self, root, cwd):
            self.root = root
        def run(self, args, **kwargs):
            calls.append(args)
            testing = "pytest" in args
            return subprocess.CompletedProcess(args, exit_code if testing else 0,
                                               output if testing else "", "")

    monkeypatch.setattr(walk_support, "ProbeLogger", FakeLogger)
    monkeypatch.setitem(sys.modules, "walk_support", walk_support)
    monkeypatch.setenv("WALK_SRC", str(src))
    monkeypatch.setenv("WALK_TMP", str(tmp_path / "tmp"))
    monkeypatch.setattr(sys, "argv", ["walk_check.py", str(src / "requirements.txt"), "example", "tests"])
    script = Path(__file__).resolve().parents[1] / "scripts/walk_check.py"
    with pytest.raises(SystemExit):
        runpy.run_path(str(script), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "baseline does not run"
    assert result["pytest_exit"] == exit_code
    assert sum("pytest" in cmd for cmd in calls) == 1


def test_missing_test_module_is_repaired_once_and_recorded(tmp_path, monkeypatch, capsys):
    """AURA's rule: one test-only module may be added, and the repair is recorded."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "requirements.txt").write_text("example==1.0\n")
    calls = []

    class FakeLogger:
        def __init__(self, root, cwd):
            self.root = root
        def run(self, args, **kwargs):
            calls.append(args)
            testing = "pytest" in args
            return subprocess.CompletedProcess(args, 2 if testing else 0,
                                               "No module named 'sklearn'" if testing else "", "")

    monkeypatch.setattr(walk_support, "ProbeLogger", FakeLogger)
    monkeypatch.setitem(sys.modules, "walk_support", walk_support)
    monkeypatch.setenv("WALK_SRC", str(src))
    monkeypatch.setenv("WALK_TMP", str(tmp_path / "tmp"))
    monkeypatch.delenv("EXTRA", raising=False)
    monkeypatch.setattr(sys, "argv", ["walk_check.py", str(src / "requirements.txt"), "example", "tests"])
    with pytest.raises(SystemExit):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/walk_check.py"), run_name="__main__")
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "baseline does not run"
    assert sum("pytest" in cmd for cmd in calls) == 2          # one retry, no more
    assert any("scikit-learn" in cmd for cmd in calls)


def test_baseline_with_preexisting_failures_is_accepted():
    """AURA tolerates tests that already fail at baseline; only new failures count."""
    assert walk_support.runnable_baseline((11, 2, 1))
    assert walk_support.acceptable((11, 2, 1), (11, 2, 1))
    assert not walk_support.acceptable((10, 2, 1), (11, 2, 1))   # lost a pass
    assert not walk_support.acceptable((11, 3, 1), (11, 2, 1))   # new failure
    assert not walk_support.acceptable((11, 0, 2), (11, 2, 1))   # pytest interrupted


def test_offline_preflight_fails_before_installing(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src"
    src.mkdir()
    (src / "requirements.txt").write_text("example==1.0\n")
    calls = []

    class FakeLogger:
        def __init__(self, root, cwd):
            self.root = root
        def run(self, args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 1, "", "Operation not permitted")

    monkeypatch.setattr(walk_support, "ProbeLogger", FakeLogger)
    monkeypatch.setitem(sys.modules, "walk_support", walk_support)
    monkeypatch.setenv("WALK_SRC", str(src))
    monkeypatch.setenv("WALK_TMP", str(tmp_path / "tmp"))
    monkeypatch.setenv("WALK_REQUIRE_OFFLINE", "1")   # fail closed only when required
    monkeypatch.setattr(sys, "argv", ["walk_check.py", str(src / "requirements.txt"), "example", "tests"])
    with pytest.raises(SystemExit):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/walk_check.py"), run_name="__main__")
    assert json.loads(capsys.readouterr().out)["error"] == "offline isolation unavailable"
    assert len(calls) == 1 and "pip" not in calls[0]
