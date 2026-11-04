"""The retry rule every arm shares (stride.repair._agent_attempts)."""
import subprocess
from pathlib import Path
from types import SimpleNamespace

from stride import guard
from stride.coders import AgentRun
from stride.repair import _agent_attempts

SOURCE = "def keep():\n    return 1\n"


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "mod.py").write_text(SOURCE)
    (root / "tests" / "test_mod.py").write_text("def test_a():\n    pass\n")
    for cmd in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                ["add", "-A"], ["commit", "-qm", "start"]):
        subprocess.run(["git", *cmd], cwd=root, check=True)
    return root


class Validator:
    """Three tests fail; each line 'FIX' in src/mod.py fixes one."""
    def run(self, python):
        text = (self.root / "src" / "mod.py").read_text()
        fixed = text.count("FIX")
        failing = {f"t{i}" for i in range(max(0, 3 - fixed))}
        return SimpleNamespace(failing=failing, output="tail")

    def judge(self, result):
        return (not result.failing, sorted(result.failing), [], None)


class Agent:
    """Runs one scripted edit per call; records the prompts it got."""
    def __init__(self, edits):
        self.edits, self.prompts = list(edits), []

    def run(self, task, workdir, env, out_dir):
        self.prompts.append(task)
        edit = self.edits.pop(0) if self.edits else None
        if edit:
            edit(workdir)
        return AgentRun(exit_status="Submitted", steps=3, seconds=1.0)


def _append(text):
    def edit(root):
        p = root / "src" / "mod.py"
        p.write_text(p.read_text() + text)
    return edit


def _go(tmp_path, edits, **kw):
    root = _repo(tmp_path)
    validator = Validator()
    validator.root = root
    protected = guard.protected_files(root, ["tests"], [])
    agent = Agent(edits)
    out = _agent_attempts(
        task="TASK", workspace=root, protected=protected,
        protected_rel=[str(p.relative_to(root)) for p in protected], agent=agent,
        agent_env={}, out_dir=tmp_path / "out", validator=validator,
        env=SimpleNamespace(python="python", freeze=lambda: {"a": "1"}),
        frozen_before={"a": "1"}, known_failing=[], start_new=3,
        max_attempts=kw.get("max_attempts", 5), patience=kw.get("patience", 2))
    return out, agent, root


def test_retries_build_on_the_last_patch_until_accepted(tmp_path):
    out, agent, root = _go(tmp_path, [_append("# FIX\n"), _append("# FIX\n# FIX\n")])
    assert out["stop"] == "accepted" and out["last"]["reason"] == "repaired"
    assert out["agent"]["calls"] == 2 and out["agent"]["steps"] == 6
    assert "Attempt 2 of 5" in agent.prompts[1] and "kept" in agent.prompts[1]
    assert (root / "src" / "mod.py").read_text().count("FIX") == 3


def test_a_call_that_changes_nothing_stops(tmp_path):
    out, agent, _ = _go(tmp_path, [None])
    assert out["stop"] == "no_change" and out["agent"]["calls"] == 1


def test_no_fewer_failures_for_two_calls_stops(tmp_path):
    out, _, _ = _go(tmp_path, [_append("# a\n"), _append("# b\n"), _append("# FIX\n")])
    assert out["stop"] == "no_progress" and out["agent"]["calls"] == 2


def test_deleted_code_is_undone_before_the_next_call(tmp_path):
    def delete(root):
        (root / "src" / "mod.py").write_text("# FIX\n# FIX\n# FIX\n")
    out, agent, root = _go(tmp_path, [delete, _append("# FIX\n# FIX\n# FIX\n")])
    assert out["log"][0]["reason"] == "removed_code"
    assert "undone" in agent.prompts[1]
    assert out["stop"] == "accepted"
    assert "def keep" in (root / "src" / "mod.py").read_text()


def test_cap_on_calls(tmp_path):
    out, _, _ = _go(tmp_path, [_append("# FIX\n")] * 5, max_attempts=2)
    assert out["stop"] == "max_attempts" and out["agent"]["calls"] == 2
