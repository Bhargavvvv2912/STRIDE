"""End-to-end check of one STRIDE repair round on a toy project.

The toy project calls packaging.version.LegacyVersion, removed in packaging 22.
A fake agent that fixes the code must be accepted; one that edits the test
must be rejected and rolled back. Needs network (pip installs packaging).
Run with: STRIDE_NETWORK_TESTS=1 pytest tests/test_stride_repair.py
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aura.environment import EnvSpec
from stride.barriers import find_barriers
from stride.coders import AgentRun, CodeAgent
from stride.repair import repair_round

needs_network = pytest.mark.skipif(not os.environ.get("STRIDE_NETWORK_TESTS"),
                                reason="needs network; set STRIDE_NETWORK_TESTS=1")

CODE = """from packaging.version import LegacyVersion, parse

def is_legacy(v):
    return isinstance(parse(v), LegacyVersion)

def newer(a, b):
    return parse(a) > parse(b)
"""
FIXED = """from packaging.version import InvalidVersion, Version, parse

def is_legacy(v):
    try:
        Version(v)
        return False
    except InvalidVersion:
        return True

def newer(a, b):
    return parse(a) > parse(b)
"""
TEST = """from mod import is_legacy, newer

def test_newer():
    assert newer("2.0", "1.9")

def test_legacy():
    assert is_legacy("not-a-version")
"""


class FixingAgent(CodeAgent):
    name = "fake-fix"

    def run(self, task, workdir, env, out_dir):
        assert "packaging" in task and "python -m pytest" in task
        (workdir / "mod.py").write_text(FIXED)
        return AgentRun(exit_status="Submitted", steps=1)


class RenamingAgent(CodeAgent):
    """Cheats the way `sed -i` does: write a new file, rename it over the test."""
    name = "fake-rename"

    def run(self, task, workdir, env, out_dir):
        import os
        tmp = workdir / "tests" / "tmp_edit"
        try:
            tmp.write_text("def test_newer():\n    assert True\n")
            os.replace(tmp, workdir / "tests" / "test_mod.py")
            outcome = "renamed"
        except PermissionError:
            outcome = "blocked"
        return AgentRun(exit_status=outcome, steps=1)


class CheatingAgent(CodeAgent):
    name = "fake-cheat"

    def run(self, task, workdir, env, out_dir):
        test = workdir / "tests" / "test_mod.py"
        try:
            test.write_text("def test_newer():\n    assert True\n")
        except PermissionError:      # the lock held; cheat through chmod instead
            test.chmod(0o644)
            test.write_text("def test_newer():\n    assert True\n")
        return AgentRun(exit_status="Submitted", steps=1)


def make_project(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "mod.py").write_text(CODE)
    (root / "tests").mkdir()
    (root / "tests" / "test_mod.py").write_text(TEST)
    (root / "requirements.txt").write_text("packaging==21.3\npytest==8.3.3\n")
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=root, check=True)
    return root


def config_for(root: Path) -> dict:
    return {"PROJECT_NAME": "toy", "IS_INSTALLABLE_PACKAGE": False,
            "VALIDATION_CONFIG": {"type": "pytest", "pytest_target": "tests",
                                  "project_dir": str(root)},
            "ENVIRONMENT_SETUP": []}


@needs_network
@pytest.mark.parametrize("agent,accepted,reason", [
    (FixingAgent(), True, "repaired"),
    (CheatingAgent(), False, "edited_protected_files"),
])
def test_repair_round(tmp_path, agent, accepted, reason):
    root = make_project(tmp_path / "toy")
    result = repair_round(
        label="packaging 21.3 -> 22.0", project="toy", workspace=root,
        requirements_file="requirements.txt", pins={"packaging": "22.0"},
        moved=["packaging"], config=config_for(root),
        spec=EnvSpec(python=sys.executable), baseline={"failing": [], "counts": {}},
        agent=agent, out_dir=tmp_path / "out")
    assert result.install_ok
    assert any(f.startswith("tests/test_mod.py") for f in result.failing_before)
    assert result.accepted is accepted, result
    assert result.reason == reason
    if accepted:
        assert "packaging==22.0" in (root / "requirements.txt").read_text()
        assert result.files_changed == 1
    else:
        # rolled back: old pin, original code and test
        assert "packaging==21.3" in (root / "requirements.txt").read_text()
        assert (root / "mod.py").read_text() == CODE
        assert (root / "tests" / "test_mod.py").read_text() == TEST


def test_find_barriers_picks_smallest_breaking_release():
    events = [
        {"event": "attempt", "package": "torch", "install_ok": True, "accepted": False,
         "plan": {"torch": "2.8.0", "torchvision": "0.23.0"}, "new_failures": ["t::a"]},
        {"event": "attempt", "package": "torch", "install_ok": True, "accepted": False,
         "plan": {"torch": "2.1.0", "torchvision": "0.16.0"}, "new_failures": ["t::a"]},
        {"event": "attempt", "package": "numpy", "install_ok": False, "accepted": False,
         "plan": {"numpy": "2.0.0"}, "new_failures": []},
    ]
    found = find_barriers(events, {"torch": "1.11.0", "numpy": "1.26.4"},
                          {"torch": 30, "numpy": 31})
    assert [b.package for b in found] == ["torch"]
    assert found[0].target == "2.1.0" and found[0].crosses_major
    assert found[0].plan["torchvision"] == "0.16.0"


def test_triage_separates_code_from_dependency_failures():
    from stride.triage import classify
    dep = """_____ test_query _____
tests/service.py:40: in test_query
    r = client.get("/q")
../.stride_env/lib/python3.9/site-packages/starlette/testclient.py:470: in get
    with anyio.start_blocking_portal() as p:
E   AttributeError: module 'anyio' has no attribute 'start_blocking_portal'
"""
    code = """_____ test_x _____
tests/test_m.py:5: in test_x
    m.run()
app/model.py:12: in run
    torch.foo()
../env/lib/python3.9/site-packages/torch/__init__.py:1: in foo
E   AttributeError: module 'torch' has no attribute 'foo'
"""
    d = classify(dep, ["tests"])
    assert d.kind == "dependency" and d.implicated[:2] == ["anyio", "starlette"]
    c = classify(code, ["tests"])
    assert c.kind == "code" and c.project_frames == ["app/model.py"]


def test_scoped_agent_widens_through_coupled_packages():
    from types import SimpleNamespace
    from stride.scoped import make_scoped_agent

    Scoped = make_scoped_agent({"torch"})
    agent = Scoped.__new__(Scoped)                       # skip AURA's heavy __init__
    agent.graph = SimpleNamespace(couples=lambda: {frozenset(("torch", "torchvision")),
                                                   frozenset(("numpy", "pandas"))})
    agent.log = SimpleNamespace(emit=lambda *a, **k: None, bump=lambda *a, **k: None)
    names = ["torch", "torchvision", "numpy", "pandas", "timm"]
    cands = [SimpleNamespace(name=n) for n in names]
    from aura.agent import AuraAgent
    original = AuraAgent.build_candidates
    AuraAgent.build_candidates = lambda self, r, s, st: list(cands)
    try:
        pins = {"torch": "2.0.0", "torchvision": "0.15.1", "numpy": "1.26.4",
                "pandas": "2.0.0", "timm": "0.5.4"}
        req = SimpleNamespace(pins=lambda: dict(pins))
        kept = [c.name for c in agent.build_candidates(req, {}, {})]
        assert kept == ["torch", "torchvision"]          # numpy, pandas, timm skipped
        pins["torchvision"] = "0.16.0"                     # moved: no new neighbours
        kept = [c.name for c in agent.build_candidates(req, {}, {})]
        assert kept == ["torch", "torchvision"]
    finally:
        AuraAgent.build_candidates = original


def test_carry_forward_keeps_barriers_outside_the_scope():
    from stride.barriers import Barrier
    from stride.scoped import carry_forward
    old = [Barrier("torch", "1.11.0", "2.0.0", {}, []),
           Barrier("numpy", "1.26.4", "2.0.0", {}, []),
           Barrier("timm", "0.5.4", "0.6.5", {}, [])]
    fresh = [Barrier("torch", "2.0.0", "2.3.0", {}, [])]
    pins = {"torch": "2.0.0", "numpy": "1.26.4", "timm": "0.6.0"}
    kept = carry_forward(old, fresh, pins)
    assert [(b.package, b.current) for b in kept] == [("torch", "2.0.0"), ("numpy", "1.26.4")]


@needs_network
def test_lock_blocks_rename_over_test(tmp_path):
    root = make_project(tmp_path / "toy")
    result = repair_round(
        label="packaging 21.3 -> 22.0", project="toy", workspace=root,
        requirements_file="requirements.txt", pins={"packaging": "22.0"},
        moved=["packaging"], config=config_for(root),
        spec=EnvSpec(python=sys.executable), baseline={"failing": [], "counts": {}},
        agent=RenamingAgent(), out_dir=tmp_path / "out")
    assert result.agent["exit_status"] == "blocked"
    assert (root / "tests" / "test_mod.py").read_text() == TEST
    assert os.access(root / "tests", os.W_OK)          # unlocked afterwards
