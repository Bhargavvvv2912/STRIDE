"""Tests for the environment layer: error taxonomy, flags, and the real probe."""

import sys
import time
from pathlib import Path

import pytest

from aura.environment import (
    EnvSpec, Environment, ErrorClass, _run, classify_error, probe,
)
from aura.reqfile import ReqFile
from aura.validation import Validator


@pytest.mark.parametrize("stderr,expected", [
    ("ERROR: ResolutionImpossible: for help visit ...", ErrorClass.RESOLUTION_CONFLICT),
    ("ERROR: No matching distribution found for torch==99.0", ErrorClass.NO_MATCHING_DIST),
    ("ERROR: Could not find a version that satisfies numpy==0", ErrorClass.NO_MATCHING_DIST),
    ("note: This error originates from a subprocess.\nerror: metadata-generation-failed",
     ErrorClass.BUILD_FAILURE),
    ("ModuleNotFoundError: No module named 'setuptools'", ErrorClass.BUILD_FAILURE),
    ("ERROR: Package 'x' requires a different Python: 3.11 not in '>=3.12'",
     ErrorClass.PYTHON_VERSION),
    ("Connection reset by peer", ErrorClass.NETWORK),
    ("something else entirely", ErrorClass.OTHER),
    ("", ErrorClass.NONE),
])
def test_error_classification(stderr, expected):
    assert classify_error(stderr) == expected


def test_python_version_wins_over_no_matching_distribution():
    """Both patterns match; the more specific diagnosis must be reported."""
    combined = ("ERROR: Could not find a version that satisfies pandas\n"
                "ERROR: Package requires a different Python: 3.9 not in '>=3.11'")
    assert classify_error(combined) == ErrorClass.PYTHON_VERSION


def test_cache_is_enabled_by_default():
    """v1 forced --no-cache-dir on every install; that was most of the runtime."""
    env = Environment("/tmp/x", EnvSpec())
    assert "--no-cache-dir" not in env._install_flags()


def test_cache_can_be_disabled_for_a_controlled_comparison():
    env = Environment("/tmp/x", EnvSpec(use_cache=False))
    assert "--no-cache-dir" in env._install_flags()


def test_build_isolation_is_on_by_default():
    """v1 forced --no-build-isolation, so sdists built against whatever was present."""
    env = Environment("/tmp/x", EnvSpec())
    assert "--no-build-isolation" not in env._install_flags()
    env2 = Environment("/tmp/x", EnvSpec(build_isolation=False))
    assert "--no-build-isolation" in env2._install_flags()


def test_uv_installer_changes_the_base_command():
    assert Environment("/tmp/x", EnvSpec(installer="uv"))._base_command()[:3] == \
        ["uv", "pip", "install"]
    assert Environment("/tmp/x", EnvSpec(installer="pip"))._base_command()[1:3] == \
        ["-m", "pip"]


def test_cache_dir_is_passed_through():
    flags = Environment("/tmp/x", EnvSpec(cache_dir="/shared/cache"))._install_flags()
    assert flags[:2] == ["--cache-dir", "/shared/cache"]


# ------------------------------------------------------ real end-to-end probe

@pytest.mark.slow
def test_probe_accepts_a_passing_environment(tmp_path):
    """Full path: build a venv, install a real package, run validation, judge it."""
    script = tmp_path / "validate.py"
    script.write_text(
        "import packaging\n"
        "print('AURA_METRICS: {\"score\": 1.0}')\n"
    )
    config = {
        "IS_INSTALLABLE_PACKAGE": False,
        "VALIDATION_CONFIG": {"type": "script", "smoke_test_script": str(script),
                              "project_dir": str(tmp_path)},
    }
    validator = Validator(config)
    rf = ReqFile.from_text("packaging==25.0\n")

    result = probe(rf, config=config, validator=validator,
                   env_path=tmp_path / "venv", record_baseline=True)
    assert result.install_ok and result.accepted
    assert result.validation.metrics == {"score": 1.0}
    assert result.frozen.get("packaging") == "25.0"


@pytest.mark.slow
def test_probe_reports_a_nonexistent_version_as_no_matching_distribution(tmp_path):
    config = {"IS_INSTALLABLE_PACKAGE": False,
              "VALIDATION_CONFIG": {"type": "script", "smoke_test_script": "x.py",
                                    "project_dir": str(tmp_path)}}
    result = probe(ReqFile.from_text("packaging==999.999.999\n"),
                   config=config, validator=Validator(config),
                   env_path=tmp_path / "venv")
    assert not result.install_ok
    assert result.error_class == ErrorClass.NO_MATCHING_DIST


@pytest.mark.slow
def test_probe_rejects_metric_drift_even_when_the_script_exits_zero(tmp_path):
    """Reviewer B's behavioural-preservation case, end to end."""
    script = tmp_path / "validate.py"
    marker = tmp_path / "second_run"
    script.write_text(
        "import pathlib\n"
        f"m = pathlib.Path({str(marker)!r})\n"
        "score = 0.20 if m.exists() else 0.90\n"
        "m.write_text('x')\n"
        "print('AURA_METRICS: {\"accuracy\": %s}' % score)\n"
    )
    config = {"IS_INSTALLABLE_PACKAGE": False,
              "VALIDATION_CONFIG": {"type": "script", "smoke_test_script": str(script),
                                    "project_dir": str(tmp_path)}}
    validator = Validator(config, tolerance=0.01)
    rf = ReqFile.from_text("packaging==25.0\n")

    base = probe(rf, config=config, validator=validator,
                 env_path=tmp_path / "v1", record_baseline=True)
    assert base.accepted and validator.baseline.metrics == {"accuracy": 0.9}

    cand = probe(rf, config=config, validator=validator, env_path=tmp_path / "v2")
    assert cand.install_ok                      # installed fine
    assert cand.validation.passed               # exited zero
    assert not cand.accepted                    # but the number moved
    assert cand.drifted_metrics == ["accuracy"]
    assert cand.error_class == ErrorClass.VALIDATION_DRIFT


# ---------------------------------------------------- reusable dry-run session

def test_dry_run_session_builds_one_sandbox_for_many_questions(monkeypatch, tmp_path):
    """The scan asks once per candidate; it must not pay for a venv each time."""
    from aura import environment as env_mod

    built = {"n": 0}
    real_enter = env_mod.Environment.__enter__

    def counting_enter(self):
        built["n"] += 1
        return self

    monkeypatch.setattr(env_mod.Environment, "__enter__", counting_enter)
    monkeypatch.setattr(env_mod.Environment, "__exit__",
                        lambda self, *a: None)
    monkeypatch.setattr(env_mod.Environment, "dry_run",
                        lambda self, reqfile, workdir: (True, ""))

    rf = ReqFile.from_text("torch==1.12.0\n")
    with env_mod.dry_run_session(env_path=tmp_path / "v", workdir=tmp_path) as resolve:
        for _ in range(8):
            assert resolve(rf)[0] is True

    assert built["n"] == 1


def test_dry_run_session_admits_everything_when_the_sandbox_cannot_be_built(tmp_path,
                                                                           monkeypatch):
    """The pre-filter is an optimisation; losing it costs time, never answers."""
    from aura import environment as env_mod

    def explode(self):
        raise RuntimeError("no disk space")

    monkeypatch.setattr(env_mod.Environment, "__enter__", explode)

    rf = ReqFile.from_text("torch==1.12.0\n")
    with env_mod.dry_run_session(env_path=tmp_path / "v", workdir=tmp_path) as resolve:
        ok, log = resolve(rf)
    assert ok is True and "unavailable" in log


# ------------------------------------------------- interpreter selection

def test_an_environment_is_built_from_the_requested_interpreter(tmp_path):
    """A subject pinned in 2021 must be validated on its own era's Python, not
    on whichever interpreter happens to be running AURA."""
    from aura.environment import EnvSpec, Environment, find_interpreter

    target = find_interpreter("3.10")
    if target is None:
        pytest.skip("Python 3.10 not installed on this machine")

    with Environment(tmp_path / "v", EnvSpec(python=target)) as env:
        out, err, code = _run([env.python, "-c",
                               "import sys; print('%d.%d' % sys.version_info[:2])"],
                              None, 120)
    assert code == 0, err
    assert out.strip() == "3.10"


def test_a_missing_interpreter_is_reported_never_substituted():
    """Silently using the wrong Python yields results that look fine and are not."""
    from aura.environment import find_interpreter
    assert find_interpreter("2.7") is None
    assert find_interpreter("") is not None          # "" means "the current one"


def test_the_runner_refuses_to_run_a_subject_without_its_python(tmp_path):
    from evaluation.arms import Arm
    from evaluation.runner import run_arm
    from evaluation.subjects import Subject

    subject = Subject(name="ancient", url="https://example.invalid/x",
                      python_version="2.7")
    result = run_arm(subject, Arm("baseline", "baseline"),
                     results_dir=tmp_path / "r", workspaces_dir=tmp_path / "w")
    assert result.outcome == "error"
    assert "2.7" in result.reason and "not installed" in result.reason


# ------------------------------------------------------------ venv isolation

@pytest.mark.slow
def test_installs_land_in_the_venv_and_never_in_the_base_interpreter(tmp_path):
    """The bug this guards against silently invalidated every result.

    ``venv/bin/python`` is a symlink to the base interpreter. Resolving it hands
    back the base binary, and running *that* installs into the base environment:
    probes stop being isolated, each inherits whatever the last one installed,
    and ``destroy()`` removes the directory while the packages stay behind. The
    symptom is pip reporting "requirement already satisfied" for a package the
    candidate never asked for.
    """
    from aura.environment import EnvSpec, Environment
    from aura.reqfile import ReqFile

    with Environment(tmp_path / "v", EnvSpec()) as env:
        prefix, _, code = _run(
            [env.python, "-c", "import sys; print(sys.prefix)"], None, 120)
        assert code == 0
        # sys.prefix must be the venv, not the interpreter it was built from.
        assert Path(prefix.strip()).resolve() == (tmp_path / "v").resolve()

        pip_home, _, _ = _run(
            [env.python, "-c",
             "import pip, pathlib; print(pathlib.Path(pip.__file__).parent.parent)"],
            None, 120)
        assert str((tmp_path / "v").resolve()) in str(Path(pip_home.strip()).resolve())

        env.install_requirements(ReqFile.from_text("iniconfig\n"), tmp_path)
        assert "iniconfig" in env.freeze()


def test_the_venv_interpreter_path_is_not_resolved(tmp_path):
    """Cheap guard on the exact line: resolving it reintroduces the bug."""
    from aura.environment import EnvSpec, Environment

    env = Environment(tmp_path / "v", EnvSpec())
    assert env.python.startswith(str((tmp_path / "v").absolute()))
    assert env.python.endswith("python")


# ------------------------------------------------- concurrent probe safety

def test_concurrent_probes_do_not_share_a_requirements_file(tmp_path, monkeypatch):
    """The batching bug: six probes wrote one filename, then installed whichever
    version won the race, and the resulting failures were recorded against the
    wrong candidate as `validation_failed` -- a class the search never rechecks.
    """
    from concurrent.futures import ThreadPoolExecutor

    from aura.environment import EnvSpec, Environment
    from aura.reqfile import ReqFile

    installed: list[str] = []

    def fake_install(self, args, cwd=None, quiet=False):
        path = Path(args[-1])
        installed.append(path.read_text().strip())
        return "", 0

    monkeypatch.setattr(Environment, "pip_install", fake_install)

    versions = [f"1.{i}.0" for i in range(6)]

    def run(version):
        env = Environment(tmp_path / f".aura_probe_{version}", EnvSpec())
        env.install_requirements(ReqFile.from_text(f"click=={version}\n"), tmp_path)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(run, versions))

    # Every probe must have installed exactly its own pin.
    assert sorted(installed) == sorted(f"click=={v}" for v in versions)
    assert len(list(tmp_path.glob("*_requirements.txt"))) == 6


def test_project_builds_are_serialised(tmp_path, monkeypatch):
    """Concurrent `pip install .` on one source tree corrupts its egg-info."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from aura.environment import EnvSpec, Environment

    concurrent, peak = 0, 0
    guard = threading.Lock()

    def fake_install(self, args, cwd=None, quiet=False):
        nonlocal concurrent, peak
        with guard:
            concurrent += 1
            peak = max(peak, concurrent)
        time.sleep(0.02)
        with guard:
            concurrent -= 1
        return "", 0

    monkeypatch.setattr(Environment, "pip_install", fake_install)

    def run(i):
        Environment(tmp_path / f"e{i}", EnvSpec()).install_project(str(tmp_path))

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(run, range(6)))

    assert peak == 1


def test_uv_managed_interpreters_are_found_newest_patch_first(tmp_path, monkeypatch):
    """uv is how a cluster gets several Pythons without root or a compiler."""
    from aura.environment import _uv_interpreters

    for name in ("cpython-3.9.18-linux-x86_64-gnu", "cpython-3.9.23-linux-x86_64-gnu",
                 "cpython-3.10.14-linux-x86_64-gnu", "pypy-3.9.19-linux-x86_64-gnu"):
        exe = tmp_path / name / "bin" / "python3"
        exe.parent.mkdir(parents=True)
        exe.write_text("")
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", str(tmp_path))

    found = _uv_interpreters("3.9")
    assert [p.parents[1].name for p in found] == [
        "cpython-3.9.23-linux-x86_64-gnu", "cpython-3.9.18-linux-x86_64-gnu"]
    assert _uv_interpreters("3.8") == []


# ------------------------------------------- building with the era's own tools

def test_an_install_that_fails_on_a_removed_build_api_is_retried(tmp_path,
                                                                 monkeypatch):
    """setuptools removed Feature in 46 and use_2to3 in 58. A package that
    built fine in its own time then fails for a reason that belongs to this
    machine, and rejecting the subject for it measures our setuptools."""
    from aura.environment import EnvSpec, Environment
    from aura.reqfile import ReqFile

    spec = EnvSpec(build_era_pins=("setuptools==44.1.1", "wheel==0.34.2"))
    env = Environment.__new__(Environment)
    env.spec = spec
    env.path = tmp_path / "venv"

    calls = []

    def fake_pip_install(args, cwd=None, quiet=False):
        calls.append(list(args))
        if args[0] == "setuptools==44.1.1":
            return "installed era tools", 0
        if "--no-build-isolation" in args:
            return "Successfully installed", 0
        return ("ImportError: cannot import name 'Feature' from 'setuptools'", 1)

    env.pip_install = fake_pip_install
    output, code = env.install_requirements(ReqFile.from_text("itsdangerous==0.24\n"),
                                            tmp_path)
    assert code == 0, "the retry should succeed"
    assert calls[1] == ["setuptools==44.1.1", "wheel==0.34.2"]
    assert "--no-build-isolation" in calls[2]
    assert "retried with era build tools" in output


# pip's own words when it skips a release for malformed metadata. The last
# line names the package as missing, which is what the screening log recorded
# and why colindecourt/record was rejected for a file that is on PyPI.
_STRICT_METADATA_OUTPUT = (
    "WARNING: Ignoring version 1.8.1 of pytorch-lightning since it has "
    "invalid metadata:\n"
    "Requested pytorch-lightning==1.8.1 from https://files.pythonhosted.org/"
    "packages/54/8f/pytorch_lightning-1.8.1-py3-none-any.whl has invalid "
    "metadata: .* suffix can only be used with `==` or `!=` operators\n"
    "    torch (>=1.9.*)\n"
    "           ~~~~~~^\n"
    "Please use pip<24.1 if you need to use this version.\n"
    "ERROR: No matching distribution found for pytorch-lightning==1.8.1")


def test_a_release_pip_refuses_for_its_metadata_is_retried_on_the_era_pip(tmp_path):
    """pip 24.1 skips such a release and then calls it missing.

    The project installed under the pip of its own era, so the fallback is to
    that installer -- not a reason to reject the subject.
    """
    from aura.environment import ERA_PIP, EnvSpec, Environment
    from aura.reqfile import ReqFile

    env = Environment.__new__(Environment)
    env.spec = EnvSpec()
    env.path = tmp_path / "venv"
    env.era_pip_used = False
    calls = []

    def fake_pip_install(args, cwd=None, quiet=False):
        calls.append(list(args))
        if args[0] == ERA_PIP:
            return "Successfully installed pip-24.0", 0
        # The old installer accepts what the new one refused.
        if any(a == ERA_PIP for call in calls[:-1] for a in call):
            return "Successfully installed pytorch-lightning-1.8.1", 0
        return _STRICT_METADATA_OUTPUT, 1

    env.pip_install = fake_pip_install
    output, code = env.install_requirements(
        ReqFile.from_text("pytorch-lightning==1.8.1\n"), tmp_path)
    assert code == 0, "the retry should succeed"
    assert calls[1] == [ERA_PIP], "the era installer is put in place first"
    assert "-r" in calls[2], "then the same requirements are installed again"
    assert env.era_pip_used, "the fallback is recorded, not silent"
    assert "retried with pip==24.0" in output
    assert env.spec.pip_pin == ERA_PIP, (
        "later environments for this subject -- the dry-run sandbox above all "
        "-- must start on the installer that was shown to work, or a plan that "
        "keeps this release pinned is called unsatisfiable")


def test_a_genuinely_missing_package_is_not_retried_on_the_era_pip(tmp_path):
    """Without the metadata warning, a missing distribution is really missing:
    cbfeeds is not on PyPI at all, and an older installer cannot conjure it."""
    from aura.environment import EnvSpec, Environment
    from aura.reqfile import ReqFile

    env = Environment.__new__(Environment)
    env.spec = EnvSpec()
    env.path = tmp_path / "venv"
    env.era_pip_used = False
    calls = []

    def fake_pip_install(args, cwd=None, quiet=False):
        calls.append(list(args))
        return ("ERROR: Could not find a version that satisfies the requirement "
                "cbfeeds (from versions: none)\n"
                "ERROR: No matching distribution found for cbfeeds", 1)

    env.pip_install = fake_pip_install
    _, code = env.install_requirements(ReqFile.from_text("cbfeeds\n"), tmp_path)
    assert code == 1 and len(calls) == 1, "no retry when nothing was skipped"
    assert not env.era_pip_used


def test_a_real_dependency_conflict_is_not_retried(tmp_path):
    """The retry is for build tools only: a project whose own pins contradict
    each other has a genuine problem, and hiding that would invent a subject."""
    from aura.environment import EnvSpec, Environment
    from aura.reqfile import ReqFile

    env = Environment.__new__(Environment)
    env.spec = EnvSpec(build_era_pins=("setuptools==44.1.1",))
    env.path = tmp_path / "venv"
    calls = []

    def fake_pip_install(args, cwd=None, quiet=False):
        calls.append(list(args))
        return ("ERROR: Cannot install -r requirements.txt and urllib3==1.26.5 "
                "because these package versions have conflicting dependencies.", 1)

    env.pip_install = fake_pip_install
    _, code = env.install_requirements(ReqFile.from_text("urllib3==1.26.5\n"), tmp_path)
    assert code == 1 and len(calls) == 1, "no retry for a genuine conflict"


def test_the_build_era_is_taken_from_the_pins_not_the_commit():
    """A project committed in 2021 can pin releases from 2017, and those are
    what have to build."""
    from aura.baseline import LEGACY_FLOOR, era_build_tools

    class Index:
        def upload_dates(self, name):
            return {"click": {"6.7": "2017-01-06"}, "jinja2": {"2.11.3": "2021-01-31"}}.get(name, {})

        def newest_at(self, name, on_or_before, python_version=None, specifier=None, prefer_wheels=False):
            # Whatever was current in early 2017 is far below the floor.
            return {"setuptools": "34.3.2", "wheel": "0.30.0"}[name]

    tools = era_build_tools(Index(), {"click": "6.7", "jinja2": "2.11.3"},
                            on_date="2021-06-16")
    # Dated from the 2017 pin, then floored at a version a current interpreter
    # can still run.
    assert tools == (f"setuptools=={LEGACY_FLOOR['setuptools']}",
                     f"wheel=={LEGACY_FLOOR['wheel']}")


def test_environments_are_built_on_local_disk_when_asked(tmp_path, monkeypatch):
    """On the cluster the workspace is on NFS, where building environments
    stalled; the job script moves them to the node's disk. Distinct probes must
    stay distinct there, and nothing moves when the variable is unset."""
    from aura.environment import Environment
    ws = tmp_path / "workspaces" / "htb-api" / "aura_llm_qwen32b"
    monkeypatch.delenv("AURA_VENV_ROOT", raising=False)
    assert Environment(ws / ".aura_probe_0").path == (ws / ".aura_probe_0").absolute()

    local = tmp_path / "local"
    monkeypatch.setenv("AURA_VENV_ROOT", str(local))
    a = Environment(ws / ".aura_probe_0").path
    b = Environment(ws / ".aura_probe_1").path
    other_arm = Environment(ws.parent / "aura_linear_greedy" / ".aura_probe_0").path
    assert a.parent == local and a != b and a != other_arm
