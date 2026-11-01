"""Isolated environment construction and the single install-and-validate probe.

v1 had *two* probe paths that behaved differently: the linear tier installed the
project with ``pip install .`` and passed requirements via ``-r``, while the
co-resolution tier installed ``./{project_dir}`` with ``cwd=project_dir`` and
passed requirements as command-line arguments. Since the ablation compares those
tiers against each other, a difference attributed to a *strategy* could just as
easily have been an artefact of the harness. There is one path here.

It also turns caching back on. v1 passed ``--no-cache-dir`` to every install, so
each candidate re-downloaded multi-gigabyte CUDA wheels; with a shared cache the
same wheel is fetched once per machine.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import venv
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from packaging.utils import canonicalize_name

from aura.reqfile import ReqFile, parse_freeze
from aura.validation import ValidationResult, Validator


class ErrorClass:
    """Failure taxonomy, so the paper can report *why* candidates fail."""

    NONE = "none"
    RESOLUTION_CONFLICT = "resolution_conflict"   # resolver could not satisfy constraints
    NO_MATCHING_DIST = "no_matching_distribution"  # version/wheel does not exist for this env
    BUILD_FAILURE = "build_failure"                # sdist compile / metadata generation failed
    PYTHON_VERSION = "python_version"              # requires-python excludes this interpreter
    NETWORK = "network"
    PROJECT_BUILD = "project_build_failure"        # the subject project itself failed to build
    VALIDATION_FAILED = "validation_failed"
    VALIDATION_DRIFT = "behavioural_drift"         # tests green, numbers moved
    TIMEOUT = "timeout"
    OTHER = "other"


_ERROR_PATTERNS: list[tuple[str, re.Pattern]] = [
    (ErrorClass.PYTHON_VERSION, re.compile(
        r"requires a different python|requires-python|python_requires", re.I)),
    (ErrorClass.NO_MATCHING_DIST, re.compile(
        r"no matching distribution|could not find a version", re.I)),
    (ErrorClass.RESOLUTION_CONFLICT, re.compile(
        r"resolutionimpossible|conflicting dependencies|cannot install .* because|"
        r"dependency conflict|incompatible", re.I)),
    (ErrorClass.BUILD_FAILURE, re.compile(
        r"error: (?:metadata-generation-failed|subprocess-exited-with-error)|"
        r"failed building wheel|error: command .* failed|no module named ['\"]?setuptools", re.I)),
    (ErrorClass.NETWORK, re.compile(
        r"connection (?:error|reset|timed out)|temporary failure in name resolution|"
        r"read timed out|network is unreachable", re.I)),
]


_DEFINITE_CONFLICT = re.compile(
    r"resolutionimpossible|conflicting dependencies|the conflict is caused by", re.I)
_NO_MATCH_FOR = re.compile(r"no matching distribution found for\s+(\S+)", re.I)
_IGNORED_FOR_PYTHON = re.compile(
    r"ignored the following versions that require a different python version:(.*)",
    re.I)


def classify_error(stderr: str) -> str:
    """Name the reason an install failed, most specific evidence first.

    pip's output mixes the decisive error with incidental notes, so the order
    matters. A resolver conflict is reported alongside the phrase "no matching
    distributions available", and pip lists versions it skipped for requiring a
    newer Python on almost every failure -- so neither phrase alone says why
    *this* install failed. The conflict report is checked first; a Python-version
    verdict is given only when the version actually requested is among the ones
    pip skipped for that reason.
    """
    text = stderr or ""
    if _DEFINITE_CONFLICT.search(text):
        return ErrorClass.RESOLUTION_CONFLICT
    missing = _NO_MATCH_FOR.search(text)
    if missing:
        requested = missing.group(1)
        version = requested.split("==", 1)[1] if "==" in requested else ""
        ignored = _IGNORED_FOR_PYTHON.search(text)
        if version and ignored and re.search(
                rf"(?<![\w.]){re.escape(version)}\s+Requires-Python", ignored.group(1)):
            return ErrorClass.PYTHON_VERSION
        return ErrorClass.NO_MATCHING_DIST
    for name, pattern in _ERROR_PATTERNS:
        if pattern.search(text):
            return name
    return ErrorClass.OTHER if stderr else ErrorClass.NONE


@dataclass
class EnvSpec:
    """Everything about *how* an environment is built. Recorded in the run log."""

    python: str = ""                       # interpreter to build venvs from;
                                           # "" means the one running AURA
    installer: str = "pip"                 # "pip" | "uv"
    use_cache: bool = True                 # v1 forced --no-cache-dir everywhere
    build_isolation: bool = True           # v1 forced --no-build-isolation everywhere
    cache_dir: str | None = None
    install_timeout_s: int = 3600
    index_url: str | None = None
    extra_pip_args: tuple[str, ...] = ()
    # Build tools as they were when the project was last touched. pip builds a
    # source distribution in an isolated environment holding *today's*
    # setuptools, and an old package's setup.py often uses an API today's
    # setuptools has removed (``setuptools.Feature`` went in 46). The failure
    # then says more about this machine's era than about the package, exactly
    # as an unpinned transitive dependency does. Applied through PIP_CONSTRAINT,
    # which pip honours inside the isolated build environment too.
    build_constraints: str | None = None
    # The installer itself, when this subject has been shown to need an older
    # one -- see ``_retry_with_era_pip``. Set once a probe has proved it, and
    # from then on every environment built from this spec starts with it: a
    # plan that *keeps* such a release pinned has to resolve in the dry-run
    # sandbox too, and a sandbox on a current pip would call it unsatisfiable.
    pip_pin: str = ""
    # Era-appropriate build tools, as ``("setuptools==44.1.1", "wheel==0.34.2")``.
    # Installed and used without build isolation only as a retry, when an
    # install has already failed on a build-tool API that no longer exists.
    build_era_pins: tuple[str, ...] = ()


@dataclass
class ProbeResult:
    install_ok: bool = False
    setup_failed: bool = False
    build_ok: bool = True
    accepted: bool = False
    validation: ValidationResult | None = None
    new_failures: list[str] = field(default_factory=list)
    drifted_metrics: list[str] = field(default_factory=list)
    metric_detail: dict = field(default_factory=dict)
    error_class: str = ErrorClass.NONE
    stderr: str = ""
    duration_s: float = 0.0
    frozen: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.accepted


def attempt_details(result: "ProbeResult") -> dict:
    """The parts of a probe a reader needs to see install and tests separately.

    Recorded on every attempt event so the log can say *which* stage failed and,
    for a test failure, which tests -- rather than a single pass/fail word that
    hides whether the candidate never installed or installed and broke.
    """
    details: dict = {
        "new_failures": sorted(getattr(result, "new_failures", None) or [])[:50],
        "drifted": list(getattr(result, "drifted_metrics", None) or []),
        "setup_failed": bool(getattr(result, "setup_failed", False)),
    }
    validation = getattr(result, "validation", None)
    if validation is not None:
        details.update(
            tests=dict(getattr(validation, "counts", None) or {}),
            failing=sorted(getattr(validation, "failing", None) or [])[:50],
            validate_s=round(getattr(validation, "duration_s", 0.0) or 0.0, 2),
            validation_reason=getattr(validation, "reason", "") or "",
        )
    return details


def _run(cmd: list[str], cwd: str | None = None, timeout_s: int = 3600,
         env: dict[str, str] | None = None) -> tuple[str, str, int]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                           timeout=timeout_s,
                           env={**os.environ, **env} if env else None)
        return p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        return "", f"TIMEOUT after {timeout_s}s", 124
    except OSError as exc:
        return "", f"{type(exc).__name__}: {exc}", 127


# A build that fails because the *build tool* has moved on, not because the
# package is wrong: setuptools removed Feature in 46 and use_2to3 in 58,
# distutils lost msvccompiler, and Python 3.12 dropped distutils entirely. The
# package built fine in its own time and will again with the tools of that time.
_BUILD_TOOL_ERROR = re.compile(
    r"cannot import name '\w+' from 'setuptools'"
    r"|use_2to3|setuptools\.command\.build_py"
    r"|No module named 'distutils[.\w]*'"
    r"|distutils\.msvccompiler"
    # An old numpy.distutils build under a setuptools that no longer puts
    # CCompiler where it looks: SforAiDl/genrl and DolphinDetection.
    r"|name 'CCompiler' is not defined"
    # setuptools 82 dropped pkg_resources, which many old setup.py files import
    # while pip builds them in isolation (openai-whisper 20231117 for wenet).
    r"|No module named 'pkg_resources'",
    re.I)

# pip 24.1 began enforcing PEP 440 on the metadata of packages it downloads,
# and *skips* every release that fails: pytorch-lightning 1.8.1 declares
# ``torch (>=1.9.*)``, so a current pip reports "No matching distribution
# found" for a file sitting on PyPI. The project installed fine under the pip
# of its own era, and the last line of pip's output names a missing package,
# which reads as a fact about the project rather than about the installer.
# pip says so itself: "Please use pip<24.1 if you need to use this version."
_STRICT_METADATA_ERROR = re.compile(
    r"has invalid metadata|Please use pip<24\.1", re.I)

# The newest pip that still installs such a release. It keeps the 2020 resolver
# and ``--dry-run --report``, both of which the search depends on, so a subject
# that needs it is not otherwise measured differently.
ERA_PIP = "pip==24.0"


# The subject's source tree is shared by every concurrent probe, and building it
# writes metadata back into that tree. Only one build may run at a time.
_PROJECT_BUILD_LOCK = threading.Lock()


def _local_venv_path(path: Path) -> Path:
    """Where an environment is actually built: AURA_VENV_ROOT when it is set.

    On a cluster the workspace is on a shared network filesystem, and building
    environments there -- thousands of small files per install, a dozen probes
    at once -- stalled it: other tasks hung in ensurepip until they timed out.
    The job script points AURA_VENV_ROOT at the node's own disk. The name keeps
    the subject, arm and probe, so every environment stays distinct; unset, the
    environment is built where it was asked for, as before.
    """
    root = os.environ.get("AURA_VENV_ROOT")
    if not root:
        return path
    return Path(root) / "__".join(path.parts[-3:])


class Environment:
    """A throwaway virtualenv with a shared, reusable download cache."""

    def __init__(self, path: str | Path, spec: EnvSpec | None = None):
        # Absolute from the start: validation runs with ``cwd`` set to the
        # subject's directory, so a relative interpreter path would not resolve
        # from there. The *directory* is resolved; the interpreter inside it is
        # deliberately not (see :attr:`python`).
        self.path = _local_venv_path(Path(path).absolute())
        self.spec = spec or EnvSpec()
        # Set when a probe had to fall back to the installer of the pins' era;
        # worth reporting, since it says the release itself is one a current
        # pip refuses rather than one that is missing.
        self.era_pip_used = False

    # ---------------------------------------------------------------- lifecycle

    def create(self) -> "Environment":
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        # Built from an explicit interpreter, not from whichever one happens to
        # be running AURA. A subject pinned in 2021 needs the interpreter of its
        # era, and validating it on a newer one measures the interpreter rather
        # than the dependency change.
        base = self.spec.python or sys.executable
        result = subprocess.run(
            [base, "-m", "venv", "--clear", str(self.path)],
            capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"could not create a virtualenv with {base}: "
                f"{(result.stderr or result.stdout)[-500:]}")
        # PEP 517 builds need setuptools/wheel present; on 3.12+ venv no longer
        # provides setuptools, which is what made v1's --no-build-isolation
        # produce build failures that looked like version incompatibilities.
        self.pip_install(["--upgrade", "pip", "setuptools", "wheel"], quiet=True)
        if self.spec.pip_pin:
            # Downgraded after the bootstrap, not instead of it: setuptools and
            # wheel still have to be present for PEP 517 builds.
            self.pip_install([self.spec.pip_pin], quiet=True)
        return self

    def destroy(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> "Environment":
        return self.create()

    def __exit__(self, *exc) -> None:
        self.destroy()

    @property
    def python(self) -> str:
        """The venv's own interpreter -- deliberately *not* resolved.

        ``venv/bin/python`` is a symlink to the base interpreter, so resolving
        it hands back the base binary. Running that installs into the base
        environment instead of the venv: probes stop being isolated, each one
        inherits whatever the last one installed, and ``destroy()`` deletes the
        directory while leaving the packages behind. Python locates a venv by
        finding ``pyvenv.cfg`` beside the executable that was invoked, which
        only works if the symlink is invoked as itself.
        """
        bindir = "Scripts" if os.name == "nt" else "bin"
        return str(self.path / bindir / "python")

    @property
    def site_packages(self) -> str:
        """Where this environment installs to. Used to confirm isolation."""
        out, _, code = _run(
            [self.python, "-c",
             "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
            timeout_s=120)
        return out.strip() if code == 0 else "" 

    # ----------------------------------------------------------------- install

    def _install_flags(self) -> list[str]:
        flags: list[str] = []
        if not self.spec.use_cache:
            flags.append("--no-cache-dir")
        elif self.spec.cache_dir:
            flags += ["--cache-dir", self.spec.cache_dir]
        if not self.spec.build_isolation:
            flags.append("--no-build-isolation")
        if self.spec.index_url:
            flags += ["--index-url", self.spec.index_url]
        flags += list(self.spec.extra_pip_args)
        return flags

    def _base_command(self) -> list[str]:
        if self.spec.installer == "uv":
            return ["uv", "pip", "install", "--python", self.python]
        return [self.python, "-m", "pip", "install"]

    def pip_install(self, args: list[str], cwd: str | None = None,
                    quiet: bool = False) -> tuple[str, int]:
        cmd = self._base_command() + self._install_flags() + args
        if quiet:
            cmd.append("-q")
        env = None
        if self.spec.build_constraints:
            env = {"PIP_CONSTRAINT": str(self.spec.build_constraints)}
        stdout, stderr, code = _run(cmd, cwd, self.spec.install_timeout_s, env=env)
        return (stdout + "\n" + stderr), code

    def install_requirements(self, reqfile: ReqFile, workdir: Path) -> tuple[str, int]:
        """Install a requirement set into this environment.

        The file is named after *this* environment, not shared across the
        workdir. Batched probing runs several of these at once from the same
        working directory; a single shared filename means one probe overwrites
        another's requirements between the write and the install, and each
        candidate is then judged on whichever version won the race. The failures
        that produces are classified ``validation_failed``, which the search
        treats as a fact about the candidate and never re-checks.
        """
        target = Path(workdir) / f"{self.path.name}_requirements.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        reqfile.save(target)
        output, code = self.pip_install(["-r", str(target.resolve())])
        if code and self.spec.build_era_pins and _BUILD_TOOL_ERROR.search(output):
            # One retry with the build tools of the project's own era. Without
            # it, a subject is rejected for a property of this machine's
            # setuptools rather than of the project -- the same mistake as
            # judging it on an interpreter it never supported.
            retry, retry_code = self._retry_with_era_build_tools(target)
            return output + "\n" + retry, retry_code
        if code and _STRICT_METADATA_ERROR.search(output):
            retry, retry_code = self._retry_with_era_pip(target)
            return output + "\n" + retry, retry_code
        return output, code

    def _retry_with_era_pip(self, target: Path) -> tuple[str, int]:
        """Install the last pip that accepts pre-24.1 metadata, then retry.

        Downgrading the installer is confined to this retry and to this
        environment, which is thrown away after the probe: every arm reaches it
        by the same rule, and a subject that never trips it is never measured on
        an old pip.
        """
        self.era_pip_used = True
        setup_out, setup_code = self.pip_install([ERA_PIP], quiet=True)
        if setup_code:
            return f"could not install {ERA_PIP}: {setup_out}", setup_code
        output, code = self.pip_install(["-r", str(Path(target).resolve())])
        if code == 0:
            # Remembered on the spec, which every later environment for this
            # subject is built from, so the dry-run sandbox agrees with what
            # actually installs here.
            self.spec.pip_pin = ERA_PIP
        return (f"retried with {ERA_PIP} (the release predates pip's strict "
                f"metadata checking)\n" + output), code

    def _retry_with_era_build_tools(self, target: Path) -> tuple[str, int]:
        """Install the era's setuptools/wheel, then build against them."""
        pins = list(self.spec.build_era_pins)
        setup_out, setup_code = self.pip_install(pins, quiet=True)
        if setup_code:
            return f"could not install era build tools {pins}: {setup_out}", setup_code
        # Isolation would reinstate today's setuptools for the build, which is
        # the thing that failed, so the retry builds against this environment.
        output, code = self.pip_install(
            ["--no-build-isolation", "-r", str(Path(target).resolve())])
        return (f"retried with era build tools ({', '.join(pins)})\n" + output), code

    def install_project(self, project_dir: str, extras: str = "",
                        editable: bool = False) -> tuple[str, int]:
        """Install the subject project itself into this environment.

        Serialised across threads. Every concurrent probe builds from the *same*
        source tree, and setuptools writes ``*.egg-info`` (and often ``build/``)
        back into it, so simultaneous builds corrupt each other's metadata. The
        installs still go to separate environments; only the build step, which
        touches shared files, is made exclusive.
        """
        spec = f"{Path(project_dir).resolve()}{extras}"
        # --no-deps: the requirement set under test is already installed, and
        # it is the thing being measured. deep-table's setup.py declares its
        # requirements.txt as install_requires, so installing it with deps put
        # torch 1.9 / pytorch-lightning 1.3 back on top of every candidate and
        # the tests ran against the original stack -- a torch 2.8 jump was
        # "accepted" that cannot even import. The screener installs the same way.
        args = (["--no-deps", "-e", spec] if editable else ["--no-deps", spec])
        with _PROJECT_BUILD_LOCK:
            return self.pip_install(args)

    def dry_run(self, reqfile: ReqFile, workdir: Path) -> tuple[bool, str]:
        """Ask the resolver whether a requirement set is satisfiable.

        ``--dry-run`` installs nothing, so this environment is unchanged and can
        be reused for every candidate in a scan. Creating a fresh virtualenv per
        candidate -- which is what the first version of this code did -- spends
        5-15 seconds on ``venv.create`` plus a pip/setuptools/wheel bootstrap
        before the resolve even starts, which for a package with 30 candidates
        is several minutes of pure overhead per package.
        """
        target = Path(workdir) / f"{self.path.name}_dryrun.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        reqfile.save(target)
        output, code = self.pip_install(["--dry-run", "-r", str(target.resolve())])
        return code == 0, output[-20000:]

    def resolve_closure(self, reqfile: ReqFile, workdir: Path) -> dict[str, str]:
        """Every package a requirement set would install, and at which version.

        A requirements file that pins some of its dependencies and not others
        says nothing about the rest, yet the rest are what a working environment
        is made of. pip's resolution report names them without installing
        anything, so the closure can be reconstructed rather than guessed.

        Returns ``{}`` when the report is unavailable (an older pip, or a set
        that does not resolve at all); the caller falls back to what the file
        declares, which is the honest answer in that case.
        """
        import json

        target = Path(workdir) / f"{self.path.name}_closure.txt"
        report = Path(workdir) / f"{self.path.name}_closure.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        reqfile.save(target)
        _, code = self.pip_install(["--dry-run", "--quiet",
                                    "--report", str(report.resolve()),
                                    "-r", str(target.resolve())])
        if code != 0 or not report.exists():
            return {}
        try:
            payload = json.loads(report.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        closure: dict[str, str] = {}
        for item in payload.get("install", []):
            meta = item.get("metadata") or {}
            name, version = meta.get("name"), meta.get("version")
            if name and version:
                closure[canonicalize_name(name)] = version
        return closure

    def uninstall(self, package: str) -> tuple[str, int]:
        """Remove a package from the environment. Used by oracle mutation checks."""
        stdout, stderr, code = _run(
            [self.python, "-m", "pip", "uninstall", "-y", package],
            timeout_s=600)
        return stdout + "\n" + stderr, code

    def run_setup(self, command: str, cwd: str | None = None) -> tuple[str, int]:
        """Run one environment-setup command inside this environment.

        Shell-quoted by the caller; ``{python}`` is substituted for this
        environment's interpreter so a subject can say what it needs without
        knowing where the venv lives.
        """
        stdout, stderr, code = _run(["/bin/sh", "-c", command], cwd=cwd,
                                    timeout_s=1800)
        return stdout + "\n" + stderr, code

    def freeze(self) -> dict[str, str]:
        stdout, _, _ = _run([self.python, "-m", "pip", "freeze"], timeout_s=300)
        return parse_freeze(stdout)


def _uv_interpreters(version: str) -> list[Path]:
    """Pythons installed by ``uv python install``, newest patch first.

    uv keeps prebuilt interpreters in its own directory (``cpython-3.9.23-...``)
    rather than on PATH. It is the easiest way to get several Python versions on
    a machine without root or a compiler -- a cluster login node, typically -- so
    those installs are searched alongside pyenv's.
    """
    roots = []
    if os.environ.get("UV_PYTHON_INSTALL_DIR"):
        roots.append(Path(os.environ["UV_PYTHON_INSTALL_DIR"]))
    data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    roots.append(Path(data) / "uv" / "python")

    found: list[tuple[list[int], Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for d in root.iterdir():
            parts = d.name.split("-")
            if len(parts) < 2 or parts[0] != "cpython":
                continue
            full = parts[1]
            if full != version and not full.startswith(version + "."):
                continue
            for exe in (f"python{version}", "python3", "python"):
                if (d / "bin" / exe).exists():
                    key = [int(x) if x.isdigit() else 0 for x in full.split(".")]
                    found.append((key, d / "bin" / exe))
                    break
    return [path for _, path in sorted(found, key=lambda item: item[0], reverse=True)]


def find_interpreter(version: str) -> str | None:
    """Locate an interpreter for ``version`` (e.g. "3.10"), or None.

    Returning None rather than falling back to the current interpreter is
    deliberate. Silently validating a subject on the wrong Python produces
    results that look ordinary and mean nothing, and the mismatch would not
    appear anywhere in the logs.
    """
    if not version:
        return sys.executable

    current = ".".join(str(p) for p in sys.version_info[:2])
    if version == current:
        return sys.executable

    candidates: list[Path] = []
    pyenv_root = Path(os.environ.get("PYENV_ROOT", Path.home() / ".pyenv"))
    versions_dir = pyenv_root / "versions"
    if versions_dir.is_dir():
        matches = sorted(
            (d for d in versions_dir.iterdir()
             if d.is_dir() and (d.name == version or d.name.startswith(version + "."))),
            key=lambda d: [int(x) if x.isdigit() else 0 for x in d.name.split(".")],
        )
        candidates += [d / "bin" / "python" for d in reversed(matches)]

    candidates += _uv_interpreters(version)

    found = shutil.which(f"python{version}")
    if found:
        candidates.append(Path(found))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def probe(
    reqfile: ReqFile,
    *,
    config: dict,
    validator: Validator,
    spec: EnvSpec | None = None,
    env_path: str | Path = "./aura_probe_venv",
    record_baseline: bool = False,
) -> ProbeResult:
    """Install a candidate environment and judge it. The one path all tiers use.

    ``record_baseline=True`` stores the result as the reference against which
    later candidates are judged (which tests already fail, what numbers to expect).
    """
    result = ProbeResult()
    started = time.time()
    workdir = Path(env_path).parent

    with Environment(env_path, spec) as env:
        # 1. Dependencies
        output, code = env.install_requirements(reqfile, workdir)
        if code != 0:
            result.stderr = output[-20000:]
            result.error_class = (ErrorClass.TIMEOUT if code == 124
                                  else classify_error(output))
            result.duration_s = time.time() - started
            return result
        result.install_ok = True

        # 2. The subject project itself, when it is an installable package
        if config.get("IS_INSTALLABLE_PACKAGE", False):
            project_dir = config.get("VALIDATION_CONFIG", {}).get("project_dir") or "."
            output, code = env.install_project(
                project_dir, config.get("PROJECT_EXTRAS", ""),
                editable=config.get("EDITABLE_INSTALL", False),
            )
            if code != 0:
                result.build_ok = False
                result.stderr = output[-20000:]
                result.error_class = ErrorClass.PROJECT_BUILD
                result.duration_s = time.time() - started
                return result

        # 3. Fixtures the suite needs that pip cannot install
        setup = config.get("ENVIRONMENT_SETUP") or []
        for command in setup:
            rendered = command.replace("{python}", env.python)
            out, code = env.run_setup(rendered, cwd=config.get(
                "VALIDATION_CONFIG", {}).get("project_dir"))
            if code != 0:
                result.stderr = out[-20000:]
                result.error_class = ErrorClass.OTHER
                result.setup_failed = True
                result.duration_s = time.time() - started
                return result

        # 4. Validation
        validation = (validator.record_baseline(env.python) if record_baseline
                      else validator.run(env.python))
        result.validation = validation
        result.frozen = env.freeze()

        if record_baseline:
            # A baseline does not have to be green, only *runnable*. The judge
            # rejects a candidate for tests that newly fail, so a suite with
            # pre-existing failures still constrains an upgrade through the
            # tests that do pass -- and demanding a perfect suite discards the
            # many real projects that ship one or two broken tests. What is not
            # acceptable is a suite that never ran: nothing can be measured
            # against a baseline of no evidence.
            ran = (validation.counts or {}).get("passed", 0) > 0
            # Not "started": that name is this probe's timer, and reusing it for
            # a flag made the duration below time.time() minus a bool -- a Unix
            # timestamp, recorded as the baseline probe's install time.
            suite_started = "could not start" not in (validation.reason or "")
            result.accepted = validation.passed or (ran and suite_started)
            if not result.accepted:
                result.error_class = ErrorClass.VALIDATION_FAILED
                result.stderr = validation.output[-20000:]
        else:
            accepted, new_failures, drifted, detail = validator.judge(validation)
            result.accepted = accepted
            result.new_failures = new_failures
            result.drifted_metrics = drifted
            result.metric_detail = detail
            if not accepted:
                result.error_class = (ErrorClass.VALIDATION_DRIFT if drifted and not new_failures
                                      else ErrorClass.VALIDATION_FAILED)
                result.stderr = validation.output[-20000:]

    result.duration_s = time.time() - started
    return result


@contextmanager
def dry_run_session(
    *, spec: EnvSpec | None = None,
    env_path: str | Path = "./aura_dryrun_venv", workdir: Path | None = None,
) -> Iterator[Callable[[ReqFile], tuple[bool, str]]]:
    """One reusable resolver sandbox for a whole scan.

    Yields ``resolve(reqfile) -> (resolvable, log)``. The scan asks this
    question once per candidate version, and ``--dry-run`` installs nothing, so
    a single virtualenv serves every question. Building one per candidate --
    which is what the first version of this code did -- spends a ``venv.create``
    plus a pip/setuptools/wheel bootstrap before each resolve, tens of seconds of
    pure overhead multiplied by the length of the version line.

    If the sandbox cannot be built at all, the session degrades to admitting
    every candidate: the pre-filter is an optimisation, and losing it must cost
    time, not answers.
    """
    spec = spec or EnvSpec()
    workdir = Path(workdir or Path(env_path).parent)
    try:
        env_cm = Environment(env_path, spec)
        env = env_cm.__enter__()
    except Exception as exc:  # sandbox unavailable -> probe everything
        # Bound now: Python clears ``exc`` when the except block ends, and the
        # lambda runs later.
        reason = f"dry-run sandbox unavailable: {exc}"
        yield lambda reqfile: (True, reason)
        return
    try:
        yield lambda reqfile: env.dry_run(reqfile, workdir)
    finally:
        env_cm.__exit__(None, None, None)


def dry_run_resolves(
    reqfile: ReqFile, *, spec: EnvSpec | None = None,
    env_path: str | Path = "./aura_dryrun_venv", workdir: Path | None = None,
) -> tuple[bool, str]:
    """One-shot form of :func:`dry_run_session`, for a single question."""
    with dry_run_session(spec=spec, env_path=env_path, workdir=workdir) as resolve:
        return resolve(reqfile)
