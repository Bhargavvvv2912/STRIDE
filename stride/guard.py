"""Keep the code agent honest: tests and requirements are read-only.

RepoRescue found agents passing suites by editing tests (38-53% of some
systems' successes). STRIDE blocks that during the run (files made read-only)
and checks it afterwards (hashes, skip markers, installed packages).
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_FILES = ("pytest.ini", "setup.cfg", "tox.ini", "pyproject.toml", "conftest.py",
                "noxfile.py")
SKIP_PATTERN = re.compile(
    r"^\+.*(pytest\.skip|pytest\.mark\.skip|pytest\.mark\.xfail|pytest\.xfail|"
    r"unittest\.skip|@skip|importorskip)", re.MULTILINE)


def protected_files(root: Path, test_targets: list[str], requirements: list[str]) -> list[Path]:
    """Test files, test config and requirements files, relative to ``root``."""
    files: set[Path] = set()
    for target in test_targets:
        path = root / target.split("::")[0]
        if path.is_dir():
            files.update(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            files.add(path)
    for pattern in ("test_*.py", "*_test.py", "conftest.py"):
        files.update(p for p in root.rglob(pattern) if ".git" not in p.parts
                     and "site-packages" not in p.parts)
    for name in CONFIG_FILES:
        if (root / name).is_file():
            files.add(root / name)
    for req in requirements:
        if (root / req).is_file():
            files.add(root / req)
    return sorted(p for p in files if not any(part.startswith(".aura") or part.startswith(
        ".stride") for part in p.relative_to(root).parts))


def digest(paths: list[Path]) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


@dataclass
class Lock:
    root: Path
    paths: list[Path]
    hashes: dict[str, str] = field(default_factory=dict)
    modes: dict[str, int] = field(default_factory=dict)

    def __enter__(self) -> "Lock":
        self.hashes = digest(self.paths)
        for p in self.paths:
            self.modes[str(p)] = p.stat().st_mode
            p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return self

    def __exit__(self, *exc) -> None:
        for p in self.paths:
            if p.exists():
                p.chmod(self.modes.get(str(p), 0o644))

    def tampered(self) -> list[str]:
        """Protected files that changed, disappeared, or were added beside them."""
        now = digest(self.paths)
        changed = [str(Path(k).relative_to(self.root)) for k, v in self.hashes.items()
                   if now.get(k) != v]
        return sorted(changed)


def restore(root: Path, files: list[str]) -> None:
    """Put protected files back as committed, after a rejected attempt."""
    if files:
        subprocess.run(["git", "checkout", "--", *files], cwd=root,
                       capture_output=True, text=True)


def added_skips(diff: str) -> list[str]:
    return SKIP_PATTERN.findall(diff)


def pip_blocker(bin_dir: Path) -> Path:
    """A directory with ``pip`` shims that refuse to run, put first on PATH."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("pip", "pip3", "uv", "conda"):
        shim = bin_dir / name
        shim.write_text("#!/bin/sh\necho 'Installing or changing packages is not allowed "
                        "in this task: change the code instead.' >&2\nexit 1\n")
        shim.chmod(0o755)
    return bin_dir


def agent_environment(venv_bin: str, shim_dir: Path) -> dict[str, str]:
    """Environment variables for the agent's shell: the subject's Python first,
    package installation blocked."""
    path = os.pathsep.join([str(shim_dir), venv_bin, os.environ.get("PATH", "")])
    return {
        "PATH": path,
        "VIRTUAL_ENV": str(Path(venv_bin).parent),
        "PIP_NO_INDEX": "1",
        "PIP_FIND_LINKS": "",
        "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R",
        "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
