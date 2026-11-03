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
TEST_SUFFIXES = {".py", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".json", ".txt"}
SKIP_PATTERN = re.compile(
    r"^\+.*(pytest\.skip|pytest\.mark\.skip|pytest\.mark\.xfail|pytest\.xfail|"
    r"unittest\.skip|@skip|importorskip)", re.MULTILINE)


def protected_files(root: Path, test_targets: list[str], requirements: list[str]) -> list[Path]:
    """Test files, test config and requirements files, relative to ``root``."""
    files: set[Path] = set()
    for target in test_targets:
        path = root / target.split("::")[0]
        if path.is_dir():
            # Test code and config only. Suites write data, caches and outputs
            # into their own folders; those changing is not tampering.
            files.update(p for p in path.rglob("*") if p.is_file()
                         and p.suffix in TEST_SUFFIXES and "__pycache__" not in p.parts)
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
    # Only files the project committed: anything a test run created (datasets,
    # caches, outputs) is untracked, so it can never count as tampering.
    tracked = set(subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                                 text=True).stdout.splitlines())
    return sorted(p for p in files
                  if (not tracked or str(p.relative_to(root)) in tracked)
                  and not any(part.startswith(".aura") or part.startswith(".stride")
                              for part in p.relative_to(root).parts))


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
        # Read-only files are not enough: `sed -i` writes a new file and renames
        # it over the old one, which only needs the folder to be writable. So
        # the folders holding test files are locked too (never the project root,
        # where the agent must edit source).
        self.dirs = sorted({p.parent for p in self.paths if p.parent != self.root},
                           key=lambda d: len(d.parts), reverse=True)
        for d in self.dirs:
            self.modes[str(d)] = d.stat().st_mode
            d.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
                    | stat.S_IROTH | stat.S_IXOTH)
        return self

    def __exit__(self, *exc) -> None:
        for d in reversed(getattr(self, "dirs", [])):
            if d.exists():
                d.chmod(self.modes.get(str(d), 0o755))
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


# ------------------------------------------------------------ code structure

def code_shape(root: Path, protected: list[str]) -> dict[str, list[str] | None]:
    """The project's source outline: every top-level function, and every class
    with its methods, in tracked .py files outside the tests.

    Keys look like ``pkg/mod.py:Class`` (value: sorted method names) or
    ``pkg/mod.py:func()`` (value: None). A file that no longer parses maps
    to ``{"pkg/mod.py": ["<syntax error>"]}``.
    """
    import ast
    skip = set(protected)
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=root, capture_output=True,
                           text=True).stdout.splitlines()
    shape: dict[str, list[str] | None] = {}
    for rel in files:
        parts = Path(rel).parts
        if rel in skip or any(p in ("tests", "test") or p.startswith((".aura", ".stride"))
                              for p in parts) or Path(rel).name.startswith("test_"):
            continue
        path = root / rel
        if not path.is_file():
            shape[rel] = ["<deleted>"]
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            shape[rel] = ["<syntax error>"]
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                shape[f"{rel}:{node.name}"] = sorted(
                    n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                shape[f"{rel}:{node.name}()"] = None
    return shape


def lost_symbols(before: dict, after: dict) -> list[str]:
    """Functions, classes and methods the change removed (or broke the file of)."""
    lost = []
    for key, methods in before.items():
        if key not in after:
            file = key.split(":")[0]
            if after.get(file) in (["<syntax error>"], ["<deleted>"]):
                lost.append(f"{file} ({after[file][0].strip('<>')})")
            else:
                lost.append(key)
        elif methods is not None and after[key] is not None:
            lost += [f"{key}.{m}" for m in methods if m not in after[key]]
    return sorted(set(lost))


def added_symbols(before: dict, after: dict) -> list[str]:
    """Top-level functions and classes the change introduced."""
    return sorted(k for k in after if k not in before and ":" in k)
