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


def pip_blocker(bin_dir: Path, block_pip: bool = True) -> Path:
    """A directory with ``pip`` shims that refuse to run, put first on PATH.
    ``block_pip=False`` keeps only the process-killing shims (agent_alone)."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("pip", "pip3", "uv", "conda") if block_pip else ():
        shim = bin_dir / name
        shim.write_text("#!/bin/sh\necho 'Installing or changing packages is not allowed "
                        "in this task: change the code instead.' >&2\nexit 1\n")
        shim.chmod(0o755)
    # Stopping processes by name can reach past the agent: in one run Codex's
    # `pkill -f` took down the STRIDE process running the job. `kill <pid>` of
    # the agent's own processes still works.
    for name in ("pkill", "killall"):
        shim = bin_dir / name
        shim.write_text("#!/bin/sh\necho 'Stopping processes by name is not allowed here: "
                        "use kill <pid> on a process you started, or run commands with "
                        "timeout.' >&2\nexit 1\n")
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


# ------------------------------------------------------------- test setup
#
# Agents may change how tests are set up (imports, fixtures, mocks, conftest,
# version checks) but not what tests check. The rule is checked on the test
# files' syntax trees, before and after an attempt:
#   - no test file deleted, no test function removed or renamed;
#   - every assert a test function had is still there, word for word
#     (pytest.raises / assertRaises blocks count as asserts);
#   - no skip or xfail added (decorator or call);
#   - no test selection added to the pytest configuration.

_SKIPS = ("skip", "skipif", "xfail", "importorskip")
_SELECT = re.compile(r"(--ignore|--deselect|(^|\s)-k\s|(^|\s)-m\s|--lf\b|-p\s+no:)")


def _is_test_func(name: str) -> bool:
    return name.startswith("test")


def _checks(node) -> list[str]:
    """The checks a test function makes, as source text."""
    import ast
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert):
            out.append(ast.unparse(sub))
        elif isinstance(sub, ast.Call):
            fn = ast.unparse(sub.func)
            if fn.endswith(("raises", "warns")) or fn.split(".")[-1].startswith("assert"):
                out.append(ast.unparse(sub))
    return out


def _marks(node) -> list[str]:
    import ast
    found = []
    for dec in getattr(node, "decorator_list", []):
        text = ast.unparse(dec)
        if any(s in text for s in _SKIPS):
            found.append(text)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            text = ast.unparse(sub.func)
            if text.split(".")[-1] in _SKIPS or text.endswith("SkipTest"):
                found.append(text)
    return found


def test_shape(root: Path, files: list[str]) -> dict:
    """{file: {test: {"checks": [...], "marks": [...]}}} for the test files,
    plus the pytest configuration's text."""
    import ast
    shape: dict = {}
    for rel in files:
        path = root / rel
        if not path.is_file():
            shape[rel] = None
            continue
        if path.suffix != ".py":
            if path.name in CONFIG_FILES:
                shape[rel] = {"<config>": path.read_text(errors="replace")}
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            shape[rel] = {"<syntax error>": None}
            continue
        tests = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and _is_test_func(node.name):
                tests[node.name] = {"checks": _checks(node), "marks": _marks(node)}
            elif isinstance(node, ast.ClassDef):
                cls_marks = _marks(ast.ClassDef(name=node.name, bases=[], keywords=[],
                                                body=[], decorator_list=node.decorator_list))
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and _is_test_func(sub.name):
                        tests[f"{node.name}.{sub.name}"] = {
                            "checks": _checks(sub), "marks": _marks(sub) + cls_marks}
        shape[rel] = tests
    return shape


def weakened_tests(before: dict, after: dict) -> list[str]:
    """What an attempt did to the tests that the rule forbids."""
    from collections import Counter
    problems = []
    for rel, tests in before.items():
        now = after.get(rel)
        if tests is None:
            continue
        if now is None:
            problems.append(f"{rel}: deleted")
            continue
        if "<config>" in tests:
            old = set(tests["<config>"].splitlines())
            added = [l for l in now.get("<config>", "").splitlines() if l not in old]
            problems += [f"{rel}: test selection added ({l.strip()})"
                         for l in added if _SELECT.search(l)]
            continue
        if "<syntax error>" in now:
            problems.append(f"{rel}: no longer parses")
            continue
        for name, info in tests.items():
            if name not in now:
                problems.append(f"{rel}::{name}: removed")
                continue
            missing = Counter(info["checks"]) - Counter(now[name]["checks"])
            problems += [f"{rel}::{name}: check changed or removed ({c[:80]})"
                         for c in missing]
            if len(now[name]["marks"]) > len(info["marks"]):
                problems.append(f"{rel}::{name}: skip/xfail added")
    return sorted(set(problems))


def test_aware_lines(diff: str) -> list[str]:
    """Added source lines that name test doubles or the test run: code that
    behaves differently under the tests (``if isinstance(es, MockElasticsearch)``).
    Flagged for the report, not rejected."""
    hits = []
    current_is_test = False
    for line in diff.splitlines():
        if line.startswith("+++ "):
            name = line[4:].strip()
            current_is_test = any(p in ("tests", "test") for p in Path(name).parts) \
                or Path(name).name.startswith(("test_", "conftest"))
            continue
        if current_is_test or not line.startswith("+") or line.startswith("+++"):
            continue
        if re.search(r"\b\w*Mock\w*\b|\bmock\b|\bpytest\b|PYTEST_CURRENT_TEST|"
                     r"test environment|under test", line):
            hits.append(line[1:].strip()[:160])
    return hits
