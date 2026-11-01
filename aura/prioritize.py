"""Dependency prioritisation strategies.

The paper defines the Usage Risk Score as the *Architectural Spread*

    R(d) = sum over files f of  I(d in Imports(f))

where I is an indicator -- a package counts once per file that imports it, no
matter how many import statements that file uses. The previous implementation
incremented per import *statement* (so ``import torch`` + ``import torch.nn`` +
``from torch.utils.data import DataLoader`` in one file scored 3, not 1) and
then sorted primarily by version-jump size rather than by R(d) at all.

This module implements R(d) as specified and exposes the alternative orderings
as first-class strategies, so the ordering used by a run is an explicit,
recorded choice and the heuristics can be compared head to head.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from pathlib import Path

from packaging.version import Version

from aura.naming import build_mapping, canonical
from aura.reqfile import safe_parse_version

DEFAULT_EXCLUDES = {
    ".git", ".devenv", "venv", ".venv", "temp_venv", "final_venv",
    "bootstrap_venv", "initial_check_venv", "temp_pip_check", "temp_co_res_venv",
    "node_modules", "build", "dist", ".tox", ".eggs", "__pycache__", "site-packages",
}

# Modules that ship with CPython never map to a distribution we can upgrade.
_STDLIB = frozenset(getattr(__import__("sys"), "stdlib_module_names", frozenset()))


@dataclass
class Candidate:
    """One package considered for update in a pass."""

    name: str                      # canonical distribution name
    current: str
    target: str
    spread: int = 0                # R(d): files importing it
    import_statements: int = 0     # legacy metric, kept for comparison
    is_primary: bool = False       # declared in primary_requirements.txt
    fanout: int = 0                # how many other requirements depend on it
    rank_key: tuple = field(default=(), repr=False)

    @property
    def version_jump(self) -> int:
        """10 major / 5 minor / 1 patch -- the legacy `severity` score."""
        cur, tgt = safe_parse_version(self.current), safe_parse_version(self.target)
        if not isinstance(cur, Version) or not isinstance(tgt, Version):
            return 5
        if tgt.major > cur.major:
            return 10
        if tgt.minor > cur.minor:
            return 5
        return 1

    @property
    def releases_behind(self) -> int:
        """Filled in by the telemetry layer; see metrics.version_age."""
        return getattr(self, "_releases_behind", 0)


def _iter_python_files(repo_root: Path, excludes: set[str]) -> list[Path]:
    out = []
    for path in repo_root.rglob("*.py"):
        # .aura_*: the harness's own probe venvs and per-probe project copies,
        # which would otherwise count every import once per slot.
        if any(part in excludes or part.startswith(".aura_") for part in path.parts):
            continue
        out.append(path)
    return out


def analyze_imports(
    repo_root: str | Path, excludes: set[str] | None = None
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Scan the repo's Python sources.

    Returns ``(files_by_module, statements_by_module)`` where the first maps a
    top-level import name to the *set of files* importing it (the paper's
    indicator semantics) and the second counts raw import statements (the legacy
    metric, retained so the two can be compared).
    """
    root = Path(repo_root)
    excludes = (excludes or set()) | DEFAULT_EXCLUDES
    files_by_module: dict[str, set[str]] = {}
    statements_by_module: dict[str, int] = {}

    for py_file in _iter_python_files(root, excludes):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError, OSError):
            continue
        rel = str(py_file.relative_to(root))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import -- internal, not a dependency.
                if node.level and node.level > 0:
                    continue
                if node.module:
                    names = [node.module]
            for dotted in names:
                top = dotted.split(".")[0]
                if not top or top in _STDLIB:
                    continue
                files_by_module.setdefault(top, set()).add(rel)
                statements_by_module[top] = statements_by_module.get(top, 0) + 1

    return files_by_module, statements_by_module


def architectural_spread(
    repo_root: str | Path,
    python_executable: str | None = None,
    excludes: set[str] | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """R(d) keyed by canonical *distribution* name, plus the legacy statement count.

    Import names are translated to distribution names before aggregation, so
    ``cv2`` contributes to ``opencv-python`` rather than to a key nothing ever
    looks up.
    """
    files_by_module, statements_by_module = analyze_imports(repo_root, excludes)
    mapping = build_mapping(set(files_by_module), python_executable)

    spread: dict[str, set[str]] = {}
    statements: dict[str, int] = {}
    for module, files in files_by_module.items():
        dist = mapping[module]
        spread.setdefault(dist, set()).update(files)
        statements[dist] = statements.get(dist, 0) + statements_by_module.get(module, 0)

    return {d: len(f) for d, f in spread.items()}, statements


# --------------------------------------------------------------------- orderings

def _key_spread(c: Candidate) -> tuple:
    # Primary (directly declared) packages break ties above transitive ones.
    return (c.spread, c.is_primary, c.fanout, c.name)


def _key_legacy_version_jump(c: Candidate) -> tuple:
    """Reproduces the ordering the v1 code actually used (jump size first)."""
    return (c.version_jump * 10, c.import_statements + (10 if c.is_primary else 0), c.name)


def _key_import_statements(c: Candidate) -> tuple:
    return (c.import_statements, c.is_primary, c.name)


def _key_fanout(c: Candidate) -> tuple:
    """Coupling computed from the dependency graph, ignoring project usage."""
    return (c.fanout, c.is_primary, c.name)


def _key_alphabetical(c: Candidate) -> tuple:
    return (c.name,)


ORDERINGS = {
    "spread": (_key_spread, True),                    # paper's R(d)  [default]
    "version_jump": (_key_legacy_version_jump, True),  # v1 behaviour
    "import_statements": (_key_import_statements, True),
    "fanout": (_key_fanout, True),
    "alphabetical": (_key_alphabetical, False),
    "reverse_spread": (_key_spread, False),            # worst-first control
    "random": (None, False),                           # seeded; order sensitivity
}


def order_candidates(
    candidates: list[Candidate], strategy: str = "spread", seed: int = 0
) -> list[Candidate]:
    """Sort candidates by the named strategy. Deterministic for a given seed."""
    if strategy not in ORDERINGS:
        raise ValueError(f"unknown ordering {strategy!r}; expected one of {sorted(ORDERINGS)}")
    if strategy == "random":
        out = list(candidates)
        random.Random(seed).shuffle(out)
        return out
    key, reverse = ORDERINGS[strategy]
    ordered = sorted(candidates, key=key, reverse=reverse)
    for c in ordered:
        c.rank_key = key(c)
    return ordered
