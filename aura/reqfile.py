"""Fidelity-preserving requirements.txt parsing and rewriting.

The previous implementation located package names with ``re.match(r'([a-zA-Z0-9\\-_]+)')``
and read versions with ``line.split('==')[1]``. That silently mangles extras
(``requests[security]``), dotted names (``ruamel.yaml``), environment markers,
ranges, URLs and inline comments -- and because the update loop writes the file
back out, the corruption propagates into every downstream result.

This module parses with ``packaging`` (the same library pip uses) and guarantees
that a line nobody asked to change is emitted byte-for-byte as it was read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from packaging.version import parse as parse_version

# A pip option line: -r other.txt, -e ., --index-url ..., -c constraints.txt
_OPTION_RE = re.compile(r"^\s*-")
# Splits a trailing inline comment off a requirement line. PEP 508 requires the
# ``#`` to be preceded by whitespace to count as a comment.
_INLINE_COMMENT_RE = re.compile(r"^(?P<body>.*?)(?P<comment>\s+#.*)$", re.DOTALL)

# Direct references (``pkg @ https://...``) and bare VCS/URL lines are preserved
# verbatim; we never attempt to re-pin them.
_URLISH_RE = re.compile(r"^\s*(?:[a-zA-Z][a-zA-Z0-9+.-]*)\+?://|^\s*git\+|@\s*(?:https?|git|file)")


class LineKind:
    BLANK = "blank"
    COMMENT = "comment"
    OPTION = "option"
    REQUIREMENT = "requirement"
    URL = "url"
    UNPARSEABLE = "unparseable"


@dataclass
class ReqLine:
    """One physical line of a requirements file.

    ``raw`` is the exact text as read (without the trailing newline). It is what
    gets written back unless an explicit rewrite has been applied, which is what
    makes round-tripping lossless.
    """

    raw: str
    kind: str
    requirement: Requirement | None = None
    inline_comment: str = ""
    _override: str | None = field(default=None, repr=False)

    @property
    def name(self) -> str | None:
        """Canonical (PEP 503) distribution name, or None for non-requirements."""
        if self.requirement is None:
            return None
        return canonicalize_name(self.requirement.name)

    @property
    def pinned_version(self) -> str | None:
        """The version if this line is pinned with a single ``==``, else None."""
        if self.requirement is None:
            return None
        specs = list(self.requirement.specifier)
        if len(specs) == 1 and specs[0].operator in ("==", "==="):
            return specs[0].version
        return None

    def render(self) -> str:
        return self._override if self._override is not None else self.raw

    def repin(self, version: str) -> None:
        """Rewrite this line to ``==version``, preserving everything else.

        Extras, environment markers and the inline comment are carried across.
        The original casing of the project name is kept so diffs stay readable.
        """
        if self.requirement is None:
            raise ValueError(f"cannot repin a {self.kind} line: {self.raw!r}")
        req = self.requirement
        extras = f"[{','.join(sorted(req.extras))}]" if req.extras else ""
        marker = f" ; {req.marker}" if req.marker else ""
        indent = self.raw[: len(self.raw) - len(self.raw.lstrip())]
        self._override = f"{indent}{req.name}{extras}=={version}{marker}{self.inline_comment}"
        # Re-parse so the in-memory Requirement agrees with the rendered text.
        # Keeping only the rendered form would make ``pins()`` report the stale
        # version, and the update loop would re-propose the same change forever.
        self.requirement = Requirement(f"{req.name}{extras}=={version}"
                                       + (f"; {req.marker}" if req.marker else ""))

    def reset(self) -> None:
        """Discard any rewrite, restoring both the text and the parsed form."""
        self._override = None
        self.requirement = _parse_line(self.raw).requirement


def _parse_line(raw: str) -> ReqLine:
    stripped = raw.strip()
    if not stripped:
        return ReqLine(raw=raw, kind=LineKind.BLANK)
    if stripped.startswith("#"):
        return ReqLine(raw=raw, kind=LineKind.COMMENT)
    if _OPTION_RE.match(raw):
        return ReqLine(raw=raw, kind=LineKind.OPTION)

    body, comment = raw, ""
    m = _INLINE_COMMENT_RE.match(raw)
    if m:
        body, comment = m.group("body"), m.group("comment")

    if _URLISH_RE.search(body.strip()) and "@" not in body.split("://")[0]:
        # A bare URL/VCS line with no project name to re-pin.
        return ReqLine(raw=raw, kind=LineKind.URL, inline_comment=comment)

    try:
        req = Requirement(body.strip())
    except InvalidRequirement:
        return ReqLine(raw=raw, kind=LineKind.UNPARSEABLE, inline_comment=comment)

    # ``pkg @ https://...`` parses fine but must never be re-pinned.
    if req.url:
        return ReqLine(raw=raw, kind=LineKind.URL, requirement=req, inline_comment=comment)
    return ReqLine(raw=raw, kind=LineKind.REQUIREMENT, requirement=req, inline_comment=comment)


class ReqFile:
    """A parsed requirements file that rewrites only what you ask it to."""

    def __init__(self, lines: list[ReqLine], trailing_newline: bool = True):
        self.lines = lines
        self.trailing_newline = trailing_newline

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_text(cls, text: str) -> "ReqFile":
        trailing_newline = text.endswith("\n")
        body = text[:-1] if trailing_newline else text
        raw_lines = body.split("\n") if body else []
        lines = [_parse_line(l) for l in raw_lines]
        # "pkg-resources==0.0.0" is not a dependency. pip freeze inside a
        # virtualenv on Debian and Ubuntu writes it for setuptools' own
        # pkg_resources module, and no release of that name exists, so any
        # file that carries it cannot be installed as written -- which is how
        # ElectionAuditWare/audit-conductor, 63 tests and a 2019 pin set,
        # failed screening. Dropping it changes nothing that installs.
        lines = [l for l in lines
                 if not (l.name == "pkg-resources" and l.pinned_version == "0.0.0")]
        return cls(lines, trailing_newline)

    @classmethod
    def load(cls, path: str | Path) -> "ReqFile":
        return cls.from_text(Path(path).read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- writing

    def to_text(self) -> str:
        out = "\n".join(l.render() for l in self.lines)
        return out + "\n" if self.trailing_newline else out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_text(), encoding="utf-8")

    # ------------------------------------------------------------- inspection

    def requirements(self) -> Iterator[ReqLine]:
        for l in self.lines:
            if l.kind == LineKind.REQUIREMENT:
                yield l

    def pins(self) -> dict[str, str]:
        """Canonical name -> pinned version, for lines pinned with a single ``==``."""
        out: dict[str, str] = {}
        for l in self.requirements():
            v = l.pinned_version
            if v is not None and l.name:
                out[l.name] = v
        return out

    def names(self) -> set[str]:
        return {l.name for l in self.requirements() if l.name}

    def find(self, name: str) -> ReqLine | None:
        target = canonicalize_name(name)
        for l in self.requirements():
            if l.name == target:
                return l
        return None

    def is_fully_pinned(self) -> bool:
        reqs = list(self.requirements())
        if not reqs:
            return False
        return all(l.pinned_version is not None for l in reqs)

    # -------------------------------------------------------------- mutation

    def repin(self, name: str, version: str) -> bool:
        """Pin ``name`` to ``version``. Returns False if the package isn't present."""
        line = self.find(name)
        if line is None:
            return False
        line.repin(version)
        return True

    def repin_all(self, versions: dict[str, str]) -> list[str]:
        """Apply many pins. Returns the canonical names that were not found."""
        missing = []
        for name, version in versions.items():
            if not self.repin(name, version):
                missing.append(canonicalize_name(name))
        return missing

    def with_pins(self, versions: dict[str, str]) -> "ReqFile":
        """A copy of this file with ``versions`` applied -- the original is untouched."""
        clone = ReqFile.from_text(self.to_text())
        clone.repin_all(versions)
        return clone

    def with_added_pins(self, versions: dict[str, str]) -> "ReqFile":
        """A copy pinning ``versions``, appending lines for packages not present.

        Reconstructing a past environment means pinning dependencies the file
        never mentioned -- the transitive ones, which is exactly where the drift
        that breaks an old project lives -- so these have to be added rather
        than repinned.
        """
        clone = ReqFile.from_text(self.to_text())
        wanted = {canonicalize_name(n): v for n, v in versions.items()}
        missing = clone.repin_all(wanted)
        if not missing:
            return clone
        text = clone.to_text()
        if text and not text.endswith("\n"):
            text += "\n"
        text += "".join(f"{name}=={wanted[name]}\n" for name in sorted(missing))
        return ReqFile.from_text(text)

    def without_pins(self, names) -> "ReqFile":
        """A copy with these packages' requirement lines removed entirely."""
        drop = {canonicalize_name(n) for n in names}
        kept = [l.render() for l in self.lines
                if not (l.kind == LineKind.REQUIREMENT and l.name in drop)]
        return ReqFile.from_text("\n".join(kept) + "\n")

    def drop_all_constraints(self) -> None:
        """Strip version specifiers, keeping names, extras and markers.

        Used by the bootstrap repair path when a pinned baseline will not install.
        """
        for l in self.requirements():
            req = l.requirement
            assert req is not None
            extras = f"[{','.join(sorted(req.extras))}]" if req.extras else ""
            marker = f" ; {req.marker}" if req.marker else ""
            indent = l.raw[: len(l.raw) - len(l.raw.lstrip())]
            l._override = f"{indent}{req.name}{extras}{marker}{l.inline_comment}"
            # Keep the parsed form in step with the text, as repin() does.
            l.requirement = Requirement(
                f"{req.name}{extras}" + (f"; {req.marker}" if req.marker else "")
            )


def parse_freeze(freeze_output: str) -> dict[str, str]:
    """Turn ``pip freeze`` output into canonical name -> version."""
    out: dict[str, str] = {}
    for raw in freeze_output.splitlines():
        line = _parse_line(raw)
        v = line.pinned_version
        if v is not None and line.name:
            out[line.name] = v
    return out


def freeze_to_reqfile(freeze_output: str) -> ReqFile:
    """Build a sorted, pruned lockfile from ``pip freeze`` output."""
    editable, pinned = [], []
    for raw in freeze_output.splitlines():
        line = _parse_line(raw)
        if line.kind == LineKind.OPTION and raw.strip().startswith("-e"):
            editable.append(raw.strip())
        elif line.pinned_version is not None:
            pinned.append(raw.strip())
    body = "\n".join(sorted(editable) + sorted(pinned, key=str.lower))
    return ReqFile.from_text(body + "\n" if body else "")


def safe_parse_version(value: str) -> Version | None:
    """``parse_version`` that returns None instead of raising on junk."""
    try:
        return parse_version(value)
    except (InvalidVersion, TypeError):
        return None
