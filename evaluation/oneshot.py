"""The one-shot LLM arm: ask the model for the whole upgraded file, once.

This is the obvious alternative to AURA that a reader will ask about: give an
open model the requirements file and let it propose every upgrade in a single
answer. No iteration, no test feedback, no resolver in the loop. The proposal
is installed and tested once, and it is kept only if the project's own tests
still pass -- the same all-or-nothing rule pip_upgrade is scored under.

Run against the same models as the LLM tier, it separates what the model knows
from what the search contributes. Where one-shot reaches as far as AURA, the
model's knowledge was enough. Where it fails, iterating against test results is
what did the work.

The model is shown which releases exist and install on the project's Python.
Without that list it answers from its training data, which is older than the
releases it would need to name, and the arm would measure recall instead of
reasoning about compatibility -- a strawman.
"""

from __future__ import annotations

import re

from packaging.utils import canonicalize_name

from aura.index import PackageIndex
from aura.reqfile import safe_parse_version

# How many of the newest usable releases each package is offered. The
# candidates shown are what the model can choose between, so the list has to
# reach back far enough to hold a compatible version, while a 59-package
# subject still has to fit the model's context.
VERSIONS_SHOWN = 8

PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._\-]*)(?:\[[^\]]*\])?\s*==\s*([^\s;#,]+)",
                 re.M)


def offered_versions(index: PackageIndex, name: str, current: str,
                     python_version: str) -> list[str]:
    """Releases at or above the current pin that install on this Python."""
    try:
        history = index.history(name)
    except Exception:
        return []
    if history is None:
        return []
    floor = safe_parse_version(current)
    usable = []
    for raw in history.versions:
        parsed = safe_parse_version(raw)
        if parsed is None or (floor is not None and parsed < floor):
            continue
        if python_version and not history.usable_on(raw, python_version):
            continue
        usable.append(raw)
    return usable[-VERSIONS_SHOWN:]


def build_prompt(pins: dict[str, str], offers: dict[str, list[str]],
                 python_version: str) -> str:
    """One prompt holding everything the model is allowed to know."""
    lines = []
    for name in sorted(pins):
        options = offers.get(name) or []
        shown = ", ".join(options) if options else "(no newer release installs)"
        lines.append(f"{name}=={pins[name]}    available: {shown}")
    return (
        "You are upgrading the dependencies of a Python project.\n"
        f"The project runs on Python {python_version}. Its tests must still pass "
        "after the upgrade.\n\n"
        "Each line is a package pinned at its current version, followed by the "
        "newer releases that install on this Python:\n\n"
        + "\n".join(lines) + "\n\n"
        "Choose, for every package, the newest version you believe is compatible "
        "with the others and with the project. Packages depend on each other, so "
        "choose them together. Keeping a package at its current version is "
        "allowed.\n\n"
        "You get exactly one attempt. Reply with the complete requirements file, "
        "one line per package, each exactly in the form name==version, and "
        "nothing else."
    )


def parse_proposal(text: str, known: set[str]) -> tuple[dict[str, str], list[str]]:
    """The pins the model proposed, and the names it invented.

    A package the model names that the project does not declare is left out:
    the arm upgrades this project's dependencies, it does not add new ones. A
    declared package the model leaves out keeps its current pin.
    """
    proposed: dict[str, str] = {}
    invented: list[str] = []
    for match in PIN.finditer(text or ""):
        name = canonicalize_name(match.group(1))
        version = match.group(2).strip()
        if name in known:
            proposed[name] = version
        else:
            invented.append(f"{name}=={version}")
    return proposed, invented


def nonexistent(index: PackageIndex, proposed: dict[str, str]) -> list[str]:
    """Proposed versions that were never released: the model's hallucinations."""
    missing = []
    for name, version in proposed.items():
        try:
            history = index.history(name)
        except Exception:
            continue
        if history is not None and not history.contains(version):
            missing.append(f"{name}=={version}")
    return missing
