"""Establishing a working baseline, including for projects that no longer have one.

Most published Python projects cannot be installed from their own requirements
file any more. Usually nothing is wrong with the *pins*: the file pins some
dependencies and leaves the rest open, and the open ones have moved on. A
project pinning ``Flask==1.1.1`` without pinning Jinja2 gets Jinja2 3.x today,
which removed ``escape``, and the project dies on import. Its environment worked
perfectly in 2020.

Refusing those projects would restrict this work to the small minority whose
requirements happen to be a full freeze. Repairing them arbitrarily would be
worse: the repair would silently *be* part of the modernization, and a repair
that differed between arms would make their results incomparable.

So the repair here is deterministic and model-free. Every arm, every model and
every rerun computes the same baseline from the same inputs, which is what makes
the comparison between arms mean anything:

  declared        the project's own file, unchanged. Tried first, always.
  reconstructed   every unpinned package in the resolved closure pinned to the
                  newest release published on or before the project's own commit
                  date. This reproduces the environment the author had rather
                  than one we chose.
  relaxed         the same, minus pins on packages that only run the tests
                  (coverage, tox, linters), which routinely fail to build on a
                  modern interpreter while contributing nothing to whether the
                  project works.

The first stage whose environment installs *and passes the project's own tests*
is the baseline. If none does, the subject has no working baseline and is
rejected -- exactly as before this module existed.

What the repair moved is recorded separately from what the search later
advances. Otherwise reconstruction would inflate every arm's score by the same
amount and "modernized 18 packages" would stop meaning what it says.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name

from aura.reqfile import ReqFile, safe_parse_version

# Tooling that runs *around* a test suite rather than being part of the project:
# old pins of these fail to build on modern interpreters while contributing
# nothing to whether the project's own code works.
TEST_TOOLING = frozenset({
    "coverage", "pytest-cov", "codecov", "coveralls", "tox", "nose", "nose2",
    "flake8", "pylint", "black", "isort", "mypy", "pre-commit", "twine",
    "bumpversion", "sphinx", "wheel", "setuptools", "pip",
})


# Built by pip inside an isolated environment, never declared by the project,
# and versioned like anything else: an old package's setup.py can use an API
# today's setuptools has removed.
BUILD_TOOLS = ("setuptools", "wheel")

# How far back the era search may go. setuptools 44.1.1 is the last release
# that still provides what later versions removed (Feature went in 46, use_2to3
# in 58) while remaining installable on a current interpreter; going by date
# alone reaches setuptools 3.3 for a project pinning a 2014 release, which no
# modern Python can use. Verified against the projects that fail this way.
LEGACY_FLOOR = {"setuptools": "44.1.1", "wheel": "0.34.2"}


# The rungs after the declared file, in order. The date-exact environment
# comes first, so a subject it works for is never moved off it. The wheel rungs
# follow for the one it cannot build: a project whose commit predates the
# interpreter, where the release of that date has no wheel for it and pip would
# compile an old C extension. Those step each such package to its nearest
# release that has one.
LADDER = (
    ("reconstructed", frozenset(), False),
    ("relaxed", TEST_TOOLING, False),
    ("reconstructed_wheels", frozenset(), True),
    ("relaxed_wheels", TEST_TOOLING, True),
)


def era_build_tools(index, pins: dict, on_date: str = "",
                    python_version: str = "") -> tuple[str, ...]:
    """The setuptools and wheel of the pinned environment's own era.

    Dated from the *pins*, not from the commit: a project committed in 2021 can
    pin releases from 2017 (dokuztas pins itsdangerous 0.24 and click 6.7), and
    those are what have to build. The oldest pin is what sets the era, because
    it is the one that will fail against a newer setuptools.
    """
    dates = []
    for name, version in (pins or {}).items():
        try:
            when = index.upload_dates(name).get(version)
        except Exception:
            when = None
        if when:
            dates.append(when)
    era = min(dates) if dates else on_date
    if not era:
        return ()
    out = []
    for name in BUILD_TOOLS:
        try:
            version = index.newest_at(name, era, python_version or None)
        except Exception:
            version = None
        floor = LEGACY_FLOOR.get(name)
        if version and floor:
            chosen, dated = safe_parse_version(floor), safe_parse_version(version)
            if chosen and dated and dated < chosen:
                version = floor
        if version:
            out.append(f"{name}=={version}")
    return tuple(out)


def build_constraints(index, on_date: str, path, python_version: str = "") -> str:
    """Write the build toolchain as it was on ``on_date``; return the path.

    pip builds source distributions in an isolated environment containing the
    newest setuptools and wheel, so a package that built perfectly in 2019 can
    fail now with ``cannot import name 'Feature' from 'setuptools'`` -- a fact
    about this machine's era, not about the package. pip applies PIP_CONSTRAINT
    to that isolated environment too, so constraining it there reconstructs the
    toolchain the same way the requirements themselves are reconstructed.

    Returns "" when no date is known or the index cannot date the tools, which
    leaves pip's own defaults in place.
    """
    if not on_date:
        return ""
    pins = []
    for name in BUILD_TOOLS:
        try:
            version = index.newest_at(name, on_date, python_version or None)
        except Exception:
            version = None
        if version:
            pins.append(f"{name}=={version}")
    if not pins:
        return ""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pins) + "\n")
    return str(path)


@dataclass
class Baseline:
    """The state every arm of an experiment starts from."""

    ok: bool
    reqfile: ReqFile | None = None
    stage: str = "declared"                     # which rung of the ladder worked
    repaired: dict[str, list] = field(default_factory=dict)  # name -> [from, to]
    reason: str = ""

    @property
    def digest(self) -> str:
        """Fingerprint of the baseline file.

        Recorded in every run's log so "all arms started from the same state" is
        something a reader can check rather than take on trust.
        """
        if self.reqfile is None:
            return ""
        return hashlib.sha256(self.reqfile.to_text().encode()).hexdigest()[:16]


def resolve_ranges_at_date(reqfile: ReqFile, index, on_date: str,
                           python_version: str = "") -> dict[str, str]:
    """Pin every requirement the file states as a range, as of ``on_date``.

    A range says what was *allowed*, not what was installed: the environment is
    whatever pip resolved on the day, and today it resolves to something else
    entirely. Pinning the newest release that the project's own specifier
    allowed on its commit date recovers the environment it had, and makes a
    ranged project measurable at all -- the search works in versions, and a
    range names none.

    Only versions the project permitted are considered, so ``flask<2`` is never
    reconstructed as Flask 2: the cap is the project's decision, not ours.
    """
    resolved: dict[str, str] = {}
    for line in reqfile.requirements():
        if not line.name or line.pinned_version is not None:
            continue
        requirement = line.requirement
        specifier = str(requirement.specifier) if requirement is not None else ""
        if requirement is not None and requirement.marker is not None:
            try:
                if not requirement.marker.evaluate({"python_version": python_version,
                                                    "extra": ""}):
                    continue
            except Exception:
                continue
        chosen = index.newest_at(line.name, on_date, python_version or None,
                                 specifier=specifier or None)
        if chosen:
            resolved[line.name] = chosen
    return resolved


def closure_at_date(index, pins: dict[str, str], on_date: str,
                    python_version: str = "", limit: int = 200) -> dict[str, str]:
    """The environment these pins implied on ``on_date``, from declared metadata.

    pip can only report the closure of a set it can *resolve*, which excludes
    the projects that need reconstruction most: a file pinning A and leaving B
    open stops resolving at all once B moves somewhere incompatible with A. In
    2020 it resolved; today pip refuses before installing anything, so there is
    no report to read and the subject looks unusable.

    Walking Requires-Dist instead needs no installation and no resolver. Each
    dependency the file does not pin is taken at the version that was newest on
    that date, and its own dependencies are followed from there -- which is what
    the resolver would have produced at the time.

    Markers are honoured only for ``python_version``; an extra's dependencies
    are skipped, since nothing here requests extras.
    """
    from packaging.requirements import Requirement

    resolved = dict(pins)
    queue = list(pins.items())
    seen: set[tuple[str, str]] = set()
    while queue and len(resolved) < limit:
        name, version = queue.pop(0)
        if (name, version) in seen:
            continue
        seen.add((name, version))
        try:
            requires = index.requires_dist(name, version)
        except Exception:
            requires = None
        for raw in requires or []:
            try:
                req = Requirement(raw)
            except Exception:
                continue
            if req.marker is not None:
                try:
                    if not req.marker.evaluate({"python_version": python_version,
                                                "extra": ""}):
                        continue
                except Exception:
                    continue          # a marker we cannot evaluate is not a dependency
            dep = canonicalize_name(req.name)
            if dep in resolved:
                continue
            try:
                # Within the range the requiring package states: without it the
                # walk took idna 3.2 for a requests that allows only idna<3.
                chosen = index.newest_at(dep, on_date, python_version or None,
                                         specifier=str(req.specifier) or None)
            except Exception:
                chosen = None
            if not chosen:
                continue
            resolved[dep] = chosen
            queue.append((dep, chosen))
    return resolved


def reconstruct(reqfile: ReqFile, closure: dict[str, str], *, index,
                on_date: str, python_version: str = "",
                skip: frozenset[str] = frozenset(),
                prefer_wheels: bool = False) -> tuple[ReqFile, dict[str, list]]:
    """Pin every unpinned package in the closure as it was on ``on_date``.

    ``closure`` is what the requirement set actually installs (from pip's own
    resolution report), so this reaches the transitive dependencies that the
    file never mentions -- which is where the drift that breaks these projects
    lives.

    Packages the file already pins are left exactly as they are: the project's
    own decisions are the subject of the experiment, not something to repair.
    """
    declared = reqfile.pins()
    additions: dict[str, str] = {}
    moved: dict[str, list] = {}
    for name, installed in sorted(closure.items()):
        if name in declared or name in skip:
            continue
        historical = index.newest_at(name, on_date, python_version or None,
                                     prefer_wheels=prefer_wheels)
        if not historical:
            continue
        # Pinning it is the point, even where the version does not change: the
        # file left it open, and an open dependency is what stops these projects
        # from installing a second time. Only a difference counts as *moved*,
        # so "what the repair changed" stays an honest number.
        additions[name] = historical
        if historical != installed:
            moved[name] = [installed, historical]
    if not additions:
        return reqfile, {}

    # Each package above was dated back on its own. The set also has to hold
    # together: what one package requires of another is a range, and the
    # newest release on that date can fall outside it -- neural-backed-
    # decision-trees got idna 3.2 beside a pin that allows only idna<3, and
    # Deformetrica a keopscore its own pins exclude, so neither could install.
    # So each is chosen again, as the newest on that date within every range
    # the rest of the set states for it. A reconstruction that installed had
    # no such conflict, so its choices already sat inside those ranges and are
    # chosen again unchanged.
    ranges = _declared_ranges(index, {**declared, **additions}, python_version)
    for name in list(additions):
        spec = ranges.get(name)
        if not spec:
            continue
        try:
            within = index.newest_at(name, on_date, python_version or None,
                                     specifier=spec, prefer_wheels=prefer_wheels)
        except Exception:
            within = None
        if within is None:
            # Nothing that old satisfies the range: take what pip itself
            # resolved, which does, rather than a version known not to.
            installed = closure.get(name)
            try:
                if installed and SpecifierSet(spec).contains(installed):
                    within = installed
            except InvalidSpecifier:
                pass
        if within and within != additions[name]:
            additions[name] = within
            if within != closure.get(name):
                moved[name] = [closure.get(name), within]
            else:
                moved.pop(name, None)
    return reqfile.with_added_pins(additions), moved


def _declared_ranges(index, versions: dict[str, str],
                     python_version: str = "") -> dict[str, str]:
    """For each package, every range the others in ``versions`` require of it.

    Read from each package's Requires-Dist at the version it has here. Markers
    are honoured for ``python_version``; extras are not requested, so their
    requirements are skipped.
    """
    from packaging.requirements import Requirement

    out: dict[str, list[str]] = {}
    for name, version in versions.items():
        try:
            requires = index.requires_dist(name, version) or []
        except Exception:
            continue
        for raw in requires:
            try:
                req = Requirement(raw)
            except Exception:
                continue
            if req.marker is not None:
                try:
                    if not req.marker.evaluate({"python_version": python_version,
                                                "extra": ""}):
                        continue
                except Exception:
                    continue
            spec = str(req.specifier)
            if spec:
                out.setdefault(canonicalize_name(req.name), []).append(spec)
    return {name: ",".join(specs) for name, specs in out.items()}


def establish(reqfile: ReqFile, *, probe, closure_of, index, commit_date: str,
              python_version: str = "", repair: bool = True,
              on_stage=None, frozen: ReqFile | None = None) -> Baseline:
    """Walk the ladder until an environment installs and passes the tests.

    ``probe(reqfile) -> result`` installs and validates; ``closure_of(reqfile)``
    returns what that set would install. Both are injected so this logic is
    testable without a network or an install.

    ``frozen`` is the subject's baseline as the first arm established it. When
    given, it is the only candidate: every arm starts from exactly that set, and
    if it no longer passes the arm fails rather than rebuilding a different one.
    Walking the ladder per arm gave flask-dj two baselines -- arms run before
    and after a change to reconstruction -- and nba2k25-plp a reconstructed one
    in two one-shot arms whose declared run once failed.
    """
    def report(stage: str, ok: bool, detail: str = "") -> None:
        if on_stage:
            on_stage(stage=stage, ok=ok, detail=detail)

    if frozen is not None:
        result = probe(frozen)
        if result.accepted:
            report("frozen", True)
            return Baseline(True, frozen, "frozen")
        detail = result.error_class or "validation failed"
        report("frozen", False, detail)
        return Baseline(False, None, "frozen",
                        reason=f"the subject's frozen baseline no longer passes: {detail}")

    result = probe(reqfile)
    if result.accepted:
        report("declared", True)
        return Baseline(True, reqfile, "declared")
    first_error = result.error_class or "baseline failed"
    report("declared", False, first_error)

    if not repair or not commit_date:
        return Baseline(False, None, "declared",
                        reason=f"baseline does not install and validate: {first_error}")

    # A file may state ranges rather than pins; those have to become versions
    # before anything can be reconstructed from them.
    ranged = resolve_ranges_at_date(reqfile, index, commit_date, python_version)
    repaired: dict[str, list] = {}
    if ranged:
        specifiers = {line.name: str(line.requirement.specifier) or "(any)"
                      for line in reqfile.requirements()
                      if line.name in ranged and line.requirement is not None}
        reqfile = reqfile.with_pins(ranged)
        repaired = {name: [specifiers.get(name, ""), version]
                    for name, version in ranged.items()}
        report("ranges_pinned", True, f"{len(ranged)} range(s) pinned as of "
                                     f"{commit_date}")
        # A range-pinned file is a candidate in its own right: for a project
        # that states ranges and nothing else, this *is* the reconstruction.
        result = probe(reqfile)
        if result.accepted:
            report("reconstructed", True)
            return Baseline(True, reqfile, "reconstructed", repaired=repaired)
        report("reconstructed", False, result.error_class or "failed")

    closure = closure_of(reqfile)
    if not closure:
        # pip reports a closure only for a set it can resolve, which excludes
        # the projects most in need of repair: once an unpinned dependency
        # moves somewhere incompatible with a pinned one, resolution fails
        # before anything is installed. Declared metadata still says what the
        # environment was, so the closure is walked from there instead.
        closure = closure_at_date(index, reqfile.pins(), commit_date,
                                  python_version)
        report("closure_from_metadata", bool(closure),
               f"{len(closure)} package(s)" if closure
               else "the index could not describe this project's dependencies")
    if not closure:
        return Baseline(False, None, "declared",
                        reason=f"baseline failed ({first_error}) and its "
                               "dependency closure could not be established")

    tried: set[str] = set()
    for stage, skip, wheels in LADDER:
        candidate, moved = reconstruct(
            reqfile, closure, index=index, on_date=commit_date,
            python_version=python_version, skip=skip, prefer_wheels=wheels)
        if stage.startswith("relaxed"):
            # Drop the project's own pins on test tooling as well, since an
            # unbuildable linter is what stopped the previous rung.
            candidate = candidate.without_pins(
                [n for n in candidate.pins() if n in TEST_TOOLING])
        if candidate.to_text() == reqfile.to_text():
            report(stage, False, "nothing to reconstruct")
            continue
        if wheels and candidate.to_text() in tried:
            # A wheel rung that moved nothing is an earlier rung again: it
            # would install the same file to learn the same thing. The two
            # original rungs are left exactly as they were.
            continue
        tried.add(candidate.to_text())
        result = probe(candidate)
        if result.accepted:
            report(stage, True)
            return Baseline(True, candidate, stage, repaired={**repaired, **moved})
        report(stage, False, result.error_class or "failed")

    return Baseline(False, None, "relaxed",
                    reason=f"no working baseline: declared failed ({first_error}) "
                           "and reconstruction at the project's commit date did "
                           "not produce one either")


def frozen_baseline(path: str | Path | None) -> ReqFile | None:
    """The subject's frozen baseline, if one has been written."""
    if path and Path(path).exists():
        return ReqFile.load(Path(path))
    return None


def freeze_once(path: str | Path | None, reqfile: ReqFile) -> bool:
    """Write the subject's baseline unless an arm already did. Atomic: two arms
    finishing at once cannot overwrite each other; the first one wins."""
    if not path:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as fh:
            fh.write(reqfile.to_text())
        return True
    except FileExistsError:
        return False
