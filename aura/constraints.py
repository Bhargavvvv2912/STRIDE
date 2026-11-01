"""The declared-constraint graph.

v1 learned that torch and torchvision are coupled by *failing*: it probed torch
alone, regex-matched pip's prose for package names, and inferred the blocker
from the wreckage. That costs a probe per discovery and depends on error text
that changes between pip versions.

The coupling is already declared. torchvision 0.13.0's own metadata says
``torch==1.12.0``; 0.15.1 says ``torch==2.0.0``. Reading it turns three things
from guesswork into arithmetic:

* **Routing.** If every candidate above the current pin is forbidden by an
  installed package, a reverse linear scan is provably futile and the search
  should co-resolve instead. Without the declared constraint you cannot know
  that, and the scan walks the whole version line learning nothing.
* **Co-resolution.** Instead of bumping the blocker to latest and hoping, the
  map inverts: "which torchvision versions permit torch 2.0.0?"
* **Evidence.** The graph is ground truth to score the regex extractor against.

Two rules govern the whole module, because the cost of the two errors is not
symmetric. Wrongly declaring a scan futile *loses a solution*; wrongly declaring
it viable only costs probes. So whenever information is missing or a marker
cannot be evaluated, the constraint is treated as **not applying**. Absent
metadata means "no information", never "no constraints".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from aura.reqfile import safe_parse_version


def marker_environment(python_version: str = "") -> dict:
    """Marker evaluation context, optionally retargeted to another interpreter."""
    env = dict(default_environment())
    if python_version:
        env["python_version"] = python_version
        env["python_full_version"] = (
            python_version if python_version.count(".") >= 2 else f"{python_version}.0")
    return env


@dataclass(frozen=True)
class Constraint:
    """``source==source_version`` declares that ``target`` must match ``specifier``."""

    source: str
    source_version: str
    target: str
    specifier: SpecifierSet
    extra: str = ""          # non-empty if gated behind an extra
    marker: str = ""

    def permits(self, version: str) -> bool:
        """Does this constraint allow ``target==version``?

        An unparseable version is allowed: we cannot judge it, so we do not veto.
        """
        if not str(self.specifier):
            return True
        if safe_parse_version(version) is None:
            return True
        return self.specifier.contains(version, prereleases=True)

    def describe(self) -> str:
        return f"{self.source}=={self.source_version} requires {self.target}{self.specifier}"


def parse_requires_dist(source: str, source_version: str, raw: list[str],
                        env: dict | None = None,
                        active_extras: frozenset[str] = frozenset()) -> list[Constraint]:
    """Turn one release's Requires-Dist into constraints that apply *here*.

    Requirements gated behind an inactive extra, or behind a marker that is
    false for this environment, are dropped -- they genuinely do not constrain
    the current install. Requirements we cannot parse are also dropped, on the
    conservative rule above.
    """
    env = env if env is not None else marker_environment()
    out: list[Constraint] = []
    for line in raw or []:
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue

        extra = ""
        if req.marker is not None:
            # Evaluate once per plausible extra so `extra == 'scipy'` resolves.
            applies = False
            for candidate_extra in ("", *sorted(active_extras)):
                try:
                    if req.marker.evaluate({**env, "extra": candidate_extra}):
                        applies = True
                        break
                except Exception:
                    continue
            if not applies:
                # Record which extra would have activated it, for reporting.
                text = str(req.marker)
                if "extra" in text:
                    extra = text
                continue

        out.append(Constraint(
            source=canonicalize_name(source), source_version=source_version,
            target=canonicalize_name(req.name), specifier=req.specifier,
            extra=extra, marker=str(req.marker or ""),
        ))
    return out


@dataclass
class ConstraintGraph:
    """Declared constraints among a specific set of pinned versions."""

    edges: list[Constraint] = field(default_factory=list)
    # Packages whose metadata could not be read; their constraints are unknown.
    unknown: set[str] = field(default_factory=set)

    def add(self, constraint: Constraint) -> None:
        self.edges.append(constraint)

    def constraining(self, target: str) -> list[Constraint]:
        """Every declared constraint bearing on ``target``."""
        key = canonicalize_name(target)
        return [c for c in self.edges if c.target == key]

    def blockers(self, target: str, version: str,
                 exclude: set[str] | None = None) -> list[Constraint]:
        """Constraints that forbid ``target==version``.

        ``exclude`` names packages that are themselves free to move (a
        co-resolution group), so their constraints do not bind.
        """
        skip = {canonicalize_name(p) for p in (exclude or set())}
        return [c for c in self.constraining(target)
                if c.source not in skip and not c.permits(version)]

    def feasible(self, target: str, candidates: list[str],
                 exclude: set[str] | None = None
                 ) -> tuple[list[str], list[dict]]:
        """Split candidates into those the graph permits and those it forbids."""
        keep, blocked = [], []
        for version in candidates:
            bad = self.blockers(target, version, exclude)
            if bad:
                blocked.append({"version": version,
                                "blockers": [c.describe() for c in bad]})
            else:
                keep.append(version)
        return keep, blocked

    def couples(self) -> set[frozenset[str]]:
        """Unordered pairs the graph says are coupled, for scoring the regex."""
        return {frozenset((c.source, c.target)) for c in self.edges
                if c.source != c.target and str(c.specifier)}


def build_graph(pins: dict[str, str], index, *, python_version: str = "",
                active_extras: frozenset[str] = frozenset()) -> ConstraintGraph:
    """Read the declared constraints among a set of pinned packages.

    One metadata lookup per pinned package, cached on the index, so a repo with
    40 requirements costs 40 requests once -- against the dozens of multi-minute
    installs the resulting routing decisions avoid.
    """
    env = marker_environment(python_version)
    graph = ConstraintGraph()
    for name, version in pins.items():
        raw = index.requires_dist(name, version)
        if raw is None:
            graph.unknown.add(canonicalize_name(name))
            continue
        for constraint in parse_requires_dist(
                name, version, raw, env=env, active_extras=active_extras):
            graph.add(constraint)
    return graph


@dataclass
class UnlockSearch:
    """Result of asking which versions of one package permit another.

    ``complete`` matters more than it looks: an empty ``versions`` with
    ``complete=False`` means "I stopped looking", not "no such version exists".
    Collapsing those two into a bare ``None`` is how a search convinces itself a
    solvable conflict is unsolvable.
    """

    versions: list[str] = field(default_factory=list)
    scanned: int = 0
    complete: bool = True

    @property
    def newest(self) -> str | None:
        return self.versions[0] if self.versions else None


def unlocking_versions(source: str, source_candidates: list[str], target: str,
                       target_version: str, index, *, python_version: str = "",
                       budget: int = 60, stop_at_first: bool = False) -> UnlockSearch:
    """Which versions of ``source`` permit ``target==target_version``?

    This is the inversion that makes co-resolution a solve rather than a guess:
    greedy asks "bump torchvision to latest and hope", this asks "which
    torchvision declares a pin compatible with torch 2.0.0".

    Searched newest-first, because the newest unlocking version is the one we
    want. Note the answer is usually *not* near the top: torchvision's pins move
    in lockstep with torch, so the newest torchvision pins the newest torch and
    the match for an older target sits well down the list. An early cap here
    silently reports "unsolvable" for conflicts that are solvable -- hence the
    budget is in metadata requests (~100ms each, cached) rather than a small
    constant, and exhausting it is reported rather than hidden.
    """
    env = marker_environment(python_version)
    target_key = canonicalize_name(target)
    ordered = list(reversed(source_candidates))
    result = UnlockSearch(complete=True)

    for version in ordered:
        if result.scanned >= budget:
            result.complete = False
            break
        result.scanned += 1
        raw = index.requires_dist(source, version)
        if raw is None:
            continue
        relevant = [c for c in parse_requires_dist(source, version, raw, env=env)
                    if c.target == target_key]
        if not relevant:
            continue                      # states nothing about target -> no help
        if all(c.permits(target_version) for c in relevant):
            result.versions.append(version)
            if stop_at_first:
                break
    return result


def co_update_plan(target: str, target_version: str, graph: ConstraintGraph,
                   pins: dict[str, str], index, *, python_version: str = "",
                   budget: int = 60,
                   floors: dict[str, str] | None = None,
                   ceilings: dict[str, str] | None = None
                   ) -> tuple[dict[str, str] | None, str]:
    """The joint pin set that would let ``target`` reach ``target_version``.

    Returns ``(plan, reason)``. A ``None`` plan is always accompanied by why,
    and the two reasons are not interchangeable: ``no_unlocking_version`` is a
    proof that this target is unreachable while that blocker is present, and is
    worth telling the LLM tier; ``search_incomplete`` is only a budget running
    out, and proves nothing.

    ``floors`` lowers how far back a blocker may be traded, per package. By
    default a blocker may only move *up* from where it is now, which is right
    while the search is climbing. It is wrong once it has stopped: a package
    taken to its own newest installable version can forbid several others
    (Flask 1.1.4 declares ``click<8``; 1.1.2 does not), and stepping it back --
    never below the version the project itself started from -- buys more than
    it costs.
    """
    floors = floors or {}
    ceilings = ceilings or {}
    blockers = graph.blockers(target, target_version)
    if not blockers:
        return {canonicalize_name(target): target_version}, "no_blockers"

    plan = {canonicalize_name(target): target_version}
    for constraint in blockers:
        source = constraint.source
        current = pins.get(source)
        floor = floors.get(source, current)
        history = index.history(source)
        if history is None:
            return None, f"unknown_project:{source}"
        candidates = (history.between(floor, history.latest or floor)
                      if floor else list(history.versions))
        if floor and floor != current and history.contains(floor):
            # The floor is itself a valid trade, but it is the *last* resort:
            # keep the list newest-first so the search takes the smallest step
            # back that unlocks the target.
            candidates = sorted(
                set(candidates) | {floor},
                key=lambda v: safe_parse_version(v) or safe_parse_version("0"))
        ceiling = ceilings.get(source)
        if ceiling:
            # Only versions *below* where this blocker sits now. Without this
            # the search answers with the newest version that unlocks the
            # target -- which is the plan the greedy tier has already tried and
            # failed -- and never finds the older release that simply does not
            # carry the constraint. Flask 3.1.3 unlocks click 8 and breaks the
            # tests; Flask 1.1.2 unlocks it by never declaring click<8.
            limit = safe_parse_version(ceiling)
            candidates = [v for v in candidates
                          if (parsed := safe_parse_version(v)) and limit
                          and parsed < limit]
        if python_version:
            # Only versions this interpreter can install are worth proposing.
            try:
                candidates, _ = history.compatible(candidates, python_version)
            except Exception:
                pass
        search = unlocking_versions(
            source, candidates, target, target_version, index,
            python_version=python_version, budget=budget, stop_at_first=True)
        if search.newest is None:
            return None, (f"search_incomplete:{source}" if not search.complete
                          else f"no_unlocking_version:{source}")
        plan[source] = search.newest
    return plan, "solved"


def close_group(group: dict[str, str], pins: dict[str, str], index, *,
                python_version: str = "", limit: int = 40
                ) -> tuple[dict[str, str], list[dict]]:
    """Grow a co-update group until every member's declared requirements hold.

    Not used by the greedy tier, which follows the original paper and learns
    blockers one resolver report at a time. Kept as an analysis tool: comparing
    what this predicts with what the resolver actually reported measures how
    much of the conflict structure declared metadata captures.

    pip's resolver reports only the *first* conflict it meets, so a group built
    from failure messages grows by one package per failed install: moving flask
    to 3.1.3 surfaces click, then importlib-metadata, then jinja2, then markupsafe,
    then werkzeug -- five installs to learn what flask 3.1.3 states in its own
    metadata. This reads that metadata instead.

    For each member, every requirement on a *pinned* package whose current pin
    it rules out pulls that package into the group, at the newest version this
    interpreter can install that satisfies the requirement. Newcomers' own
    requirements are then checked the same way, until nothing changes.

    Only upward moves are proposed. When a requirement could only be met by
    going *below* a package's current pin, it is left for the resolver to
    report, and recorded in the returned notes. Requirements on packages that
    are not pinned (a new dependency such as blinker) are pip's to satisfy.

    Returns the closed group and one note per package added or left unresolved.
    """
    env = marker_environment(python_version)
    group = {canonicalize_name(k): v for k, v in group.items()}
    pinned = {canonicalize_name(k): v for k, v in pins.items()}
    notes: list[dict] = []
    frontier = list(group)
    seen: set[tuple[str, str]] = set()

    while frontier and len(group) < limit:
        name = frontier.pop(0)
        version = group[name]
        if (name, version) in seen:
            continue
        seen.add((name, version))
        raw = index.requires_dist(name, version)
        if not raw:
            continue
        for constraint in parse_requires_dist(name, version, raw, env=env):
            dep = constraint.target
            if dep == name or dep not in pinned:
                continue
            current = group.get(dep, pinned[dep])
            if constraint.permits(current):
                continue

            history = index.history(dep)
            if history is None:
                continue
            floor = pinned[dep]
            options = history.between(floor, history.latest or floor)
            if python_version:
                try:
                    options, _ = history.compatible(options, python_version)
                except Exception:
                    pass
            fitting = [v for v in options if constraint.permits(v)]
            if not fitting:
                notes.append({"package": dep, "required_by": f"{name}=={version}",
                              "requirement": f"{dep}{constraint.specifier}",
                              "resolved": None,
                              "why": "no newer installable version satisfies it"})
                continue
            chosen = max(fitting, key=lambda v: safe_parse_version(v)
                         or safe_parse_version("0"))
            if group.get(dep) == chosen:
                continue
            group[dep] = chosen
            frontier.append(dep)
            notes.append({"package": dep, "required_by": f"{name}=={version}",
                          "requirement": f"{dep}{constraint.specifier}",
                          "resolved": chosen})
    return group, notes
