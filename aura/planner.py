"""Conflict diagnosis and LLM-guided co-resolution planning.

The validation here is what the paper *claimed* v1 did and it did not. v1
computed a version ceiling and then never compared against it, and never asked
the index whether a proposed version exists at all -- so hallucinated versions
went straight to a full install-and-validate cycle, and no hallucination rate
could be reported. Every rejection is now categorised and logged, which is what
makes a per-model comparison of wrong-version rates possible.

``explanation_first`` is a toggle rather than a constant so the claim that
explanation-first prompting helps can be tested by an ablation instead of
asserted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from packaging.utils import canonicalize_name

from aura.index import IndexError_, PackageIndex
from aura.llm import PLAN_SCHEMA, LLMUnavailable
from aura.reqfile import safe_parse_version
from aura.telemetry import LLMUsage

# Names that appear in resolver output but are never the conflict.
_NOISE = {
    "python", "pip", "setuptools", "wheel", "setup", "dependencies", "versions",
    "requirement", "requirements", "conflict", "conflicts", "between", "and",
    "the", "version", "package", "packages", "for", "from", "with", "error",
    "warning", "note", "info", "install", "installed", "installing", "user",
    "requested", "depends", "requires", "incompatible", "cannot", "could",
    "https", "http", "org", "com", "simple", "index", "none", "any",
}


class RejectionReason:
    NOT_ON_INDEX = "not_on_index"          # hallucinated: no such release
    NOT_AN_UPGRADE = "not_an_upgrade"      # downgrade or no-op
    EXCEEDS_CEILING = "exceeds_ceiling"    # newer than the latest known release
    UNPARSEABLE = "unparseable"            # not a "name==version" requirement
    NOT_IN_ENV = "not_in_env"              # package is not part of this project
    MISSING_TARGET = "missing_target"      # plan omits the package it was asked about


@dataclass
class DecisionPoint:
    """The exact inputs an LLM escalation was made from.

    Logged verbatim so the same decision can later be replayed against a
    different backend. Comparing backends at identical inputs is a controlled
    comparison; comparing whole end-to-end runs is not, because each backend
    takes a different search trajectory and pays different install costs.
    """

    target: str = ""
    error_log: str = ""
    current: dict = field(default_factory=dict)
    available: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    iteration: int = 0

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            # The prompt itself only ever sees the tail of the log, so storing
            # that same tail is sufficient for an exact replay.
            "error_log": self.error_log[-6000:],
            "current": self.current,
            "available": self.available,
            "history": [list(h) for h in self.history],
            "iteration": self.iteration,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "DecisionPoint":
        return cls(
            target=raw.get("target", ""), error_log=raw.get("error_log", ""),
            current=raw.get("current", {}) or {}, available=raw.get("available", {}) or {},
            history=[tuple(h) for h in raw.get("history", []) or []],
            iteration=raw.get("iteration", 0),
        )


@dataclass
class Plan:
    requirements: list[str] = field(default_factory=list)   # ["torch==2.0.1", ...]
    rejected: list[dict] = field(default_factory=list)
    explanation: str = ""
    resolvable: bool = True
    reason: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: str = ""
    decision: "DecisionPoint | None" = None

    @property
    def is_empty(self) -> bool:
        return not self.requirements

    def as_pins(self) -> dict[str, str]:
        out = {}
        for req in self.requirements:
            name, _, version = req.partition("==")
            out[canonicalize_name(name)] = version
        return out


_REQUIREMENT_ECHO = re.compile(
    r"^\s*(?:Collecting|Requirement already satisfied|Using cached|Downloading|"
    r"Obtaining|Processing|Installing collected|Successfully|Looking in|"
    r"Attempting uninstall|Found existing|Uninstalling)\b|\(from -r |\(line \d+\)",
    re.I)
_REPORT_START = re.compile(r"The conflict is caused by:?", re.I)
_REPORT_REQUESTED = re.compile(
    r"The user requested (?:\(constraint\) )?(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)", re.I)
_REPORT_DEPENDS = re.compile(
    r"(?P<source>[A-Za-z0-9][A-Za-z0-9._-]*)\s+\S+\s+depends on\s+"
    r"(?P<target>[A-Za-z0-9][A-Za-z0-9._-]*)", re.I)


def parse_conflict_report(error_log: str) -> dict | None:
    """Read pip's own "The conflict is caused by:" section, if it printed one.

    This is pip stating the conflict in a fixed format -- who was requested, and
    which installed package constrains what -- so it is the authoritative list of
    packages involved, and the reference the regex extractor is scored against.
    Returns ``None`` when there is no such section.
    """
    lines = (error_log or "").splitlines()
    starts = [i for i, l in enumerate(lines) if _REPORT_START.search(l)]
    if not starts:
        return None
    requested: set[str] = set()
    constraints: list[dict] = []
    for start in starts:
        for line in lines[start + 1:]:
            if not line.strip():
                break
            if m := _REPORT_REQUESTED.search(line):
                requested.add(canonicalize_name(m.group("name")))
            elif m := _REPORT_DEPENDS.search(line):
                constraints.append({"source": canonicalize_name(m.group("source")),
                                    "target": canonicalize_name(m.group("target")),
                                    "text": line.strip()})
    names = requested | {c["source"] for c in constraints} | {
        c["target"] for c in constraints}
    return {"requested": sorted(requested), "constraints": constraints,
            "packages": sorted(names)}


def extract_conflicting_packages(error_log: str, known: set[str] | None = None) -> list[str]:
    """Pull candidate blocker names out of resolver output.

    ``known`` restricts results to names that actually exist in the environment,
    which raises precision sharply -- the unrestricted regex matches a great deal
    of prose. Both the raw and restricted sets are logged so the component's
    precision can be reported rather than assumed.
    """
    report = parse_conflict_report(error_log)
    if report is not None:
        found = set(report["packages"])
        if known is not None:
            found &= {canonicalize_name(k) for k in known}
        return sorted(found)

    # No structured report. Drop the lines where pip merely echoes the
    # requirements it is installing -- otherwise every name in the file comes
    # back as a "blocker" and co-resolution degenerates into upgrading all of it.
    error_log = "\n".join(l for l in (error_log or "").splitlines()
                          if not _REQUIREMENT_ECHO.search(l))
    found: set[str] = set()
    patterns = [
        re.compile(r"(?P<name>[A-Za-z][A-Za-z0-9._-]{1,60})\s*(?:==|>=|<=|~=|!=|<|>)\s*\d"),
        re.compile(r"(?P<name>[A-Za-z][A-Za-z0-9._-]{1,60})\s+\d+(?:\.\d+)+"),
        re.compile(r"(?P<name>[A-Za-z][A-Za-z0-9._-]{1,60})\s*\(\d+(?:\.\d+)+\)"),
        re.compile(r"(?:depends on|requires)\s+(?P<name>[A-Za-z][A-Za-z0-9._-]{1,60})"),
        re.compile(r"The conflict is caused by:?\s*\n\s*(?P<name>[A-Za-z][A-Za-z0-9._-]{1,60})"),
    ]
    for pattern in patterns:
        for match in pattern.finditer(error_log or ""):
            name = canonicalize_name(match.group("name"))
            if len(name) > 1 and name not in _NOISE:
                found.add(name)
    if known is not None:
        found &= {canonicalize_name(k) for k in known}
    return sorted(found)


def validate_plan(
    raw_requirements: list[str],
    *,
    current: dict[str, str],
    ceilings: dict[str, str],
    index: PackageIndex | None = None,
    allowed: set[str] | None = None,
) -> tuple[list[str], list[dict]]:
    """Filter a proposed plan down to requirements we are willing to execute.

    Rejects, in order: unparseable entries, packages outside the project,
    non-upgrades, versions above the known ceiling, and -- when an index is
    supplied -- versions that do not exist at all.
    """
    accepted: list[str] = []
    rejected: list[dict] = []

    def reject(req: str, reason: str, **extra):
        rejected.append({"req": req, "reason": reason, **extra})

    for raw in raw_requirements:
        req = str(raw).strip()
        name, sep, version = req.partition("==")
        name, version = canonicalize_name(name.strip()), version.strip()
        if not sep or not name or not version:
            reject(req, RejectionReason.UNPARSEABLE)
            continue

        if allowed is not None and name not in allowed:
            reject(req, RejectionReason.NOT_IN_ENV)
            continue

        proposed = safe_parse_version(version)
        if proposed is None:
            reject(req, RejectionReason.UNPARSEABLE)
            continue

        installed = safe_parse_version(current.get(name, "0"))
        if installed is not None and proposed <= installed:
            reject(req, RejectionReason.NOT_AN_UPGRADE, current=current.get(name))
            continue

        ceiling = safe_parse_version(ceilings.get(name, ""))
        if ceiling is not None and proposed > ceiling:
            # v1 computed this bound and then never compared against it.
            reject(req, RejectionReason.EXCEEDS_CEILING, ceiling=ceilings.get(name))
            continue

        if index is not None:
            try:
                if not index.version_exists(name, version):
                    reject(req, RejectionReason.NOT_ON_INDEX)
                    continue
            except IndexError_:
                pass  # index unreachable: fall through rather than block the run

        accepted.append(f"{name}=={version}")

    return accepted, rejected


PROMPT_HEADER = """You are resolving a Python dependency deadlock.

TARGET PACKAGE: {target}

The deterministic strategies have been exhausted:
  - a reverse linear scan over {target}'s own version line
  - a greedy joint update of {target} together with every blocker found in the logs

Your job is to propose a set of package versions to install together that
satisfies the resolver and keeps the project's tests passing.
"""

PROMPT_HEADER_FIRST = """You are upgrading a Python project's dependencies.

TARGET PACKAGE: {target}

Nothing has been tried yet. Propose a set of package versions to install
together that moves {target} as far forward as possible, satisfies the resolver
and keeps the project's tests passing.
"""

PROMPT_RULES = """
STRICT RULES
1. FORWARD ONLY. Never propose a version lower than or equal to the currently
   installed version listed under CURRENT.
2. STAY WITHIN THE CEILING. Never propose a version higher than the value listed
   for that package under AVAILABLE.
3. REAL VERSIONS ONLY. Only propose versions that genuinely exist on the package
   index. Invented versions are rejected before installation and waste an attempt.
4. INTERMEDIATE VERSIONS ARE ALLOWED and are often the answer: you may propose
   any real version between CURRENT and AVAILABLE for a package.
5. SCOPE. The plan must include '{target}', and may only contain packages that
   appear in CURRENT.

EARLY EXIT
If no combination of versions can resolve this -- for example the chain requires
a Python version this runtime does not provide, or a package needs a system
library that is absent -- set "resolvable" to false and explain why. Do not
propose a plan in that case.
"""

PROMPT_CONTEXT = """
CURRENT (floor -- what is installed now):
{current}

AVAILABLE (ceiling -- the newest release of each package):
{available}

RESOLVER / TEST OUTPUT:
{log}
{history}"""

SCHEMA_WITH_EXPLANATION = """
Return one JSON object and nothing else:
{
  "explanation": "one or two sentences naming the specific constraint that
                  conflicts and why your chosen versions satisfy it",
  "resolvable": true,
  "reason": "required only when resolvable is false",
  "plausible": true,
  "proposed_plan": ["package==version", "..."]
}"""

SCHEMA_PLAN_ONLY = """
Return one JSON object and nothing else:
{
  "resolvable": true,
  "reason": "required only when resolvable is false",
  "plausible": true,
  "proposed_plan": ["package==version", "..."]
}"""


def build_prompt(
    *, target: str, error_log: str, current: dict[str, str],
    available: dict[str, str], history: list[tuple[str, str]] | None = None,
    explanation_first: bool = True, log_chars: int = 6000,
    style: str = "full",
) -> str:
    if style == "compact":
        return build_compact_prompt(target=target, error_log=error_log,
                                    current=current, available=available,
                                    history=history)
    if style != "full":
        raise ValueError(f"unknown prompt style {style!r}; expected 'full' or 'compact'")
    history_text = ""
    if history:
        lines = ["\nPREVIOUS FAILED ATTEMPTS (do not repeat these):"]
        for i, (plan, failure) in enumerate(history, 1):
            lines.append(f"  {i}. {plan}\n     -> {str(failure)[:400]}")
        history_text = "\n".join(lines)

    header = PROMPT_HEADER.format(target=target)
    if not (error_log or "").strip():
        # Model-only search, first request: no deterministic strategy has run.
        header = PROMPT_HEADER_FIRST.format(target=target)
    return (
        header
        + PROMPT_RULES.format(target=target)
        + PROMPT_CONTEXT.format(
            current=json.dumps(current, indent=2, sort_keys=True),
            available=json.dumps(available, indent=2, sort_keys=True),
            log=(error_log or "")[-log_chars:],
            history=history_text,
        )
        + (SCHEMA_WITH_EXPLANATION if explanation_first else SCHEMA_PLAN_ONLY)
    )


# --------------------------------------------------------------- compact style

# Observed failure modes when replaying real conflicts against small local
# models, which the full prompt does nothing to discourage:
#
#   qwen2.5-coder:14b  declared a solvable coupling "unresolvable" and stopped.
#                      The early-exit clause reads as an accepted way out.
#   llama3.1:8b        upgraded a different package and dropped the target, so
#                      the plan could not resolve anything it was asked about.
#
# The compact style is not merely shorter. It (a) narrows the package table to
# what the conflict is actually about, since a fifty-entry JSON dump buries the
# two lines that matter, (b) restates the target requirement immediately before
# the answer, where it is hardest to lose, (c) narrows early exit to causes that
# no version choice can fix, explicitly excluding package-vs-package conflicts,
# and (d) shows one worked example, which small instruction-tuned models follow
# far more reliably than prose rules.

COMPACT_LOG_CHARS = 1800
COMPACT_MAX_PACKAGES = 14


def _relevant_packages(target: str, current: dict[str, str], available: dict[str, str],
                       error_log: str, limit: int = COMPACT_MAX_PACKAGES) -> list[str]:
    """Target first, then packages the error log actually mentions."""
    target_key = canonicalize_name(target)
    ordered = [target_key] if target_key in current else []
    mentioned = set(extract_conflicting_packages(error_log, known=set(current)))
    ordered += sorted(n for n in mentioned if n != target_key)
    for name in sorted(available):
        if len(ordered) >= limit:
            break
        if name not in ordered:
            ordered.append(name)
    return ordered[:limit]


def build_compact_prompt(*, target: str, error_log: str, current: dict[str, str],
                         available: dict[str, str],
                         history: list[tuple[str, str]] | None = None) -> str:
    target_key = canonicalize_name(target)
    names = _relevant_packages(target_key, current, available, error_log)

    table = []
    for name in names:
        installed = current.get(name, "?")
        newest = available.get(name, installed)
        mark = "  <-- MUST APPEAR IN YOUR ANSWER" if name == target_key else ""
        table.append(f"  {name:<24} installed {installed:<12} newest {newest}{mark}")

    tried = ""
    if history:
        lines = [f"  {plan}  ->  failed: {str(reason)[:120]}"
                 for plan, reason in history[-3:]]
        tried = "\nALREADY TRIED (do not repeat):\n" + "\n".join(lines) + "\n"

    log = (error_log or "").strip()
    if log:
        opening = ("Two Python packages have incompatible version requirements.\n"
                   "Choose versions that resolve the conflict.")
        evidence = f"ERROR\n{log[-COMPACT_LOG_CHARS:]}\n\n"
    else:
        # The first request of a model-only search: nothing has been installed
        # yet, so there is no error to show. Where to start is the model's call.
        opening = (f"Upgrade {target_key} in this Python project as far as possible\n"
                   "while keeping its dependencies compatible. Nothing has been\n"
                   "tried yet. If other packages must move with it, include them.")
        evidence = ""
    return f"""{opening}

{evidence}PACKAGES YOU MAY CHANGE
{chr(10).join(table)}
{tried}
RULES
- Every version must be HIGHER than "installed" and NOT higher than "newest".
- Use only versions that really exist. Do not invent one.
- When two packages are coupled, the answer is usually a SMALLER step on both,
  not the newest of either. Prefer intermediate versions.

Set "resolvable": false ONLY when no choice of versions could ever work, such as
a package requiring a newer Python than this runtime. A conflict between two
packages is NOT such a case - it is what you are here to solve, so solve it.

EXAMPLE OF A GOOD ANSWER
{{"resolvable": true, "plausible": true, "proposed_plan": ["examplea==1.4.2", "exampleb==2.1.0"]}}

Now answer for {target_key}. Your "proposed_plan" MUST contain {target_key}.
Reply with one JSON object and nothing else."""


def _parse_json_object(text: str) -> dict | None:
    cleaned = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", (text or "").strip())
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class Planner:
    """Wraps an LLM client with constrained prompting and plan validation."""

    def __init__(self, llm, index: PackageIndex | None = None,
                 explanation_first: bool = True, max_output_tokens: int = 2048,
                 prompt_style: str = "full"):
        self.llm = llm
        self.index = index
        self.explanation_first = explanation_first
        self.max_output_tokens = max_output_tokens
        self.prompt_style = prompt_style

    def replay(self, decision: DecisionPoint) -> Plan:
        """Re-run a recorded decision point, unchanged, against this backend."""
        return self.propose(
            target=decision.target, error_log=decision.error_log,
            current=decision.current, available=decision.available,
            history=[tuple(h) for h in decision.history],
        )

    def propose(
        self, *, target: str, error_log: str, current: dict[str, str],
        available: dict[str, str], history: list[tuple[str, str]] | None = None,
    ) -> Plan:
        decision = DecisionPoint(
            target=target, error_log=error_log or "", current=dict(current),
            available=dict(available), history=list(history or []),
        )
        prompt = build_prompt(
            target=target, error_log=error_log, current=current,
            available=available, history=history,
            explanation_first=self.explanation_first, style=self.prompt_style,
        )
        try:
            response = self.llm.generate(
                prompt, max_output_tokens=self.max_output_tokens, schema=PLAN_SCHEMA)
        except LLMUnavailable:
            raise
        plan = Plan(usage=response.usage, raw=response.text, decision=decision)

        if not response.usage.ok:
            plan.resolvable, plan.reason = True, f"llm_error: {response.usage.error}"
            return plan

        parsed = _parse_json_object(response.text)
        if parsed is None:
            plan.reason = "unparseable_response"
            return plan

        plan.explanation = str(parsed.get("explanation", ""))[:2000]
        if not parsed.get("resolvable", True):
            plan.resolvable = False
            plan.reason = str(parsed.get("reason", "model declared conflict unresolvable"))
            return plan
        if not parsed.get("plausible", True):
            plan.reason = "model reported no plausible plan"
            return plan

        raw_plan = parsed.get("proposed_plan") or []
        if not isinstance(raw_plan, list):
            plan.reason = "proposed_plan was not a list"
            return plan

        plan.requirements, plan.rejected = validate_plan(
            [str(r) for r in raw_plan],
            current=current, ceilings=available,
            index=self.index, allowed=set(current),
        )
        # Prompt rule 3 says the plan must include the target package. Weaker
        # backends drop it and "solve" the conflict by upgrading something else,
        # which leaves the package that triggered the escalation untouched.
        # Enforcing the rule turns that into a measurable failure mode instead
        # of a silently accepted no-op.
        target_key = canonicalize_name(target)
        if plan.requirements and target_key not in plan.as_pins():
            plan.rejected.append({"req": ", ".join(plan.requirements),
                                  "reason": RejectionReason.MISSING_TARGET,
                                  "target": target_key})
            plan.requirements = []
            plan.reason = f"plan omitted the target package {target_key!r}"
            return plan
        if not plan.requirements:
            plan.reason = "every proposed requirement was rejected by validation"
        return plan
