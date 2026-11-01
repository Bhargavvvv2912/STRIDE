"""Subject registry and the pre-registered selection protocol.

All three reviewers flagged the sample: "convenience sample", "heavily skewed
toward ML research artifacts", "32 is small". Reviewer A asked directly how we
ensure the sample is not biased toward showing AURA in a good light.

The answer has to be procedural, so the protocol lives here as executable rules:
criteria are declared before screening, every candidate that is screened is
recorded with its verdict, and rejections carry a reason drawn from a fixed
vocabulary. The rejection log ships with the paper. A subject is never dropped
because of how AURA performed on it -- ``OUTCOME_DEPENDENT`` exists only so that
such a rejection would be conspicuous if it ever appeared.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


class Rejection:
    """Fixed vocabulary of reasons a screened candidate can be excluded."""

    NO_DEPENDENCY_SPEC = "no_dependency_spec"        # no requirements.txt or equivalent
    BASELINE_DOES_NOT_INSTALL = "baseline_does_not_install"
    BASELINE_FAILS_VALIDATION = "baseline_fails_validation"
    NO_VALIDATION_POSSIBLE = "no_validation_possible"  # nothing executable to check
    ALREADY_CURRENT = "already_current"               # no outdated dependency to update
    REQUIRES_GPU = "requires_gpu"
    REQUIRES_CREDENTIALS = "requires_credentials"     # needs private data or API keys
    NOT_PYTHON = "not_python"
    DUPLICATE_DOMAIN = "duplicate_domain"             # quota reached for this domain
    TOO_LARGE = "too_large"                           # exceeds the compute budget
    # The naive upgrade already passes (screening's "weak"), so the authors
    # spent the budget on subjects where it breaks. Decided from screening,
    # before any AURA run on the subject -- never from how AURA did on it.
    WEAK_DEPRIORITIZED = "weak_deprioritized"
    OUTCOME_DEPENDENT = "outcome_dependent"           # must never be used; see above


VALID_REJECTIONS = {v for k, v in vars(Rejection).items() if not k.startswith("_")}


class Provenance:
    """Where a subject's validation logic came from. Recorded per subject."""

    REPO_TESTS = "repo_tests"              # the project's own test suite, unmodified
    REPO_CI = "repo_ci"                    # derived from the project's CI config
    REPO_README = "repo_readme"            # derived from documented usage examples
    REPO_EXAMPLE = "repo_example"          # an example script shipped by the project
    AUTHORED = "authored"                  # written by us; requires a review record


# Dependency rot is a function of time, so the sample is stratified by it: a
# result drawn from one era is a result about that era, and "AURA advances more
# packages" means something different for a project last touched in 2016 than
# for one from last year. Boundaries follow the ecosystem's own breaks rather
# than round numbers -- they are where a project's environment stops being
# reproducible by default.
ERAS = (
    ("current", "2024-01-01", "2100-01-01",
     "pins are still close to the releases they name"),
    ("recent", "2021-01-01", "2023-12-31",
     "Flask 2, setuptools 60 and the end of easy_install"),
    ("stale", "2018-01-01", "2020-12-31",
     "Python 3.7-3.8, Flask 1.x, before the metadata rewrites"),
    ("old", "2015-01-01", "2017-12-31",
     "Python 3.5-3.6, before wheels were universal"),
    ("ancient", "1990-01-01", "2014-12-31",
     "the Python 2 era; most of these no longer install at all"),
)


def era_of(date: str) -> str:
    """Which stratum a commit date falls in, or "" when it is unknown."""
    if not date:
        return ""
    for name, start, end in ((e[0], e[1], e[2]) for e in ERAS):
        if start <= date[:10] <= end:
            return name
    return ""


@dataclass
class Subject:
    name: str
    url: str = ""
    commit: str = ""                       # pinned so the subject is reproducible
    commit_date: str = ""                  # YYYY-MM-DD of that commit. A baseline
                                           # that no longer installs is rebuilt as
                                           # of this date; derived from the clone
                                           # when the registry does not state it.
    domain: str = ""
    is_ml_research: bool = False           # tracked to report the skew explicitly
    python_version: str = "3.11"
    requirements_file: str = "requirements.txt"
    primary_requirements_file: str = "primary_requirements.txt"
    # Per-dependency bumps collected out of band (real Dependabot output).
    # Absent means the proposal arm reports "not collected" rather than
    # quietly standing in something else.
    is_installable: bool = False
    project_extras: str = ""
    validation: dict = field(default_factory=dict)
    # Commands run in each probe environment after install, before validation.
    # Some suites need fixtures pip cannot provide -- nltk corpora, spaCy
    # models, generated test data. Leaving that to ambient machine state makes
    # the result depend on what happens to be in the runner's home directory,
    # which is neither reproducible nor visible in the log.
    environment_setup: list = field(default_factory=list)
    # Environment variables the project reads at test time, as placeholders.
    # A Django project refuses to start without a SECRET_KEY, and boto3 without
    # a region, even when every request its tests make is replayed from a
    # recording -- so "no working baseline" there is a missing string, not a
    # broken project. Values here must be placeholders and nothing else: a
    # subject whose tests need a real credential or a live service is excluded
    # (requires_credentials), not configured.
    environment: dict = field(default_factory=dict)
    validation_provenance: str = Provenance.AUTHORED
    validation_reviewed_by: str = ""
    validation_authored_date: str = ""
    validation_notes: str = ""
    metric_tolerance: float = 0.01
    screened_date: str = ""
    included: bool = True
    rejection_reason: str = ""
    rejection_note: str = ""

    @property
    def era(self) -> str:
        """The stratum this subject represents, from its pinned commit's date."""
        return era_of(self.commit_date)

    def __post_init__(self):
        if not self.included and self.rejection_reason not in VALID_REJECTIONS:
            raise ValueError(
                f"{self.name}: rejection reason {self.rejection_reason!r} is not in "
                f"the declared vocabulary {sorted(VALID_REJECTIONS)}"
            )
        if self.rejection_reason == Rejection.OUTCOME_DEPENDENT:
            raise ValueError(
                f"{self.name}: subjects must never be excluded based on how AURA "
                "performed on them."
            )

    def agent_config(self, repo_root: str | Path) -> dict:
        return {
            "PROJECT_NAME": self.name,
            "IS_INSTALLABLE_PACKAGE": self.is_installable,
            "PROJECT_EXTRAS": self.project_extras,
            "REQUIREMENTS_FILE": self.requirements_file,
            "PRIMARY_REQUIREMENTS_FILE": self.primary_requirements_file,
            "ENVIRONMENT_SETUP": list(self.environment_setup),
            "VALIDATION_CONFIG": {**self.validation,
                                  "project_dir": str(Path(repo_root).resolve())},
            "METRIC_TOLERANCE": self.metric_tolerance,
            "ON_BROKEN_BASELINE": "fail",
            # Set by the runner: where this subject's baseline is frozen.
            "FROZEN_BASELINE": getattr(self, "frozen_baseline", ""),
        }


SELECTION_CRITERIA = [
    "C1. Publicly available on GitHub with a pinned or partially pinned "
    "dependency specification (requirements.txt, setup.py or pyproject.toml).",
    "C2. Installs and passes its validation from its own declared requirements "
    "on the recorded Python version -- the working-baseline precondition.",
    "C3. At least one declared dependency is behind its latest release, so "
    "there is something to modernize.",
    "C4. Has executable validation logic whose provenance can be recorded: the "
    "project's own tests, its CI configuration, a documented example, or a "
    "script we author and review against the project's documentation.",
    "C5. Runs to completion on CPU within the per-subject compute budget, with "
    "no private credentials or datasets required.",
    "C6. Domain quota: at most N subjects per application domain, so the sample "
    "is not dominated by any single area. Non-ML domains are screened first "
    "to counteract the ML skew of the previous benchmark.",
]

NON_ML_DOMAIN_TARGET = 0.40   # at least 40% of included subjects outside ML research


@dataclass
class Registry:
    subjects: list[Subject] = field(default_factory=list)

    @property
    def included(self) -> list[Subject]:
        return [s for s in self.subjects if s.included]

    @property
    def rejected(self) -> list[Subject]:
        return [s for s in self.subjects if not s.included]

    def era_counts(self) -> dict[str, int]:
        """How many included subjects each stratum holds, in era order.

        Reported so the shape of the sample is visible rather than implied: a
        conclusion drawn from four projects of one vintage is a conclusion
        about that vintage.
        """
        counts = {name: 0 for name, _, _, _ in ERAS}
        counts[""] = 0
        for subject in self.included:
            counts[subject.era] = counts.get(subject.era, 0) + 1
        return {k: v for k, v in counts.items() if k or v}

    def domain_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.included:
            counts[s.domain] = counts.get(s.domain, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.rejected:
            counts[s.rejection_reason] = counts.get(s.rejection_reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def provenance_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.included:
            counts[s.validation_provenance] = counts.get(s.validation_provenance, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def composition(self) -> dict:
        """The numbers that go in the sample-composition table."""
        included = self.included
        n = len(included)
        ml = sum(1 for s in included if s.is_ml_research)
        return {
            "screened": len(self.subjects),
            "included": n,
            "rejected": len(self.rejected),
            "ml_research": ml,
            "non_ml": n - ml,
            "non_ml_fraction": round((n - ml) / n, 3) if n else 0.0,
            "meets_non_ml_target": ((n - ml) / n >= NON_ML_DOMAIN_TARGET) if n else False,
            "domains": self.domain_counts(),
            "rejections": self.rejection_counts(),
            "validation_provenance": self.provenance_counts(),
        }

    def audit(self) -> list[str]:
        """Protocol violations that must be fixed before results are reported."""
        problems = []
        for s in self.included:
            if not s.commit:
                problems.append(f"{s.name}: no pinned commit (C1: not reproducible)")
            if not s.validation:
                problems.append(f"{s.name}: no validation configured (C4)")
            if s.validation_provenance == Provenance.AUTHORED and not s.validation_reviewed_by:
                problems.append(
                    f"{s.name}: authored validation script with no recorded reviewer (C4)")
            if not s.screened_date:
                problems.append(f"{s.name}: no screening date recorded")
        comp = self.composition()
        if comp["included"] and not comp["meets_non_ml_target"]:
            problems.append(
                f"sample is {100 * (1 - comp['non_ml_fraction']):.0f}% ML research; "
                f"target is at most {100 * (1 - NON_ML_DOMAIN_TARGET):.0f}%")
        return problems

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([asdict(s) for s in self.subjects], indent=2) + "\n",
            encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Subject(**item) for item in raw])
