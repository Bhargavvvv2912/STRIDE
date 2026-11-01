"""Small, dependency-free statistics for paired arm comparisons.

Reviewer C noted the absence of significance testing. Arms are run on the *same*
subjects, so comparisons are paired and McNemar's test is the right instrument;
with tens of subjects the exact binomial form is preferable to the chi-square
approximation. Proportions get Wilson intervals, which behave sensibly at the
0% and 100% ends where a benchmark of this size frequently lands.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Proportion:
    successes: int
    total: int
    low: float
    high: float

    @property
    def value(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def __str__(self) -> str:
        return (f"{self.successes}/{self.total} "
                f"({100 * self.value:.1f}%, 95% CI {100 * self.low:.1f}-{100 * self.high:.1f})")


def wilson_interval(successes: int, total: int, z: float = 1.959963985) -> Proportion:
    """Wilson score interval: well behaved at 0/n and n/n, unlike the normal one."""
    if total == 0:
        return Proportion(0, 0, 0.0, 0.0)
    p = successes / total
    denominator = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denominator
    return Proportion(successes, total, max(0.0, centre - spread), min(1.0, centre + spread))


def _binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


@dataclass
class McNemarResult:
    b: int              # arm A succeeded, arm B failed
    c: int              # arm B succeeded, arm A failed
    n_discordant: int
    p_value: float
    odds_ratio: float | None

    def __str__(self) -> str:
        ratio = "inf" if self.odds_ratio is None else f"{self.odds_ratio:.2f}"
        return (f"b={self.b}, c={self.c}, n_discordant={self.n_discordant}, "
                f"p={self.p_value:.4g}, OR={ratio}")


def mcnemar_exact(b: int, c: int) -> McNemarResult:
    """Two-sided exact McNemar test on the discordant pairs.

    ``b`` and ``c`` count the subjects where exactly one of the two arms
    succeeded. Concordant pairs carry no information about a difference and are
    correctly ignored.
    """
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 0, 1.0, None)
    smaller = min(b, c)
    tail = sum(_binom_pmf(k, n) for k in range(smaller + 1))
    p = min(1.0, 2 * tail)
    odds = (b / c) if c else None
    return McNemarResult(b, c, n, p, odds)


def paired_outcomes(a: dict[str, bool], b: dict[str, bool]) -> McNemarResult:
    """McNemar over two {subject: success} mappings, on their shared subjects."""
    shared = sorted(set(a) & set(b))
    b_count = sum(1 for s in shared if a[s] and not b[s])
    c_count = sum(1 for s in shared if b[s] and not a[s])
    return mcnemar_exact(b_count, c_count)


def bootstrap_ci(values: list[float], statistic=None, iterations: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns ``(point, low, high)``."""
    if not values:
        return 0.0, 0.0, 0.0
    statistic = statistic or (lambda xs: sum(xs) / len(xs))
    point = statistic(values)
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(statistic([values[rng.randrange(n)] for _ in range(n)])
                     for _ in range(iterations))
    lo = samples[int((alpha / 2) * iterations)]
    hi = samples[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return point, lo, hi


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 4),
        "median": round(median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "total": round(sum(values), 4),
    }
