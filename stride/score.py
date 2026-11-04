"""One score for every arm: releases behind, over the same package list.

The raw age reports (``age_start``/``age_end``) score whatever is pinned at the
end, so two arms that reach the same versions can still differ: pip freeze hides
setuptools, so it falls off a baseline's list, and a package STRIDE removed
leaves STRIDE's. Here every arm is scored over the packages the project
declared at the start, with two rules:

- build tools (pip, setuptools, wheel) are not scored in any arm;
- a declared package the patch removed counts as 0 behind: nothing outdated is
  left. Removals are listed in ``removed`` so a reader can see them.
"""
from __future__ import annotations

BUILD_TOOLS = {"pip", "setuptools", "wheel"}


def _norm(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def stats(per_package: dict[str, int], unknown: list[str] | None = None) -> dict:
    values = sorted(per_package.values())
    n = len(values)
    at_latest = sum(1 for v in values if v == 0)
    return {
        "n_packages": n,
        "mean_releases_behind": round(sum(values) / n, 3) if n else 0.0,
        "median_releases_behind": (
            values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
        ) if n else 0.0,
        "max_releases_behind": values[-1] if n else 0,
        "at_latest": at_latest,
        "at_latest_pct": round(100.0 * at_latest / n, 2) if n else 0.0,
        "per_package": per_package,
        "unknown": sorted(unknown or []),
    }


def same_list_score(age_start: dict, age_end: dict, dropped: list[str]) -> dict:
    """Start and end scores over the declared packages, build tools left out."""
    start = {k: v for k, v in (age_start.get("per_package") or {}).items()
             if _norm(k) not in BUILD_TOOLS}
    end_by = {_norm(k): v for k, v in (age_end.get("per_package") or {}).items()}
    removed = sorted(d for d in dropped or [] if _norm(d) not in BUILD_TOOLS)
    gone = {_norm(d) for d in removed}
    end, unknown = {}, []
    for name in start:
        if _norm(name) in gone:
            end[name] = 0
        elif _norm(name) in end_by:
            end[name] = end_by[_norm(name)]
        else:
            unknown.append(name)
    return {"start": stats(start), "end": stats(end, unknown), "removed": removed,
            "rule": "declared packages; build tools not scored; removed = 0 behind"}


def add_score(summary: dict) -> dict:
    if summary.get("age_start") and summary.get("age_end"):
        summary["score"] = same_list_score(summary["age_start"], summary["age_end"],
                                           summary.get("dropped") or [])
    return summary
