"""Package index queries: version listings, existence checks, and version age.

Adds three things v1 lacked:

* **A real existence check.** The paper claims AURA rejects "hallucinated
  versions ... or suggestions that exceed known latest versions". In v1 the
  ceiling was computed and then never compared, and nothing was checked against
  the index at all. ``version_exists`` is what makes that claim true, and what
  lets a hallucination *rate* be reported per model.
* **Working retries.** v1 decorated ``get_latest_version`` with tenacity but
  swallowed every exception inside the function, so it never retried; a
  transient index blip silently dropped a package from the update plan.
* **Version age.** ``releases_behind`` quantifies "how much newer", which is
  what turns the word "modernized" into a number.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from aura.reqfile import safe_parse_version

# The PyPI that every version-age number is measured against, as YYYY-MM-DD.
#
# PyPI moves while a campaign runs, and "at latest" is relative to whenever it
# was asked. Arms of one subject run together and stay consistent, but the
# dependabot arm is collected separately -- days later -- and was marked down
# for releases published in between, having done nothing differently. Fixing
# one date puts every arm on the same index however far apart they ran.
#
# Set AURA_VERSION_AGE_AS_OF to re-measure a whole campaign on another date.
MEASURED_AS_OF = os.environ.get("AURA_VERSION_AGE_AS_OF", "2026-09-23")


# name-version(-build)?-python-abi-platform.whl
_WHEEL = re.compile(r"^.+?-[^-]+(?:-\d[^-]*)?-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$")


class IndexError_(RuntimeError):
    """Raised when the index cannot be reached after retries."""


@dataclass
class ReleaseHistory:
    name: str
    versions: tuple[str, ...]          # ascending, stable releases only
    prereleases: tuple[str, ...] = ()
    # version -> {"requires_python": str|None, "yanked": bool}
    metadata: dict = field(default_factory=dict)

    @property
    def latest(self) -> str | None:
        return self.versions[-1] if self.versions else None

    def between(self, low: str, high: str, inclusive_high: bool = True) -> list[str]:
        """Stable versions v with low < v <= high, ascending."""
        lo, hi = safe_parse_version(low), safe_parse_version(high)
        if lo is None or hi is None:
            return []
        out = []
        for raw in self.versions:
            v = safe_parse_version(raw)
            if v is None:
                continue
            if v > lo and (v <= hi if inclusive_high else v < hi):
                out.append(raw)
        return out

    def releases_behind(self, current: str) -> int:
        """How many stable releases exist above ``current``."""
        cur = safe_parse_version(current)
        if cur is None:
            return 0
        return sum(1 for raw in self.versions
                   if (v := safe_parse_version(raw)) is not None and v > cur)

    def ships_binary_wheels(self) -> bool:
        """Whether this project publishes wheels built for specific CPythons.

        A project that does is one whose releases need compiling when no wheel
        matches -- and whose older releases have no wheel for an interpreter
        that did not exist yet. A pure-Python project ships one wheel for every
        interpreter, or only a source archive, and is not affected.
        """
        return any(_WHEEL.match(f) and _WHEEL.match(f).group("py").startswith("cp")
                   for entry in self.metadata.values() for f in entry.get("files", ()))

    def has_wheel_for(self, version: str, python_version: str) -> bool:
        """Whether this release has a wheel that installs here, on Linux x86-64."""
        want = int("".join(python_version.split(".")[:2]))
        for f in (self.metadata.get(version) or {}).get("files", ()):
            m = _WHEEL.match(f)
            if not m:
                continue
            plat = m.group("plat")
            if plat != "any" and "x86_64" not in plat:
                continue
            if plat != "any" and ("linux" not in plat or "musllinux" in plat):
                continue
            for tag in m.group("py").split("."):
                if tag in ("py3", "py2.py3") or tag.startswith("py3"):
                    return True
                if tag.startswith("cp") and tag[2:].isdigit():
                    have = int(tag[2:])
                    if have == want or (m.group("abi") == "abi3" and have <= want):
                        return True
        return False

    def requires_python(self, version: str) -> str | None:
        return (self.metadata.get(version) or {}).get("requires_python")

    def is_yanked(self, version: str) -> bool:
        return bool((self.metadata.get(version) or {}).get("yanked"))

    def usable_on(self, version: str, python_version: str,
                  allow_yanked: bool = False) -> bool:
        """Can this release even be considered on this interpreter?

        Answered from index metadata alone -- no download, no virtualenv, no
        resolver call. A release whose ``requires-python`` excludes the runtime
        cannot install no matter what else is in the environment, so probing it
        is a guaranteed-wasted install-and-validate cycle.
        """
        if not allow_yanked and self.is_yanked(version):
            return False
        spec = self.requires_python(version)
        if not spec:
            return True
        try:
            return SpecifierSet(spec).contains(python_version)
        except InvalidSpecifier:
            return True   # unparseable metadata: do not exclude on a guess

    def compatible(self, candidates: list[str], python_version: str,
                   allow_yanked: bool = False) -> tuple[list[str], list[dict]]:
        """Split candidates into (worth probing, eliminated-with-reason)."""
        keep, dropped = [], []
        for version in candidates:
            if self.is_yanked(version) and not allow_yanked:
                dropped.append({"version": version, "reason": "yanked"})
                continue
            if not self.usable_on(version, python_version, allow_yanked):
                dropped.append({"version": version, "reason": "requires_python",
                                "requires_python": self.requires_python(version),
                                "runtime": python_version})
                continue
            keep.append(version)
        return keep, dropped

    def contains(self, version: str) -> bool:
        target = safe_parse_version(version)
        if target is None:
            return False
        return any(safe_parse_version(v) == target
                   for v in (*self.versions, *self.prereleases))


class PackageIndex:
    """Thread-safe, cached client for a PEP 503 simple index."""

    def __init__(self, retries: int = 3, backoff_s: float = 2.0):
        self.retries = retries
        self.backoff_s = backoff_s
        self._cache: dict[str, ReleaseHistory | None] = {}
        # (name, version) -> the release's JSON payload, or None if unknown.
        # Requires-Dist and wheel sizes both come from it, so one fetch serves both.
        self._releases: dict[tuple[str, str], dict | None] = {}
        # name -> {version: YYYY-MM-DD}, for reconstructing a past environment.
        self._dates: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()
        self._client = None
        self.lookups = 0
        self.failures = 0
        self.metadata_lookups = 0

    def _pypi(self):
        if self._client is None:
            from pypi_simple import PyPISimple
            self._client = PyPISimple()
        return self._client

    def history(self, name: str) -> ReleaseHistory | None:
        """Full release history, or None if the project is unknown to the index."""
        key = canonicalize_name(name)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        history = self._fetch(key)
        with self._lock:
            self._cache[key] = history
        return history

    def _fetch(self, name: str) -> ReleaseHistory | None:
        import time

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                self.lookups += 1
                page = self._pypi().get_project_page(name)
                if page is None or not page.packages:
                    return None
                stable, pre = set(), set()
                metadata: dict[str, dict] = {}
                for pkg in page.packages:
                    if not pkg.version:
                        continue
                    parsed = safe_parse_version(pkg.version)
                    if parsed is None:      # non-PEP 440 fossil
                        continue
                    (pre if parsed.is_prerelease else stable).add(pkg.version)
                    entry = metadata.setdefault(
                        pkg.version, {"requires_python": None, "yanked": True})
                    if getattr(pkg, "requires_python", None):
                        entry["requires_python"] = pkg.requires_python
                    if getattr(pkg, "filename", None):
                        entry.setdefault("files", []).append(pkg.filename)
                    # A version counts as yanked only if *every* file is yanked.
                    if not getattr(pkg, "is_yanked", False):
                        entry["yanked"] = False
                return ReleaseHistory(
                    name=name,
                    versions=tuple(sorted(stable, key=lambda s: safe_parse_version(s) or Version("0"))),
                    prereleases=tuple(sorted(pre, key=lambda s: safe_parse_version(s) or Version("0"))),
                    metadata=metadata,
                )
            except Exception as exc:      # transient network / index errors
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(self.backoff_s * (2 ** attempt))
        self.failures += 1
        # Distinguish "no such project" (None above) from "could not reach the
        # index", so a network blip is never silently read as 'no update'.
        raise IndexError_(f"index lookup failed for {name!r}: {last_error}")

    # ------------------------------------------------------------ shorthands

    def latest(self, name: str) -> str | None:
        h = self.history(name)
        return h.latest if h else None

    def versions_between(self, name: str, low: str, high: str) -> list[str]:
        h = self.history(name)
        return h.between(low, high) if h else []

    def compatible_versions(self, name: str, candidates: list[str],
                            python_version: str,
                            allow_yanked: bool = False) -> tuple[list[str], list[dict]]:
        """Metadata-only pre-filter; see ReleaseHistory.compatible."""
        history = self.history(name)
        if history is None:
            return list(candidates), []
        return history.compatible(candidates, python_version, allow_yanked)

    def release(self, name: str, version: str) -> dict | None:
        """One release's metadata payload, cached. ``None`` if the index is silent."""
        key = (canonicalize_name(name), version)
        with self._lock:
            if key in self._releases:
                return self._releases[key]

        payload = self._fetch_release(key[0], version)
        with self._lock:
            self._releases[key] = payload
        return payload

    def requires_dist(self, name: str, version: str) -> list[str] | None:
        """Declared dependencies of one release, as raw PEP 508 strings.

        This is the coupling that v1 rediscovered by failing: torchvision 0.13.0
        states ``torch==1.12.0`` in its own metadata, so the conflict is knowable
        before a single probe. ``None`` means the index would not tell us, which
        callers must treat as "no information", never as "no constraints".
        """
        payload = self.release(name, version)
        if payload is None:
            return None
        return list(payload.get("info", {}).get("requires_dist") or [])

    def wheel_size_mb(self, name: str, version: str) -> float | None:
        """Median wheel size for a release, in MB.

        The basis for how many probes may run at once: a batch of k candidates
        needs k installed copies on disk, so the batch width has to come from a
        measured footprint rather than a guessed constant. Median across the
        release's wheels, because one release ships many platform wheels of
        similar size and the outliers are debug or source builds.
        """
        payload = self.release(name, version)
        if payload is None:
            return None
        sizes = sorted(u.get("size") or 0 for u in payload.get("urls", [])
                       if u.get("packagetype") == "bdist_wheel" and u.get("size"))
        if not sizes:
            sizes = sorted(u.get("size") or 0 for u in payload.get("urls", [])
                           if u.get("size"))
        if not sizes:
            return None
        return round(sizes[len(sizes) // 2] / 1e6, 3)

    def _fetch_release(self, name: str, version: str) -> dict | None:
        import json
        import time
        import urllib.error
        import urllib.request

        url = f"https://pypi.org/pypi/{name}/{version}/json"
        for attempt in range(self.retries):
            try:
                self.metadata_lookups += 1
                with urllib.request.urlopen(url, timeout=30) as fh:
                    return json.load(fh)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:          # release exists but has no JSON entry
                    return None
            except Exception:
                pass
            if attempt < self.retries - 1:
                time.sleep(self.backoff_s * (2 ** attempt))
        self.failures += 1
        return None

    # ------------------------------------------------- when a release appeared

    def upload_dates(self, name: str) -> dict[str, str]:
        """``{version: YYYY-MM-DD}`` for every release of this project.

        Needed to reconstruct the environment a project had when it was last
        touched: a requirements file that pins some dependencies and not others
        stops working as the unpinned ones move on, and the honest repair is the
        set of versions that existed at the time, not a set we choose.

        One project-level request serves every version. PyPI's per-release JSON
        is the fallback, and is only reached for projects whose project page
        omits the release listing.
        """
        key = canonicalize_name(name)
        with self._lock:
            if key in self._dates:
                return self._dates[key]
        payload = self._project_json(key) or {}
        dates: dict[str, str] = {}
        for version, files in (payload.get("releases") or {}).items():
            stamps = [f.get("upload_time_iso_8601") or f.get("upload_time")
                      for f in files or [] if f.get("upload_time_iso_8601")
                      or f.get("upload_time")]
            if stamps:
                # The first file uploaded is when the release became usable.
                dates[version] = min(stamps)[:10]
        with self._lock:
            self._dates[key] = dates
        return dates

    def days_behind(self, name: str, current: str) -> int:
        """How old the pinned release is, in days, against the newest one.

        Release counts are not comparable between projects: sixteen releases of
        one package can span a year and fifteen of another five years, so
        adding them up says little about how stale an environment is. Time is
        the same unit everywhere.
        """
        dates = self.upload_dates(name)
        history = self.history(name)
        if not dates or history is None or not history.latest:
            return 0
        here, newest = dates.get(current), dates.get(history.latest)
        if not here or not newest:
            return 0
        from datetime import date
        try:
            a = date.fromisoformat(here)
            b = date.fromisoformat(newest)
        except ValueError:
            return 0
        return max((b - a).days, 0)

    def newest_at(self, name: str, on_or_before: str,
                  python_version: str | None = None,
                  specifier: str | None = None,
                  prefer_wheels: bool = False) -> str | None:
        """The newest stable release published on or before ``on_or_before``.

        Yanked releases are skipped: a release withdrawn by its author is not
        what a working environment would have been built from, and pip will not
        install it without being asked by exact version.

        ``specifier`` restricts the answer to what the project itself allowed.
        A file saying ``flask<2`` has made a decision, and reconstructing its
        environment with Flask 2 would replace that decision with ours.
        """
        history = self.history(name)
        if history is None or not history.versions:
            return None
        dates = self.upload_dates(name)
        candidates = [v for v in history.versions
                      if dates.get(v) and dates[v] <= on_or_before
                      and not (history.metadata.get(v) or {}).get("yanked")]
        if python_version:
            candidates, _ = self.compatible_versions(name, candidates, python_version)
        if specifier:
            try:
                allowed = SpecifierSet(specifier)
                candidates = [v for v in candidates if allowed.contains(v)]
            except InvalidSpecifier:
                pass
        if not candidates:
            return None
        best = max(candidates, key=lambda v: safe_parse_version(v) or Version("0"))
        if prefer_wheels and python_version and history.ships_binary_wheels() \
                and not history.has_wheel_for(best, python_version):
            # The release of that date has no wheel for this interpreter --
            # typically because the interpreter did not exist yet -- so pip
            # would compile it, and for an old C extension that fails. The
            # nearest later release that does have one is the environment of
            # that date as closely as this interpreter can hold it: dated back
            # to April 2019 on Python 3.8, negativei2-server's grpcio had to
            # compile, and failed. Only when asked: the step can cross a major
            # version (msgpack 0.6.1 -> 1.0.0), so it is a later rung of the
            # baseline ladder, tried once the date-exact environment has failed.
            floor = safe_parse_version(best) or Version("0")
            later = [v for v in history.versions
                     if (safe_parse_version(v) or Version("0")) > floor
                     and not (history.metadata.get(v) or {}).get("yanked")
                     and history.usable_on(v, python_version)
                     and history.has_wheel_for(v, python_version)]
            if specifier:
                try:
                    allowed = SpecifierSet(specifier)
                    later = [v for v in later if allowed.contains(v)]
                except InvalidSpecifier:
                    pass
            if later:
                return min(later, key=lambda v: safe_parse_version(v) or Version("0"))
        return best

    def _project_json(self, name: str) -> dict | None:
        import json
        import time
        import urllib.error
        import urllib.request

        url = f"https://pypi.org/pypi/{name}/json"
        for attempt in range(self.retries):
            try:
                self.metadata_lookups += 1
                with urllib.request.urlopen(url, timeout=30) as fh:
                    return json.load(fh)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
            except Exception:
                pass
            if attempt < self.retries - 1:
                time.sleep(self.backoff_s * (2 ** attempt))
        self.failures += 1
        return None

    def version_exists(self, name: str, version: str) -> bool:
        """The hallucination check: is this an actual release of this project?"""
        h = self.history(name)
        return h.contains(version) if h else False

    def releases_behind(self, name: str, current: str) -> int:
        h = self.history(name)
        return h.releases_behind(current) if h else 0


def _releases_behind_at(index: PackageIndex, history, name: str,
                        version: str, as_of: str) -> int | None:
    """Releases above ``version`` that existed on ``as_of``.

    Falls back to counting them all when the index cannot date them, which is
    the live answer and the one this returned before dates were considered.
    """
    if not as_of:
        return history.releases_behind(version)
    try:
        dates = index.upload_dates(name)
    except Exception:
        return history.releases_behind(version)
    if not dates:
        return history.releases_behind(version)
    current = safe_parse_version(version)
    if current is None:
        return 0
    above = 0
    for raw in history.versions:
        parsed = safe_parse_version(raw)
        if parsed is None or parsed <= current:
            continue
        when = dates.get(raw)
        # A release the index cannot date is counted: dropping it would make an
        # undatable package look more modern than it is.
        if when is None or when <= as_of:
            above += 1
    return above


def version_age_report(
    index: PackageIndex, pins: dict[str, str], as_of: str = ""
) -> dict[str, object]:
    """Quantify 'how modernized' a pinned environment is.

    Reported before and after a run so "modernized" is a number rather than a
    word: mean/median releases behind the newest stable release, and how many
    packages sit exactly at latest.

    ``as_of`` (``YYYY-MM-DD``) is the PyPI this is measured against: releases
    published after it are not counted, so "at latest" means latest on that
    date. Without it the answer is relative to the moment of the call, and two
    arms of the same subject measured days apart are not comparable -- the one
    measured later is marked down for releases that appeared in between, having
    done nothing differently. Between two collections a day apart, and with no
    package changed, mean-review-collector's dependabot arm fell from 26.32% to
    21.05% this way. Every arm of a campaign should pass the same date.
    """
    as_of = as_of or MEASURED_AS_OF
    per_package: dict[str, int] = {}
    at_latest = 0
    unknown: list[str] = []
    for name, version in pins.items():
        try:
            history = index.history(name)
        except IndexError_:
            unknown.append(name)
            continue
        if history is None or not history.versions:
            unknown.append(name)
            continue
        behind = _releases_behind_at(index, history, name, version, as_of)
        if behind is None:
            unknown.append(name)
            continue
        per_package[name] = behind
        if behind == 0:
            at_latest += 1

    values = sorted(per_package.values())
    n = len(values)
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
        "unknown": sorted(unknown),
    }
