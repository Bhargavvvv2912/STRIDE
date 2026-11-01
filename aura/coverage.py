"""How much did the oracle actually look at?

AURA's claim after an upgrade is "validation passed". Reviewer B's objection is
that this only means *no failure was observed for the covered executions*, and a
validation exercising 5% of a project supports a far weaker claim than one
exercising 80% -- yet the two are reported identically.

This module measures the difference, at two levels:

* **Repo coverage** -- the ordinary number. How much of the subject's own code
  ran during validation.
* **Dependency coverage** -- the one that matters here, and the one nobody in
  this space reports. When upgrading numpy, what matters is not how much of the
  *repo* ran but how much of *numpy* the repo actually touched. A validation
  that covers 90% of the repo while calling three numpy functions gives almost
  no evidence about a numpy upgrade.

Coverage is a property of the validation script, not of the candidate version:
whether the script reaches 312 numpy functions does not change when numpy 1.24
becomes 1.26. So it is measured twice per subject -- once at baseline, once on
the accepted result -- rather than on every probe, and the instrumentation cost
stays negligible against a search that installs hundreds of candidates.

The two measurements are not compared for equality. A *drop* is the interesting
signal: it means a code path stopped executing, which usually indicates an
optional import now fails and a fallback silently took over. Tests still pass,
and the numeric-drift check would not see it.

One limitation to state rather than bury: ``coverage.py`` sees Python lines
only. numpy, torch and scipy are largely compiled, so for those this measures
the Python wrapper layer and understates what ran. Packages shipping extension
modules are flagged ``compiled`` so the number is read correctly instead of
being taken at face value.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aura.naming import import_to_distribution


@dataclass
class PackageCoverage:
    """Executed-line counts for one measured unit (the repo, or one dependency)."""

    name: str
    statements: int = 0
    covered: int = 0
    files: int = 0
    compiled: bool = False      # ships .so/.pyd; line coverage understates it

    @property
    def pct(self) -> float | None:
        return round(self.covered / self.statements, 4) if self.statements else None


@dataclass
class CoverageReport:
    repo: PackageCoverage | None = None
    dependencies: dict[str, PackageCoverage] = field(default_factory=dict)
    ok: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "repo": ({**asdict(self.repo), "pct": self.repo.pct}
                     if self.repo else None),
            "dependencies": {n: {**asdict(c), "pct": c.pct}
                             for n, c in sorted(self.dependencies.items())},
        }


def _run(command: list[str], cwd: str | None, timeout_s: int) -> tuple[str, int]:
    try:
        proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout_s)
        return (proc.stdout or "") + (proc.stderr or ""), proc.returncode
    except subprocess.TimeoutExpired:
        return "coverage run timed out", 124
    except Exception as exc:                       # pragma: no cover - defensive
        return f"coverage run failed: {exc}", 1


def site_packages(python_executable: str) -> Path | None:
    out, code = _run([python_executable, "-c",
                      "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                     None, 120)
    return Path(out.strip()) if code == 0 and out.strip() else None


def dependency_directories(python_executable: str,
                           distributions: list[str]) -> dict[str, list[Path]]:
    """Map distribution names to the directories their code lives in.

    A distribution's import name is often not its install name (``scikit-learn``
    imports as ``sklearn``), so this asks the environment rather than guessing.
    """
    if not distributions:
        return {}
    script = (
        "import json,importlib.metadata as m\n"
        "out={}\n"
        "for d in m.distributions():\n"
        "    n=(d.metadata['Name'] or '')\n"
        "    if not n: continue\n"
        "    tops=set()\n"
        "    raw=d.read_text('top_level.txt') or ''\n"
        "    tops.update(x.strip() for x in raw.splitlines() if x.strip())\n"
        "    for f in (d.files or []):\n"
        "        parts=str(f).split('/')\n"
        "        if len(parts)>1 and parts[0].endswith('.dist-info'): continue\n"
        "        if len(parts)>1 and not parts[0].startswith('..'): tops.add(parts[0])\n"
        "    base=str(d.locate_file(''))\n"
        "    out[n.lower().replace('_','-')]=[base,sorted(t for t in tops if t)]\n"
        "print(json.dumps(out))"
    )
    out, code = _run([python_executable, "-c", script], None, 300)
    if code != 0:
        return {}
    try:
        table = json.loads(out.strip().splitlines()[-1])
    except Exception:
        return {}

    found: dict[str, list[Path]] = {}
    for dist in distributions:
        key = dist.lower().replace("_", "-")
        entry = table.get(key)
        if not entry:
            continue
        base, tops = Path(entry[0]), entry[1]
        dirs = [base / t for t in tops if (base / t).is_dir()]
        if dirs:
            found[dist] = dirs
    return found


def _is_compiled(dirs: list[Path]) -> bool:
    for directory in dirs:
        for pattern in ("*.so", "*.pyd"):
            if any(directory.rglob(pattern)):
                return True
    return False


def measure(python_executable: str, command: list[str], *, sources: list[Path],
            cwd: str | None, workdir: Path, timeout_s: int = 3600
            ) -> tuple[dict | None, str]:
    """Run ``command`` under coverage and return the parsed JSON report."""
    if not sources:
        return None, "no sources to measure"
    data_file = Path(workdir) / ".aura_coverage"
    json_file = Path(workdir) / "aura_coverage.json"
    for stale in (data_file, json_file):
        stale.unlink(missing_ok=True)

    source_arg = ",".join(str(s) for s in sources)
    instrumented = [python_executable, "-m", "coverage", "run",
                    f"--data-file={data_file}", f"--source={source_arg}"]
    # `-m pytest ...` must stay a module invocation under coverage.
    instrumented += command[1:] if command[1:2] == ["-m"] else command[1:]
    output, _ = _run(instrumented, cwd, timeout_s)

    report_out, code = _run(
        [python_executable, "-m", "coverage", "json",
         f"--data-file={data_file}", "-o", str(json_file)], cwd, 600)
    if code != 0 or not json_file.exists():
        return None, f"coverage json failed: {(report_out or output)[-400:]}"
    try:
        return json.loads(json_file.read_text()), ""
    except Exception as exc:
        return None, f"unreadable coverage json: {exc}"


def build_report(payload: dict, *, repo_root: Path,
                 dependency_dirs: dict[str, list[Path]]) -> CoverageReport:
    """Bucket measured files into the repo and each dependency."""
    report = CoverageReport(ok=True)
    repo = PackageCoverage(name="repo")
    buckets = {name: PackageCoverage(name=name, compiled=_is_compiled(dirs))
               for name, dirs in dependency_dirs.items()}
    owners = [(name, [d.resolve() for d in dirs])
              for name, dirs in dependency_dirs.items()]
    root = repo_root.resolve()

    for filename, entry in (payload.get("files") or {}).items():
        summary = entry.get("summary", {})
        statements = summary.get("num_statements", 0)
        covered = summary.get("covered_lines", 0)
        path = Path(filename)
        if not path.is_absolute():
            path = (root / path)
        path = path.resolve()

        target = None
        for name, dirs in owners:
            if any(path == d or d in path.parents for d in dirs):
                target = buckets[name]
                break
        if target is None:
            try:
                path.relative_to(root)
            except ValueError:
                continue              # neither the repo nor a tracked dependency
            target = repo
        target.statements += statements
        target.covered += covered
        target.files += 1

    report.repo = repo if repo.files else None
    report.dependencies = {n: c for n, c in buckets.items() if c.files}
    return report


def measure_validation(env, validator, *, repo_root: Path, dependencies: list[str],
                       workdir: Path, timeout_s: int = 3600) -> CoverageReport:
    """Install coverage into an already-built probe env and measure validation.

    Deliberately a *separate* run from the one that decided anything. Installing
    ``coverage`` into the environment first and then validating would put an
    extra package into the very resolution under test; measuring afterwards
    cannot change a verdict that has already been reached.
    """
    command = validator.command(env.python)
    if command is None:
        return CoverageReport(ok=False, reason="validation runs no code")

    output, code = env.pip_install(["coverage>=7.0"])
    if code != 0:
        return CoverageReport(ok=False,
                              reason=f"could not install coverage: {output[-300:]}")

    dep_dirs = dependency_directories(env.python, dependencies)
    missing = sorted(set(dependencies) - set(dep_dirs))
    sources = [repo_root] + [d for dirs in dep_dirs.values() for d in dirs]

    payload, error = measure(env.python, command, sources=sources,
                             cwd=validator.cwd, workdir=workdir, timeout_s=timeout_s)
    if payload is None:
        return CoverageReport(ok=False, reason=error)

    report = build_report(payload, repo_root=repo_root, dependency_dirs=dep_dirs)
    if missing:
        report.reason = "not located in the environment: " + ", ".join(missing)
    return report


def compare(baseline: CoverageReport, final: CoverageReport,
            tolerance: float = 0.02) -> dict:
    """What changed between the baseline and accepted measurements.

    Equality is not the point; a *drop* is. Less code running after an upgrade
    usually means an optional import started failing and a fallback took over
    silently -- the tests still pass and the numbers still match, so nothing
    else in the pipeline would notice.
    """
    out = {"dropped": [], "repo_delta": None, "dependency_deltas": {}}
    if baseline.repo and final.repo and baseline.repo.pct is not None \
            and final.repo.pct is not None:
        delta = round(final.repo.pct - baseline.repo.pct, 4)
        out["repo_delta"] = delta
        if delta < -tolerance:
            out["dropped"].append({"unit": "repo", "before": baseline.repo.pct,
                                   "after": final.repo.pct, "delta": delta})
    for name, before in baseline.dependencies.items():
        after = final.dependencies.get(name)
        if after is None or before.pct is None or after.pct is None:
            continue
        delta = round(after.pct - before.pct, 4)
        out["dependency_deltas"][name] = delta
        if delta < -tolerance:
            out["dropped"].append({"unit": name, "before": before.pct,
                                   "after": after.pct, "delta": delta})
    return out
