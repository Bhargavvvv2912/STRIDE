"""Human-readable run log, written alongside the machine-readable JSONL.

v1's console output was genuinely good for following a run -- the prioritised
plan table, the per-package progress lines -- and v2 initially dropped all of it
in favour of JSONL. JSONL is right for generating tables; it is useless for
watching a six-hour run or working out afterwards why a package was rejected.

Both streams are written from the same events by `RunLog`, so they cannot drift
apart. The text file is safe to `tail -f`.
"""

from __future__ import annotations

import re

import sys
from pathlib import Path
from typing import TextIO

RULE = "=" * 78
THIN = "-" * 78

_TIER_LABEL = {
    "baseline": "BASE  ", "linear": "LINEAR", "greedy": "GREEDY",
    "llm": "LLM   ", "merge": "MERGE ", "pr": "PR    ", "all": "ALL   ",
    "": "      ",
}


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _plan(plan: dict) -> str:
    if not plan:
        return "<baseline: the project's own requirements>"
    items = [f"{k}=={v}" for k, v in sorted(plan.items())]
    text = " ".join(items)
    return text if len(text) <= 58 else f"{items[0]} +{len(items) - 1} more"


def _counts(counts: dict) -> str:
    """'13 passed, 6 failed' from pytest's summary counts, in a stable order."""
    order = ("passed", "failed", "error", "errors", "skipped", "xfailed", "xpassed")
    parts = [f"{counts[k]} {k}" for k in order if counts.get(k)]
    return ", ".join(parts)


_REQ_FILE_REF = re.compile(r"-r \S*?(?:_requirements|requirements)[^\s()]*\s*\(line (\d+)\)")
_ABS_PATH = re.compile(r"(?:/[^\s/]+){3,}/([^\s/]+)")


def _first_error_line(text: str) -> str:
    """The most informative line of an error dump, for a one-line summary.

    Workspace paths are shortened: "-r /Users/.../.aura_probe_4_requirements.txt
    (line 4)" says nothing a reader needs beyond "requirement line 4".
    """
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(("ERROR:", "E   ", "error:")) or "Error:" in stripped:
            stripped = _REQ_FILE_REF.sub(r"requirement line \1", stripped)
            stripped = _ABS_PATH.sub(r"\1", stripped)
            return stripped[:110]
    return ""


class ConsoleWriter:
    """Formats run events as readable text into a file (and optionally stdout)."""

    def __init__(self, path: str | Path | None, echo: bool = True):
        self.path = Path(path) if path else None
        self.echo = echo
        self._fh: TextIO | None = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # One run per file. The run log archives the previous file before
            # this opens it, so nothing is lost by starting clean.
            self._fh = self.path.open("w", encoding="utf-8")
        self._package_started: dict[str, float] = {}
        self._last_couples: list | None = None
        self._newest_rejected: str | None = None
        # A later pass that only re-confirms earlier answers is collapsed to a
        # line per package, so its lines are held until the package finishes.
        self._held: list[str] | None = None
        self._seen_packages: set[str] = set()

    def write(self, line: str = "") -> None:
        if self._held is not None:
            self._held.append(line)
            return
        self._emit(line)

    def _emit(self, line: str) -> None:
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()          # so `tail -f` keeps up and kills survive
        if self.echo:
            print(line, flush=True)

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    # ------------------------------------------------------------- formatting

    def handle(self, event: str, record: dict) -> None:
        handler = getattr(self, f"_on_{event}", None)
        if handler:
            handler(record)

    def _on_run_start(self, r: dict) -> None:
        # The readable log carries only what a reader needs to follow the run.
        # Everything else -- timings, seeds, installer flags, commits, host --
        # stays in the .jsonl beside it, which is what the paper is computed from.
        config, env = r.get("config", {}) or {}, r.get("env", {}) or {}
        self.write(RULE)
        self.write(f"AURA  {config.get('repo', '?')}   arm={config.get('arm', '?')}")
        self.write(RULE)
        notes = config.get("notes") or ""
        if notes.startswith("github-actions:"):
            # Run on GitHub, not on this machine: say where, not what ran here.
            self.write(f"  ran on   : GitHub ({notes.split(':', 1)[1]}), "
                       f"Dependabot + GitHub Actions")
            self.write(f"  python   : {config.get('python_version')} (in CI)")
            self.write("")
            return
        self.write(f"  tiers    : {' -> '.join(config.get('tiers', [])) or 'none'}")
        if config.get("llm_model") and config.get("llm_model") != "none":
            self.write(f"  model    : {config['llm_model'].split(':', 1)[-1]}")
        self.write(f"  python   : {env.get('python')}")
        self.write("")

    def _on_pass_start(self, r: dict) -> None:
        self.write("")
        self.write(RULE)
        self.write(f"PASS {r.get('pass_num')}")
        self.write(RULE)

    def _on_prioritization(self, r: dict) -> None:
        queue = r.get("queue", [])
        self.write(f"{len(queue)} package(s) behind latest, in the order they will be tried:")
        self.write("")
        self.write(f"  {'#':>3}  {'package':<30} {'current':>14} -> latest")
        self.write(f"  {'-' * 3}  {'-' * 30} {'-' * 14}    {'-' * 14}")
        for i, item in enumerate(queue, 1):
            self.write(f"  {i:>3}  {item['package']:<30} {item['current']:>14} -> "
                       f"{item['target']}")
        self.write("")

    def _on_attempt(self, r: dict) -> None:
        package, tier = r.get("package", "?"), r.get("tier", "")

        label = _TIER_LABEL.get(tier) or tier.upper()[:6].ljust(6)
        in_round = r.get("round") is not None
        plan = _plan(r.get("plan", {}))
        # One-shot arms probe twice: the project's own pins first, as the
        # reference the upgrade is judged against, then the upgrade itself.
        role = {"<reference>": "original pins", "<arm>": "upgraded",
                "<baseline>": "original pins"}.get(package)
        if role:
            plan = f"[{role}] {plan}"
        if r.get("merge_round") is not None:
            version = (r.get("plan") or {}).get(package, "?")
            plan = (f"{package}=={version}   (on top of {r.get('merged_before', 0)} "
                    f"merged)")
        if r.get("batch_size"):
            plan += f"   (one of {r['batch_size']} tried in parallel)"
        elif r.get("confirming"):
            plan += "   (newest that passed, re-tested alone to confirm)"
        if not in_round:                   # a greedy round printed it already
            self.write(f"  {label}  {plan}")

        pad = " " * 10

        # -- installation -------------------------------------------------
        if not r.get("install_ok"):
            reason = r.get("error_class") or "error"
            self.write(f"{pad}installation : FAILED ({reason})")
            line = _first_error_line(r.get("error_excerpt", ""))
            if line:
                self.write(f"{pad}               {line}")
            self.write(f"{pad}validation   : not run (nothing to test)")
            self.write(f"{pad}result       : REJECTED")
            return
        self.write(f"{pad}installation : ok")

        # -- validation ---------------------------------------------------
        if r.get("setup_failed"):
            self.write(f"{pad}validation   : not run (environment setup failed)")
            self.write(f"{pad}result       : REJECTED")
            return
        counts = _counts(r.get("tests") or {})
        new = r.get("new_failures") or []
        failing = r.get("failing") or []
        if r.get("validate_ok"):
            status = "passed"
        elif failing and not new and r.get("accepted"):
            status = "passed (only failures already present at baseline)"
        else:
            status = "FAILED"
        detail = f"{status}{'  ' + counts if counts else ''}"
        self.write(f"{pad}validation   : {detail}")
        shown = new or ([] if r.get("validate_ok") or r.get("accepted") else failing)
        startup = [x for x in shown if x.startswith("tests could not start")]
        if startup:
            # Not a failing test: the suite never ran. Show why.
            self.write(f"{pad}               {startup[0][:140]}")
            shown = []
        if shown:
            head = "newly failing" if new else "failing"
            self.write(f"{pad}               {head}: {', '.join(shown[:3])}"
                       + (f"  (+{len(shown) - 3} more)" if len(shown) > 3 else ""))
        elif status == "FAILED" and not startup:
            why = r.get("validation_reason") or _first_error_line(
                r.get("error_excerpt", ""))
            if why:
                self.write(f"{pad}               {why[:100]}")
        if r.get("drifted"):
            self.write(f"{pad}               numbers moved: {', '.join(r['drifted'])}")

        # -- verdict ------------------------------------------------------
        if r.get("accepted"):
            self.write(f"{pad}result       : ACCEPTED")
        else:
            why = {"behavioural_drift": "tests pass but results changed",
                   "validation_failed": "tests fail"}.get(r.get("error_class"),
                                                           r.get("error_class") or "")
            self.write(f"{pad}result       : REJECTED" + (f" ({why})" if why else ""))

    def _on_proposal(self, r: dict) -> None:
        if r.get("pr") is None:
            return
        verdict = "merged" if r.get("accepted") else r.get("pr_state", "open")
        ci = {"success": "CI passed", "failure": "CI failed"}.get(
            r.get("ci"), f"CI {r.get('ci') or 'did not finish'}")
        move = f"{r.get('package')} {r.get('from_version')} -> {r.get('to_version')}"
        self.write(f"  PR #{r['pr']:<4} {move:<40} {ci:<12} {verdict}")
        if r.get("reason"):
            self.write(f"            {r['reason'][:130]}")

    def _on_llm_call(self, r: dict) -> None:
        proposed = r.get("proposed", [])
        rejected = r.get("rejected", [])
        self.write(f"  LLM     request #{r.get('iteration')} to {r.get('model')}")
        if r.get("explanation"):
            self.write(f"          reasoning: {r['explanation'][:200]}")
        if proposed:
            self.write(f"          proposes : {' '.join(proposed)}")
        for item in rejected:
            self.write(f"          REJECTED : {item.get('req')}  [{item.get('reason')}]")
        if not r.get("resolvable", True):
            self.write("          model declared the conflict unresolvable; stopping early")

    def _on_conflict_detection(self, r: dict) -> None:
        # Recorded for the extraction-quality figures; the reason a blocker was
        # added is shown on its greedy round instead.
        return

    def _on_behavior_check(self, r: dict) -> None:
        self.write(f"          DRIFT    : {', '.join(r.get('drifted', []))} "
                   f"moved beyond tolerance {r.get('tolerance')}")
        for key in r.get("drifted", [])[:5]:
            base = (r.get("baseline") or {}).get(key)
            cand = (r.get("candidate") or {}).get(key)
            self.write(f"                     {key}: {base} -> {cand}")

    def _on_constraint_graph(self, r: dict) -> None:
        if r.get("phase") == "closure":
            added = [a for a in r.get("additions") or [] if a.get("resolved")]
            stuck = [a for a in r.get("additions") or [] if not a.get("resolved")]
            if added:
                self.write("          also needs: "
                           "(from the declared requirements of the versions above)")
                for a in added:
                    self.write(f"                     {a['package']}=={a['resolved']:<12}"
                               f" because {a['required_by']} needs {a['requirement']}")
            for a in stuck:
                self.write(f"          cannot meet: {a['requirement']} "
                           f"(required by {a['required_by']}; {a['why']})")
            return
        if r.get("phase") == "co_update_plan":
            # Metadata only suggests blocker versions now; it no longer adds
            # packages, so a "solved" line here would overstate its role.
            reason = r.get("reason", "")
            if reason.startswith("no_unlocking_version"):
                self.write(f"          proof    : nothing unlocks "
                           f"{r.get('package')}=={r.get('target')} "
                           f"({reason.split(':')[-1]} has no compatible release)")
            return
        couples = r.get("couples") or []
        # Recomputed before every package, but only worth a line when it changes.
        if couples and couples != self._last_couples:
            self._last_couples = couples
            self.write("  pinned packages that constrain each other: "
                       + ", ".join(" <-> ".join(pair) for pair in couples[:8]))

    def _on_coverage(self, r: dict) -> None:
        phase = r.get("phase", "")
        if phase == "delta":
            for drop in r.get("dropped") or []:
                self.write(f"  ** COVERAGE DROP: {drop['unit']} "
                           f"{drop['before']:.1%} -> {drop['after']:.1%} "
                           f"(less code ran than at baseline)")
            return
        if not r.get("ok"):
            self.write(f"  coverage ({phase}): unavailable - {r.get('reason','')}")
            return
        repo = (r.get("repo") or {}).get("pct")
        self.write(f"  coverage ({phase}): repo "
                   + (f"{repo:.1%}" if repo is not None else "n/a"))
        for name, entry in sorted((r.get("dependencies") or {}).items()):
            pct = entry.get("pct")
            if pct is None:
                continue
            flag = "  [compiled: understated]" if entry.get("compiled") else ""
            self.write(f"                    {name:<28} {pct:>6.1%}{flag}")

    def _on_routing(self, r: dict) -> None:
        chose = r.get("chose", "")
        if chose == "wheel_only_retry":
            verdict = "RESCUED" if r.get("rescued") else "confirmed incompatible"
            self.write(f"          routing  : build failed -> wheel-only retry "
                       f"-> {verdict}")
            return
        if chose == "greedy":
            self.write(f"          routing  : all {r.get('blocked')} older "
                       f"version(s) forbidden by a declared pin -> not scanning "
                       f"them; on to the next tier")
            for blocker in (r.get("blockers") or [])[:4]:
                self.write(f"                     {blocker}")
        else:
            self.write(f"          routing  : {r.get('permitted')} older version(s) "
                       f"still permitted -> scanning them")

    def _greedy_round(self, r: dict) -> None:
        pad = " " * 10
        added = r.get("added") or []
        verdict = "dry run ok" if r.get("dry_run") == "ok" else "dry run: rejected"
        for i, a in enumerate(added or [{}]):
            move = (f"+ {a['package']} {a['from']} -> {a['version']}" if a else
                    "no blocker added")
            label = f"  GREEDY  round {r.get('round'):<2} " if i == 0 else " " * 18
            why = a.get("because", "") if a else ""
            if a and a.get("chosen_as", "").startswith("newest version that allows"):
                why = (why + "; " if why else "") + a["chosen_as"]
            tail = f"  [{verdict}]" if i == len(added or [{}]) - 1 else ""
            self.write(f"{label}{move:<34} {why}{tail}".rstrip())
        if r.get("dry_run") == "ok":
            plan = ", ".join(f"{k}=={v}" for k, v in sorted((r.get("plan") or {}).items()))
            self.write(f"{pad}plan         : {plan}")
        elif not added and r.get("dry_run_reason"):
            self.write(f"{pad}               {r['dry_run_reason']}")

    def _package_start(self, r: dict) -> None:
        package = r.get("package", "?")
        self._newest_rejected = None
        repeat = r.get("pass_num", 1) > 1 and package in self._seen_packages
        self._seen_packages.add(package)
        header = ["", THIN, f"PACKAGE  {package}   {r.get('current')} -> "
                           f"{r.get('target')}", THIN]
        if repeat:
            self._held = list(header)
        else:
            for line in header:
                self.write(line)

    def _package_result(self, r: dict) -> None:
        package, pins = r.get("package", "?"), r.get("pins") or {}
        if pins:
            moved = ", ".join(f"{k} -> {v}" for k, v in sorted(pins.items()))
            verdict = f"  => {package}: advanced ({moved}) by {r.get('tier')}"
        else:
            verdict = f"  => {package}: not advanced, stays at {r.get('current')}"
        held, self._held = self._held, None
        if held is not None and not r.get("new_installs") and not pins:
            # Nothing new was installed for this package in this pass: every
            # plan was either ruled out by a dry run or already tried. The
            # full story is in the earlier pass and in the .jsonl.
            for line in held[:4]:
                self._emit(line)
            self._emit("  (same outcome as the previous pass; no new plan to install)")
            self._emit(verdict)
            return
        for line in held or []:
            self._emit(line)
        self.write(verdict)

    def _cached_attempt(self, r: dict) -> None:
        if r.get("round") is not None:
            # A greedy line above already shows the plan.
            self.write("          (same plan as tried earlier; result reused)")
        else:
            self.write(f"  {(_TIER_LABEL.get(r.get('tier', '')) or '').strip():<6}  "
                       f"{_plan(r.get('plan', {}))}  (same plan as tried earlier; "
                       f"result reused)")
        self._on_attempt({**r, "round": r.get("round", 0), "duration_s": 0})

    def _on_note(self, r: dict) -> None:
        note = r.get("note", "")
        if note == "package_start":
            self._package_start(r)
            return
        if note == "package_result":
            self._package_result(r)
            return
        if note == "probe_already_seen":
            self._cached_attempt(r)
            return
        if note == "proposal_phase":
            title = {"alone": "EACH BUMP ALONE", "merge": "MERGED ONE AT A TIME"}.get(
                r.get("phase"), str(r.get("phase")).upper())
            self.write("")
            self.write(f"{THIN}")
            self.write(f"{title}   ({r.get('detail', '')})")
            self.write(f"{THIN}")
            return
        if note == "merge_round":
            still = r.get("still_open") or []
            self.write(f"  round {r.get('round')}: merged {r.get('merged_now')}, "
                       f"{r.get('merged_total')} in total"
                       + (f"; still open: {', '.join(still)}" if still else ""))
            return
        if note == "github_renovate":
            self.write(f"Renovate opened {r.get('prs')} PR(s), packages released "
                       f"together grouped into one; CI ran the project's tests on "
                       f"each, and {r.get('merged')} passed and were merged.")
            self.write("")
            return
        if note == "github_dependabot":
            self.write(f"Dependabot opened {r.get('prs')} PR(s), one per dependency; "
                       f"CI ran the project's tests on each, and {r.get('merged')} "
                       f"passed and were merged.")
            self.write("")
            return
        if note == "github_merge_order":
            order = r.get("order") or []
            self.write("")
            self.write("  merged, in order: " + (", ".join(order) or "nothing"))
            return
        if note == "github_main_tested":
            verdict = {"success": "passed", "failure": "FAILED"}.get(
                r.get("conclusion"), r.get("conclusion"))
            self.write(f"  final main, tested again on GitHub: {verdict}")
            return
        if note == "baseline_stage":
            if not r.get("ok"):
                self.write(f"  BASE    {r.get('stage')}: {r.get('detail', 'failed')}")
            return
        if note == "baseline_repaired":
            moved = r.get("packages") or {}
            self.write(f"  BASE    the project's own requirements no longer work; "
                       f"rebuilt as of {r.get('on_date')} ({len(moved)} package(s) "
                       f"dated back)")
            for name, (was, now) in sorted(moved.items())[:6]:
                self.write(f"            {name:<24} {was:>12} -> {now}")
            if len(moved) > 6:
                self.write(f"            (+{len(moved) - 6} more)")
            return
        if note == "baseline_has_failures":
            tests = r.get("tests") or []
            self.write(f"  BASE    {r.get('count')} test(s) already fail before any "
                       f"upgrade; they are tolerated, the rest are the oracle")
            for name in tests[:3]:
                self.write(f"            {name}")
            return
        if note == "baseline_established":
            if r.get("stage") != "declared":
                self.write(f"  BASE    baseline: {r.get('stage')} "
                           f"[{r.get('digest')}] -- every arm starts here")
            return
        if note == "regroup_start":
            self.write("")
            self.write(RULE)
            self.write("REGROUP   the search has stopped; can a group reach further "
                       "by stepping one pin back?")
            self.write(RULE)
            self.write(f"  {r.get('behind')} release(s) behind in total; still "
                       f"behind: {', '.join(r.get('packages') or [])}")
            return
        if note == "regroup_try":
            traded = ", ".join(f"{k} {v[0]} -> {v[1]}"
                               for k, v in (r.get("traded") or {}).items())
            self.write(f"  TRADE   {r.get('package')} -> {r.get('target')}  "
                       f"by stepping back: {traded}")
            return
        if note == "regroup_accepted":
            self.write(f"          ACCEPTED: {r.get('behind')} release(s) behind "
                       f"in total now")
            return
        if note == "regroup_rejected":
            plan = ", ".join(f"{k}=={v}" for k, v in sorted((r.get("plan") or {}).items()))
            self.write(f"  TRADE   {r.get('package')}: no ({r.get('reason')})"
                       + (f"  [{plan}]" if r.get("reason") != "no closer to the "
                          "current releases" else ""))
            return
        if note == "greedy_round":
            self._greedy_round(r)
            return
        if note == "greedy_stopped":
            rounds = r.get("rounds") or 0
            self.write(f"  GREEDY  gave up after {rounds} round(s): {r.get('reason')}"
                       if rounds else f"  GREEDY  nothing to co-update: {r.get('reason')}")
            return
        if note == "greedy_start":
            plan = " ".join(f"{k}=={v}" for k, v in sorted((r.get("plan") or {}).items()))
            ok = r.get("dry_run") == "ok"
            if not ok and plan == self._newest_rejected:
                return        # the LINEAR line above already shows this rejection
            self.write(f"  GREEDY  start    {plan} alone"
                       f"  [{'dry run ok' if ok else 'dry run: rejected'}]")
            if not ok:
                why = r.get("reason") or _first_error_line(r.get("error_excerpt", ""))
                if why:
                    self.write(f"                   {why}")
            return
        if note == "linear_newest_rejected":
            self._newest_rejected = f"{r.get('package')}=={r.get('version')}"
            self.write(f"  LINEAR  {r.get('package')}=={r.get('version')}  "
                       f"[dry run: rejected, not installed]")
            why = r.get("reason") or _first_error_line(r.get("error_excerpt", ""))
            if why:
                self.write(f"          {why}")
            return
        if note == "routing_fallback_to_linear":
            self.write(f"  LINEAR  every later tier failed; scanning the "
                       f"{r.get('versions')} older version(s) routing skipped")
            return
        interesting = {
            "baseline_failed": "baseline does not install and validate",
            "baseline_repaired_by_unpinning": "baseline repaired by removing constraints",
            "llm_early_exit": f"LLM early exit: {r.get('reason', '')}",
            "llm_plan_accepted": f"LLM plan accepted on iteration {r.get('iteration')}",
            "index_lookup_failed": f"index lookup failed for {r.get('package')}",
            "dry_run_rejected": None,     # summarised per package instead
            "linear_dry_run_summary": None,       # printed as a LINEAR line
            "post_hoc_verdict": f"post-hoc validation: "
                                f"{'PASSED' if r.get('passed') else 'FAILED'}",
            "agent_crashed": f"CRASHED: {r.get('error', '')}",
            "routing_fallback_rescued": None,
            "constraint_graph_failed": f"constraint graph unavailable: "
                                       f"{r.get('error', '')}",
            "target_demoted": f"{r.get('package')}: newest installable version is "
                              f"{r.get('target')}, not {r.get('latest')}",
            "no_usable_versions": f"{r.get('package')}: no version above "
                                  f"{r.get('current')} is installable here",
            "metadata_prefilter": None,   # counted in the summary, too noisy here
        }
        if note == "linear_dry_run_summary":
            rejected = r.get("rejected") or []
            self.write(f"  LINEAR  {len(rejected)} of {r.get('considered')} older "
                       f"version(s) ruled out by a resolver dry run, no install needed")
            return
        message = interesting.get(note, None)
        if message:
            self.write(f"  ** {message}")

    def _on_pass_end(self, r: dict) -> None:
        self.write("")
        self.write(f"PASS {r.get('pass_num')} complete: {r.get('changed')} change(s)"
                   + ("  [converged]" if r.get("converged") else ""))
        self._package_started.clear()

    def _on_run_end(self, r: dict) -> None:
        counters = r.get("counters", {}) or {}
        age = r.get("version_age", {}) or {}
        before, after = age.get("before", {}) or {}, age.get("after", {}) or {}
        self.write("")
        self.write(RULE)
        self.write(f"RESULT  {r.get('outcome', '?').upper()}"
                   + (f"   ({r['reason']})" if r.get("reason") else ""))
        self.write(RULE)
        updated = r.get("updated", {}) or {}
        if updated:
            self.write(f"  {len(updated)} package(s) advanced:")
            for name, (old, new) in sorted(updated.items()):
                self.write(f"    {name:<32} {old:>16} -> {new}")
        else:
            self.write("  no package advanced; the original pins are kept")
        if r.get("unchanged"):
            self.write(f"  could not be advanced: {', '.join(r['unchanged'])}")
        if before and after:
            self.write("")
            self.write(f"  packages at their latest version: "
                       f"{before.get('at_latest_pct')}% -> {after.get('at_latest_pct')}%")
        if counters.get("proposals_validated"):
            self.write(f"  bumps that passed alone      : "
                       f"{int(counters.get('proposals_accepted', 0))} of "
                       f"{int(counters['proposals_validated'])}")
            self.write(f"  merged one at a time (result): "
                       f"{int(counters.get('proposals_merged', 0))}, over "
                       f"{int(counters.get('merge_rounds', 0))} round(s)")
        if counters.get("llm_calls"):
            self.write(f"  LLM requests: {int(counters['llm_calls'])}")
        self.write(RULE)
