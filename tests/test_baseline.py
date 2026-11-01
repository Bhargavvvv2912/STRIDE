"""Establishing a baseline, including repairing one that no longer works.

The repair exists so that projects which pin some dependencies and not others
-- the majority -- are usable as subjects. It must be deterministic: every arm
and every model has to start from the identical environment, or their results
are not comparable, and what the repair moved must stay separate from what the
search later advances.
"""

from types import SimpleNamespace

import pytest

from aura.baseline import TEST_TOOLING, Baseline, establish, reconstruct
from aura.reqfile import ReqFile


class FakeIndex:
    """Release dates, as PyPI would report them."""

    def __init__(self, dates: dict[str, dict[str, str]]):
        self.dates = dates

    def newest_at(self, name, on_or_before, python_version=None, specifier=None,
                  prefer_wheels=False):
        from packaging.specifiers import SpecifierSet
        releases = self.dates.get(name, {})
        eligible = [v for v, d in releases.items() if d <= on_or_before]
        if specifier:
            eligible = [v for v in eligible if SpecifierSet(specifier).contains(v)]
        if not eligible:
            return None
        return max(eligible, key=lambda v: tuple(int(p) for p in v.split(".")))


INDEX = FakeIndex({
    "jinja2": {"2.10.3": "2019-10-04", "2.11.2": "2020-04-13", "3.1.6": "2025-03-05"},
    "werkzeug": {"0.16.1": "2020-01-27", "1.0.1": "2020-03-31", "3.1.3": "2024-11-08"},
    "coverage": {"5.0.3": "2020-01-19", "7.6.1": "2024-08-04"},
})


def ok(**kw):
    return SimpleNamespace(accepted=True, error_class="", **kw)


def bad(error="validation_failed"):
    return SimpleNamespace(accepted=False, error_class=error)


# --------------------------------------------------------------- reconstruct

def test_unpinned_transitives_are_pinned_as_they_were_at_the_commit():
    """The drift that breaks these projects is in dependencies the file never
    mentions, so the reconstruction has to reach them."""
    reqfile = ReqFile.from_text("Flask==1.1.1\n")
    closure = {"flask": "1.1.1", "jinja2": "3.1.6", "werkzeug": "3.1.3"}
    out, moved = reconstruct(reqfile, closure, index=INDEX, on_date="2020-06-01")
    assert out.pins() == {"flask": "1.1.1", "jinja2": "2.11.2", "werkzeug": "1.0.1"}
    assert moved == {"jinja2": ["3.1.6", "2.11.2"], "werkzeug": ["3.1.3", "1.0.1"]}


def test_what_the_project_pinned_itself_is_never_touched():
    """The project's own decisions are the subject of the experiment."""
    reqfile = ReqFile.from_text("Flask==1.1.1\nJinja2==2.10.3\n")
    closure = {"flask": "1.1.1", "jinja2": "2.10.3", "werkzeug": "3.1.3"}
    out, moved = reconstruct(reqfile, closure, index=INDEX, on_date="2020-06-01")
    assert out.pins()["jinja2"] == "2.10.3"       # not moved to 2.11.2
    assert set(moved) == {"werkzeug"}


def test_a_package_the_index_cannot_date_is_left_alone():
    reqfile = ReqFile.from_text("Flask==1.1.1\n")
    closure = {"flask": "1.1.1", "mystery-lib": "9.9.9"}
    out, moved = reconstruct(reqfile, closure, index=INDEX, on_date="2020-06-01")
    assert "mystery-lib" not in out.pins() and moved == {}


def test_the_reconstruction_is_the_same_every_time():
    """Arms are only comparable if they start from the identical file."""
    reqfile = ReqFile.from_text("Flask==1.1.1\n")
    closure = {"flask": "1.1.1", "jinja2": "3.1.6", "werkzeug": "3.1.3"}
    first, _ = reconstruct(reqfile, closure, index=INDEX, on_date="2020-06-01")
    second, _ = reconstruct(reqfile, closure, index=INDEX, on_date="2020-06-01")
    assert first.to_text() == second.to_text()
    assert Baseline(True, first).digest == Baseline(True, second).digest


# ------------------------------------------------------------------- ladder

def test_a_working_baseline_is_used_as_it_is():
    """No repair when none is needed: most frozen files still work."""
    reqfile = ReqFile.from_text("Flask==1.1.1\n")
    called = []
    result = establish(reqfile, probe=lambda rf: ok(),
                       closure_of=lambda rf: called.append(rf) or {},
                       index=INDEX, commit_date="2020-06-01")
    assert result.ok and result.stage == "declared" and result.repaired == {}
    assert called == [], "the closure is only needed when a repair is"


def test_a_broken_baseline_is_repaired_at_the_commit_date():
    reqfile = ReqFile.from_text("Flask==1.1.1\n")
    closure = {"flask": "1.1.1", "jinja2": "3.1.6"}

    def probe(rf):
        # The project's own file installs Jinja2 3.x and dies on import; the
        # reconstruction pins the version that existed at the time.
        return ok() if rf.pins().get("jinja2") == "2.11.2" else bad()

    result = establish(reqfile, probe=probe, closure_of=lambda rf: closure,
                       index=INDEX, commit_date="2020-06-01")
    assert result.ok and result.stage == "reconstructed"
    assert result.repaired == {"jinja2": ["3.1.6", "2.11.2"]}
    assert result.reqfile.pins()["jinja2"] == "2.11.2"


def test_test_tooling_is_only_dropped_as_a_last_resort():
    """An unbuildable linter should not cost a subject, but dropping pins is a
    bigger change than dating them and comes second."""
    reqfile = ReqFile.from_text("Flask==1.1.1\ncoverage==5.0.3\n")
    closure = {"flask": "1.1.1", "coverage": "5.0.3", "jinja2": "3.1.6"}
    seen = []

    def probe(rf):
        seen.append(sorted(rf.pins()))
        return ok() if "coverage" not in rf.pins() else bad("build_failure")

    result = establish(reqfile, probe=probe, closure_of=lambda rf: closure,
                       index=INDEX, commit_date="2020-06-01")
    assert result.ok and result.stage == "relaxed"
    assert "coverage" not in result.reqfile.pins()
    assert "jinja2" in result.reqfile.pins()       # the dating is still applied
    assert seen[0] == ["coverage", "flask"]        # declared first, unchanged


def test_a_subject_with_no_working_baseline_is_still_rejected():
    """Repair widens the pool; it must not manufacture a subject."""
    reqfile = ReqFile.from_text("Flask==1.1.1\n")
    result = establish(reqfile, probe=lambda rf: bad(),
                       closure_of=lambda rf: {"flask": "1.1.1", "jinja2": "3.1.6"},
                       index=INDEX, commit_date="2020-06-01")
    assert not result.ok and result.reqfile is None
    assert "no working baseline" in result.reason


def test_repair_can_be_switched_off():
    """The strict regime, for reporting results without repaired subjects."""
    reqfile = ReqFile.from_text("Flask==1.1.1\n")
    result = establish(reqfile, probe=lambda rf: bad(), closure_of=lambda rf: {},
                       index=INDEX, commit_date="2020-06-01", repair=False)
    assert not result.ok and "does not install and validate" in result.reason


def test_every_stage_is_reported_so_the_log_shows_what_was_tried():
    reqfile = ReqFile.from_text("Flask==1.1.1\n")
    stages = []
    establish(reqfile, probe=lambda rf: bad(),
              closure_of=lambda rf: {"flask": "1.1.1", "jinja2": "3.1.6"},
              index=INDEX, commit_date="2020-06-01",
              on_stage=lambda **kw: stages.append((kw["stage"], kw["ok"])))
    assert [s for s, _ in stages] == ["declared", "reconstructed", "relaxed"]
    assert not any(ok for _, ok in stages)


def test_the_digest_changes_when_the_file_does():
    """The digest is what lets a reader check that all arms started together."""
    a = Baseline(True, ReqFile.from_text("Flask==1.1.1\n"))
    b = Baseline(True, ReqFile.from_text("Flask==1.1.2\n"))
    assert a.digest and a.digest != b.digest
    assert Baseline(False).digest == ""


# ------------------------------- when the resolver cannot describe the project

class MetadataIndex(FakeIndex):
    """An index that also knows what each release declared it needed."""

    def __init__(self, dates, requires):
        super().__init__(dates)
        self.requires = requires

    def requires_dist(self, name, version):
        return self.requires.get((name, version))


def test_the_closure_is_walked_from_metadata_when_pip_cannot_resolve():
    """The projects most in need of repair are exactly the ones pip refuses to
    resolve: an unpinned dependency has moved somewhere incompatible with a
    pinned one, so there is no report to read."""
    from aura.baseline import closure_at_date

    index = MetadataIndex(
        {"jinja2": {"2.11.2": "2020-04-13", "3.1.6": "2025-03-05"},
         "markupsafe": {"1.1.1": "2019-02-23", "3.0.3": "2025-09-27"}},
        {("flask", "1.1.1"): ["Jinja2 (>=2.10.1)", "gunicorn ; extra == 'deploy'"],
         ("jinja2", "2.11.2"): ["MarkupSafe (>=0.23)"]})
    closure = closure_at_date(index, {"flask": "1.1.1"}, "2020-06-01", "3.9")
    assert closure == {"flask": "1.1.1", "jinja2": "2.11.2", "markupsafe": "1.1.1"}
    assert "gunicorn" not in closure, "an extra's dependencies are not requested"


def test_a_marker_for_another_python_is_not_a_dependency():
    from aura.baseline import closure_at_date

    index = MetadataIndex(
        {"importlib-metadata": {"1.5.0": "2020-01-01"}},
        {("flask", "1.1.1"): ["importlib-metadata ; python_version < '3.8'"]})
    assert closure_at_date(index, {"flask": "1.1.1"}, "2020-06-01", "3.9") == {
        "flask": "1.1.1"}
    assert "importlib-metadata" in closure_at_date(
        index, {"flask": "1.1.1"}, "2020-06-01", "3.7")


def test_a_baseline_is_repaired_even_though_the_install_never_resolved():
    """The gap this closes: previously such a subject was rejected outright."""
    index = MetadataIndex(
        {"jinja2": {"2.11.2": "2020-04-13", "3.1.6": "2025-03-05"}},
        {("flask", "1.1.1"): ["Jinja2 (>=2.10.1)"]})
    reqfile = ReqFile.from_text("flask==1.1.1\n")

    def probe(rf):
        return ok() if rf.pins().get("jinja2") == "2.11.2" else bad("resolution_conflict")

    result = establish(reqfile, probe=probe,
                       closure_of=lambda rf: {},      # pip could not resolve it
                       index=index, commit_date="2020-06-01", python_version="3.9")
    assert result.ok and result.stage == "reconstructed"
    assert result.reqfile.pins()["jinja2"] == "2.11.2"


# ------------------------------------------- projects that state ranges, not pins

class RangeIndex(FakeIndex):
    def newest_at(self, name, on_or_before, python_version=None, specifier=None, prefer_wheels=False):
        from packaging.specifiers import SpecifierSet
        releases = self.dates.get(name, {})
        eligible = [v for v, d in releases.items() if d <= on_or_before]
        if specifier:
            eligible = [v for v in eligible if SpecifierSet(specifier).contains(v)]
        if not eligible:
            return None
        return max(eligible, key=lambda v: tuple(int(p) for p in v.split(".")))


RANGES = RangeIndex({"flask": {"1.1.2": "2020-04-03", "2.0.0": "2021-05-11",
                               "3.1.3": "2025-08-19"},
                     "pytest": {"5.4.2": "2020-05-08", "8.4.2": "2025-09-03"}})


def test_a_range_is_pinned_to_what_the_project_allowed_on_the_day():
    """A range says what was allowed, not what was installed: the environment
    was whatever pip resolved that day, and today it resolves elsewhere."""
    from aura.baseline import resolve_ranges_at_date

    reqfile = ReqFile.from_text("flask>=1.0,<2\npytest\n")
    assert resolve_ranges_at_date(reqfile, RANGES, "2022-01-01", "3.9") == {
        "flask": "1.1.2", "pytest": "5.4.2"}


def test_the_projects_own_cap_is_never_overridden():
    """Reconstructing flask<2 as Flask 2 would replace its decision with ours."""
    from aura.baseline import resolve_ranges_at_date

    reqfile = ReqFile.from_text("flask<2\n")
    assert resolve_ranges_at_date(reqfile, RANGES, "2025-01-01", "3.9") == {
        "flask": "1.1.2"}


def test_a_pin_the_project_wrote_is_left_alone():
    from aura.baseline import resolve_ranges_at_date

    reqfile = ReqFile.from_text("flask==1.1.2\npytest>=5\n")
    out = resolve_ranges_at_date(reqfile, RANGES, "2022-01-01", "3.9")
    assert "flask" not in out and out["pytest"] == "5.4.2"


def test_a_ranged_project_gets_a_baseline_the_search_can_work_on():
    """The whole point: the search works in versions, and a range names none."""
    reqfile = ReqFile.from_text("flask>=1.0,<2\n")

    def probe(rf):
        return ok() if rf.pins().get("flask") == "1.1.2" else bad()

    result = establish(reqfile, probe=probe, closure_of=lambda rf: {},
                       index=RANGES, commit_date="2022-01-01",
                       python_version="3.9")
    assert result.ok and result.reqfile.pins()["flask"] == "1.1.2"


def test_a_package_is_dated_back_within_what_the_rest_of_the_set_allows():
    """Dated back on its own, idna lands on 3.2 -- beside a requests that
    allows only idna<3, which no installer can satisfy. It has to be the
    newest release of that date *inside* the range the set declares."""
    from aura.baseline import reconstruct
    index = MetadataIndex(
        {"idna": {"2.10": "2020-06-27", "3.2": "2021-05-30"}},
        {("requests", "2.25.1"): ["idna (<3,>=2.5)"]})
    fixed, moved = reconstruct(ReqFile.from_text("requests==2.25.1\n"),
                               {"requests": "2.25.1", "idna": "3.10"},
                               index=index, on_date="2021-07-01")
    assert fixed.pins()["idna"] == "2.10"
    assert moved["idna"] == ["3.10", "2.10"]


class _Probe:
    def __init__(self, accept):
        self.accept, self.seen = accept, []

    def __call__(self, reqfile):
        self.seen.append(reqfile.to_text())
        ok = self.accept(reqfile)
        return type("R", (), {"accepted": ok, "error_class": "" if ok else "validation_failed"})()


def test_a_frozen_baseline_is_the_only_candidate_every_arm_tries():
    """flask-dj's arms ended on two baselines and nba2k25-plp's on two stages;
    once frozen, every arm starts from exactly the first arm's set."""
    from aura.baseline import establish
    from aura.reqfile import ReqFile
    declared = ReqFile.from_text("flask==1.1.2\n")
    frozen = ReqFile.from_text("flask==1.1.2\njinja2==2.11.3\n")
    probe = _Probe(lambda r: True)
    out = establish(declared, probe=probe, closure_of=lambda r: {}, index=None,
                    commit_date="2020-01-01", frozen=frozen)
    assert out.ok and out.stage == "frozen" and out.reqfile.to_text() == frozen.to_text()
    assert probe.seen == [frozen.to_text()], "the declared file is not tried at all"


def test_a_frozen_baseline_that_stops_passing_fails_rather_than_drifts():
    from aura.baseline import establish
    from aura.reqfile import ReqFile
    out = establish(ReqFile.from_text("flask==1.1.2\n"), probe=_Probe(lambda r: False),
                    closure_of=lambda r: {}, index=None, commit_date="2020-01-01",
                    frozen=ReqFile.from_text("flask==1.1.2\njinja2==2.11.3\n"))
    assert not out.ok and "frozen baseline no longer passes" in out.reason


def test_the_first_arm_to_freeze_wins(tmp_path):
    from aura.baseline import freeze_once, frozen_baseline
    from aura.reqfile import ReqFile
    path = tmp_path / "s" / "s.baseline.txt"
    assert frozen_baseline(path) is None
    assert freeze_once(path, ReqFile.from_text("a==1\n"))
    assert not freeze_once(path, ReqFile.from_text("a==2\n"))
    assert frozen_baseline(path).pins() == {"a": "1"}
