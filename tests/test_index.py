"""Tests for release-history handling, the hallucination check, and version age."""

import pytest

from aura.index import IndexError_, PackageIndex, ReleaseHistory, version_age_report


def hist(name, versions, pre=()):
    return ReleaseHistory(name=name, versions=tuple(versions), prereleases=tuple(pre))


TORCH = hist("torch", ["1.12.0", "1.12.1", "1.13.0", "2.0.0", "2.0.1", "2.1.0"],
             pre=["2.2.0rc1"])


class FakeIndex(PackageIndex):
    """PackageIndex with the network swapped out."""

    def __init__(self, table, fail=(), requires=None):
        super().__init__()
        self.table, self.fail = table, set(fail)
        self.requires = requires or {}

    def _fetch(self, name):
        if name in self.fail:
            raise IndexError_(f"boom: {name}")
        return self.table.get(name)

    def requires_dist(self, name, version):
        return self.requires.get((name, version))


def test_latest_ignores_prereleases():
    assert TORCH.latest == "2.1.0"


def test_between_is_exclusive_low_inclusive_high():
    assert TORCH.between("1.12.0", "2.0.0") == ["1.12.1", "1.13.0", "2.0.0"]
    assert TORCH.between("1.12.0", "2.0.0", inclusive_high=False) == ["1.12.1", "1.13.0"]


def test_between_with_unparseable_bounds_is_empty():
    assert TORCH.between("not-a-version", "2.0.0") == []


def test_releases_behind_counts_newer_stable_releases():
    assert TORCH.releases_behind("1.12.0") == 5
    assert TORCH.releases_behind("2.1.0") == 0


def test_contains_accepts_real_versions_including_prereleases():
    assert TORCH.contains("1.13.0")
    assert TORCH.contains("2.2.0rc1")


def test_contains_rejects_invented_versions():
    """The hallucination guard the paper claimed but never implemented."""
    assert not TORCH.contains("1.12.5")
    assert not TORCH.contains("99.0.0")
    assert not TORCH.contains("banana")


def test_contains_normalizes_equivalent_version_spellings():
    assert hist("x", ["1.0"]).contains("1.0.0")
    assert hist("x", ["1.0.0"]).contains("1.0")


def test_version_exists_via_index():
    idx = FakeIndex({"torch": TORCH})
    assert idx.version_exists("Torch", "2.0.1")      # canonicalized lookup
    assert not idx.version_exists("torch", "2.0.9")
    assert not idx.version_exists("unknown-pkg", "1.0")


def test_history_is_cached_so_a_package_is_fetched_once():
    calls = []

    class Counting(FakeIndex):
        def _fetch(self, name):
            calls.append(name)
            return super()._fetch(name)

    idx = Counting({"torch": TORCH})
    idx.history("torch"); idx.history("torch"); idx.latest("TORCH")
    assert calls == ["torch"]


def test_index_failure_raises_rather_than_reading_as_no_update():
    """v1 swallowed this and treated the package as already up to date."""
    idx = FakeIndex({}, fail={"flaky"})
    with pytest.raises(IndexError_):
        idx.history("flaky")


def test_unknown_project_returns_none_not_an_error():
    assert FakeIndex({}).history("never-published") is None


def test_version_age_report_quantifies_modernization():
    idx = FakeIndex({
        "torch": TORCH,
        "numpy": hist("numpy", ["1.22.0", "1.24.0", "1.26.4"]),
    })
    before = version_age_report(idx, {"torch": "1.12.0", "numpy": "1.22.0"})
    assert before["mean_releases_behind"] == pytest.approx(3.5)   # (5 + 2) / 2
    assert before["at_latest"] == 0

    after = version_age_report(idx, {"torch": "2.1.0", "numpy": "1.26.4"})
    assert after["mean_releases_behind"] == 0.0
    assert after["at_latest"] == 2 and after["at_latest_pct"] == 100.0


def test_version_age_is_measured_against_a_fixed_date_not_the_moment():
    """Two arms measured days apart must be comparable.

    PyPI moves while a campaign runs. Arms of one subject run together, but the
    dependabot arm is collected separately, and releases published in between
    marked it down for having done nothing. mean-review-collector's fell from
    26.32% to 21.05% between two collections a day apart, with no package
    changed.
    """
    # A project this index invents, with dates PyPI could not supply: if this
    # test can be satisfied from the network it is not testing anything.
    dated = hist("only-in-this-test", ["1.0.0", "2.0.0"])
    idx = FakeIndex({"only-in-this-test": dated})
    idx.upload_dates = lambda name: {"1.0.0": "2001-01-01", "2.0.0": "2002-01-01"}

    # Pinned to a date before the newest release, 1.0.0 *is* the latest.
    early = version_age_report(idx, {"only-in-this-test": "1.0.0"},
                               as_of="2001-06-01")
    assert early["at_latest"] == 1 and early["per_package"]["only-in-this-test"] == 0

    # The same pin, measured later, is one release behind -- through no act of
    # whichever arm produced it.
    later = version_age_report(idx, {"only-in-this-test": "1.0.0"},
                               as_of="2002-06-01")
    assert later["at_latest"] == 0 and later["per_package"]["only-in-this-test"] == 1


def test_version_age_falls_back_to_the_live_answer_without_dates():
    """An index that cannot date its releases still has to answer."""
    idx = FakeIndex({"numpy": hist("numpy", ["1.22.0", "1.24.0", "1.26.4"])})
    idx.upload_dates = lambda name: {}          # an index that dates nothing
    report = version_age_report(idx, {"numpy": "1.24.0"}, as_of="2023-06-01")
    assert report["per_package"]["numpy"] == 1, (
        "with no upload dates the count is every release above the pin, "
        "which is what this reported before dates were considered")


def test_version_age_report_lists_unknown_packages_separately():
    idx = FakeIndex({"torch": TORCH}, fail={"private-pkg"})
    report = version_age_report(idx, {"torch": "2.0.0", "private-pkg": "1.0"})
    assert report["unknown"] == ["private-pkg"]
    assert report["n_packages"] == 1


def _binary(name, releases):
    """A history whose releases carry the wheel files given."""
    return ReleaseHistory(name=name, versions=tuple(releases),
                          metadata={v: {"requires_python": None, "yanked": False,
                                        "files": files}
                                    for v, files in releases.items()})


def test_a_release_with_no_wheel_for_the_interpreter_is_stepped_past_only_when_asked():
    """grpcio 1.19 predates Python 3.8 and has no cp38 wheel, so pip compiles it
    and the compile fails. Asked, the ladder steps to the nearest release that
    has one; not asked, the date-exact release is kept."""
    releases = {
        "1.19.0": ["grpcio-1.19.0-cp37-cp37m-manylinux1_x86_64.whl",
                   "grpcio-1.19.0.tar.gz"],
        "1.24.3": ["grpcio-1.24.3-cp38-cp38-manylinux2010_x86_64.whl"],
        "1.30.0": ["grpcio-1.30.0-cp38-cp38-manylinux2014_x86_64.whl"],
    }
    idx = FakeIndex({"grpcio": _binary("grpcio", releases)})
    idx.upload_dates = lambda name: {"1.19.0": "2019-02-26", "1.24.3": "2019-10-23",
                                     "1.30.0": "2020-06-23"}
    assert idx.newest_at("grpcio", "2019-04-02", "3.8") == "1.19.0"
    assert idx.newest_at("grpcio", "2019-04-02", "3.8", prefer_wheels=True) == "1.24.3", (
        "the nearest later release with a wheel, not the newest")


def test_pure_python_and_abi3_releases_are_left_where_they_are():
    six = _binary("six", {"1.12.0": ["six-1.12.0-py2.py3-none-any.whl"]})
    crypto = _binary("cryptography", {
        "2.6.1": ["cryptography-2.6.1-cp34-abi3-manylinux1_x86_64.whl"]})
    assert not six.ships_binary_wheels()
    assert six.has_wheel_for("1.12.0", "3.8")
    assert crypto.ships_binary_wheels()
    assert crypto.has_wheel_for("2.6.1", "3.8"), "an abi3 wheel serves every later CPython"


def test_macos_and_musl_wheels_do_not_count_for_a_linux_glibc_machine():
    h = _binary("x", {"1.0": ["x-1.0-cp38-cp38-macosx_10_9_x86_64.whl",
                              "x-1.0-cp38-cp38-musllinux_1_1_x86_64.whl"]})
    assert not h.has_wheel_for("1.0", "3.8")
