"""The revamp: test-setup rule, floors, the series search, missing test packages."""
from pathlib import Path
from types import SimpleNamespace

from aura.reqfile import ReqFile

from stride import guard
from stride.repair import _install_missing, unpinned
from stride.search import plan_tries, search_series, series_key


# ------------------------------------------------------------- test rule

TEST = '''
import pytest
from app import thing

def test_a():
    assert thing() == 1
    with pytest.raises(ValueError):
        thing(bad=True)

class TestB:
    def test_b(self):
        assert thing() > 0
'''


def _shape(tmp_path, text, name="tests/test_x.py"):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return guard.test_shape(tmp_path, [name])


def test_setup_changes_are_allowed(tmp_path):
    before = _shape(tmp_path, TEST)
    after = _shape(tmp_path, TEST.replace("from app import thing",
                                          "from app.core import thing  # moved"))
    assert guard.weakened_tests(before, after) == []


def test_changing_an_assert_is_rejected(tmp_path):
    before = _shape(tmp_path, TEST)
    after = _shape(tmp_path, TEST.replace("assert thing() == 1", "assert thing() >= 0"))
    assert any("check changed" in p for p in guard.weakened_tests(before, after))


def test_removing_raises_or_a_test_or_adding_skip_is_rejected(tmp_path):
    before = _shape(tmp_path, TEST)
    no_raises = TEST.replace("    with pytest.raises(ValueError):\n        thing(bad=True)\n", "")
    assert guard.weakened_tests(before, _shape(tmp_path, no_raises))
    renamed = TEST.replace("def test_b", "def check_b")
    assert any("removed" in p for p in guard.weakened_tests(before, _shape(tmp_path, renamed)))
    skipped = TEST.replace("def test_a", "@pytest.mark.skip\ndef test_a")
    assert any("skip" in p for p in guard.weakened_tests(before, _shape(tmp_path, skipped)))


def test_test_selection_in_config_is_rejected(tmp_path):
    before = _shape(tmp_path, "[pytest]\naddopts = -q\n", "pytest.ini")
    after = _shape(tmp_path, "[pytest]\naddopts = -q --deselect tests/test_x.py::test_a\n",
                   "pytest.ini")
    assert any("selection" in p for p in guard.weakened_tests(before, after))


def test_test_aware_source_lines_are_flagged():
    diff = ("+++ b/app/search.py\n+    if isinstance(es, MockElasticsearch):\n"
            "+    x = 1\n+++ b/tests/conftest.py\n+    m = Mock()\n")
    assert guard.test_aware_lines(diff) == ["if isinstance(es, MockElasticsearch):"]


# ------------------------------------------------------------- floors

def test_unpinned_packages_keep_their_current_version_as_floor():
    req = ReqFile.from_text("graphene==2.1.6\ngraphql-core==2.2\n")
    out = unpinned(req.with_pins({"graphql-core": "3.0.0"}), ["graphene"],
                   floors={"graphene": "2.1.6"})
    assert "graphene>=2.1.6" in out.to_text() and "graphql-core==3.0.0" in out.to_text()


# ------------------------------------------------------------- series search

def test_series_keys_and_order():
    assert series_key("9.5.1") == (9,) and series_key("0.4.2") == (0, 4)
    assert plan_tries(["8.1", "9.0", "9.2", "8.0", "9.5"]) == [["9.0", "9.2", "9.5"],
                                                               ["8.0", "8.1"]]


def _attempts(works):
    tried = []

    def attempt(v):
        tried.append(v)
        return v if v in works else None
    return attempt, tried


def test_newest_works_in_one_try():
    attempt, tried = _attempts({"9.5"})
    assert search_series(["9.0", "9.2", "9.4", "9.5"], attempt) == "9.5" and tried == ["9.5"]


def test_binary_search_finds_newest_fixable_inside_series():
    versions = [f"9.{i}" for i in range(9)]           # 9.0 .. 9.8
    attempt, tried = _attempts({f"9.{i}" for i in range(6)})   # 9.0-9.5 fixable
    assert search_series(versions, attempt) == "9.5"
    assert tried[:2] == ["9.8", "9.0"] and len(tried) <= 6


def test_series_out_when_first_release_fails():
    attempt, tried = _attempts(set())
    assert search_series(["9.0", "9.3"], attempt) is None and tried == ["9.3", "9.0"]


# ------------------------------------------------------------- missing test packages

def test_missing_test_package_is_installed(tmp_path):
    installed = []
    env = SimpleNamespace(pip_install=lambda args: (installed.extend(args) or ("", 0)))
    index = SimpleNamespace(newest_at=lambda name, date, py: "0.28.1")
    out = ("RuntimeError: The starlette.testclient module requires the httpx package "
           "to be installed.\n")
    assert _install_missing(env, out, index, "3.9", tmp_path) == {"httpx": "0.28.1"}
    assert installed == ["httpx==0.28.1"]


def test_project_modules_are_not_installed(tmp_path):
    (tmp_path / "app").mkdir()
    env = SimpleNamespace(pip_install=lambda args: ("", 0))
    index = SimpleNamespace(newest_at=lambda *a: "1.0")
    assert _install_missing(env, "No module named 'app'", index, "3.9", tmp_path) == {}
