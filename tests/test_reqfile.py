"""Round-trip fidelity and re-pinning tests for aura.reqfile.

The headline test is `test_round_trip_is_byte_identical`: reading any of these
files and writing it straight back out must produce identical bytes. The legacy
regex + split('==') parser fails most of these cases silently.
"""

import pytest

from aura.reqfile import ReqFile, LineKind, freeze_to_reqfile, parse_freeze

# Every awkward line shape that appears in real requirements files.
HARD_CASES = """\
# Core numerics
numpy==1.22.0
requests[security,socks]==2.25.0
torch>=1.8,<2.0
ruamel.yaml==0.17.21
zope.interface==5.4.0
backports.zoneinfo==0.2.1 ; python_version < "3.9"
importlib-metadata==4.11.0  # pinned for py3.7 compat
Pillow==9.0.0
PyYAML==6.0
scikit-learn==1.0.2

-e .
-r base.txt
--index-url https://pypi.org/simple
git+https://github.com/psf/requests.git@v2.28.0#egg=requests
mypkg @ https://example.com/mypkg-1.0.tar.gz
uvicorn[standard]>=0.20 ; sys_platform != "win32"  # server
"""


def test_round_trip_is_byte_identical():
    text = HARD_CASES
    assert ReqFile.from_text(text).to_text() == text


@pytest.mark.parametrize("text", [
    "",
    "\n",
    "numpy==1.0",              # no trailing newline
    "numpy==1.0\n",
    "\n\n# only comments\n\n",
    "  numpy==1.0  \n",        # surrounding whitespace preserved
])
def test_round_trip_edge_cases(text):
    assert ReqFile.from_text(text).to_text() == text


def test_line_classification():
    rf = ReqFile.from_text(HARD_CASES)
    kinds = {l.raw.strip(): l.kind for l in rf.lines if l.raw.strip()}
    assert kinds["numpy==1.22.0"] == LineKind.REQUIREMENT
    assert kinds["-e ."] == LineKind.OPTION
    assert kinds["-r base.txt"] == LineKind.OPTION
    assert kinds["--index-url https://pypi.org/simple"] == LineKind.OPTION
    assert kinds["git+https://github.com/psf/requests.git@v2.28.0#egg=requests"] == LineKind.URL
    assert kinds["mypkg @ https://example.com/mypkg-1.0.tar.gz"] == LineKind.URL
    assert kinds['# Core numerics'] == LineKind.COMMENT


def test_canonical_names_and_pins():
    rf = ReqFile.from_text(HARD_CASES)
    pins = rf.pins()
    # PEP 503 canonicalization: case folded, _ and . collapsed to -
    assert pins["pillow"] == "9.0.0"
    assert pins["pyyaml"] == "6.0"
    assert pins["scikit-learn"] == "1.0.2"
    assert pins["ruamel-yaml"] == "0.17.21"
    assert pins["zope-interface"] == "5.4.0"
    assert pins["backports-zoneinfo"] == "0.2.1"
    assert pins["importlib-metadata"] == "4.11.0"
    # Ranges are not pins
    assert "torch" not in pins
    assert "uvicorn" not in pins


def test_repin_preserves_extras():
    rf = ReqFile.from_text("requests[security,socks]==2.25.0\n")
    assert rf.repin("requests", "2.31.0")
    assert rf.to_text() == "requests[security,socks]==2.31.0\n"


def test_repin_preserves_marker():
    rf = ReqFile.from_text('backports.zoneinfo==0.2.1 ; python_version < "3.9"\n')
    assert rf.repin("backports-zoneinfo", "0.2.2")
    assert rf.to_text() == 'backports.zoneinfo==0.2.2 ; python_version < "3.9"\n'


def test_repin_preserves_inline_comment():
    rf = ReqFile.from_text("importlib-metadata==4.11.0  # pinned for py3.7 compat\n")
    assert rf.repin("importlib_metadata", "6.0.0")
    assert rf.to_text() == "importlib-metadata==6.0.0  # pinned for py3.7 compat\n"


def test_repin_converts_a_range_to_a_pin():
    rf = ReqFile.from_text("torch>=1.8,<2.0\n")
    assert rf.repin("torch", "1.13.1")
    assert rf.to_text() == "torch==1.13.1\n"


def test_repin_leaves_every_other_line_untouched():
    rf = ReqFile.from_text(HARD_CASES)
    rf.repin("numpy", "1.26.4")
    out = rf.to_text().splitlines()
    original = HARD_CASES.splitlines()
    diff = [(a, b) for a, b in zip(original, out) if a != b]
    assert diff == [("numpy==1.22.0", "numpy==1.26.4")]


def test_repin_unknown_package_returns_false():
    rf = ReqFile.from_text("numpy==1.0\n")
    assert rf.repin("does-not-exist", "1.0") is False
    assert rf.to_text() == "numpy==1.0\n"


def test_url_lines_are_never_repinned():
    rf = ReqFile.from_text("mypkg @ https://example.com/mypkg-1.0.tar.gz\n")
    assert rf.repin("mypkg", "2.0") is False
    assert rf.to_text() == "mypkg @ https://example.com/mypkg-1.0.tar.gz\n"


def test_with_pins_does_not_mutate_the_original():
    rf = ReqFile.from_text("numpy==1.22.0\nscipy==1.8.0\n")
    clone = rf.with_pins({"numpy": "1.26.4"})
    assert rf.to_text() == "numpy==1.22.0\nscipy==1.8.0\n"
    assert clone.to_text() == "numpy==1.26.4\nscipy==1.8.0\n"


def test_drop_all_constraints_keeps_extras_and_markers():
    rf = ReqFile.from_text(
        'requests[security]==2.25.0\n'
        'backports.zoneinfo==0.2.1 ; python_version < "3.9"\n'
        '-e .\n'
    )
    rf.drop_all_constraints()
    assert rf.to_text() == (
        'requests[security]\n'
        'backports.zoneinfo ; python_version < "3.9"\n'
        '-e .\n'
    )


def test_is_fully_pinned():
    assert ReqFile.from_text("numpy==1.0\nscipy==1.1\n").is_fully_pinned()
    assert not ReqFile.from_text("numpy==1.0\ntorch>=1.8\n").is_fully_pinned()
    assert not ReqFile.from_text("").is_fully_pinned()


def test_parse_freeze_and_lockfile_generation():
    freeze = "numpy==1.26.4\n-e git+https://x/y.git#egg=proj\nPillow==10.0.0\n"
    assert parse_freeze(freeze) == {"numpy": "1.26.4", "pillow": "10.0.0"}
    rf = freeze_to_reqfile(freeze)
    assert rf.to_text() == (
        "-e git+https://x/y.git#egg=proj\n"
        "numpy==1.26.4\n"
        "Pillow==10.0.0\n"
    )


# --- Regression tests against the *old* parser's specific failure modes -------

def _legacy_name(line):
    """The parser AURA used to ship, reproduced for comparison."""
    import re
    m = re.match(r"([a-zA-Z0-9\-_]+)", line)
    return m.group(1) if m else None


def _legacy_version(line):
    return line.split("==")[1].split(";")[0].strip()


@pytest.mark.parametrize("line,correct_name", [
    ("requests[security]==2.25.0", "requests"),
    ("ruamel.yaml==0.17.21", "ruamel-yaml"),
    ("zope.interface==5.4.0", "zope-interface"),
    ("Pillow==9.0.0", "pillow"),
])
def test_new_parser_fixes_names_the_old_one_got_wrong(line, correct_name):
    rf = ReqFile.from_text(line + "\n")
    got = next(rf.requirements()).name
    assert got == correct_name


def test_old_parser_mangled_inline_comments_new_one_does_not():
    line = "importlib-metadata==4.11.0  # pinned for py3.7 compat"
    assert _legacy_version(line) == "4.11.0  # pinned for py3.7 compat"  # broken
    rf = ReqFile.from_text(line + "\n")
    assert next(rf.requirements()).pinned_version == "4.11.0"  # fixed


# --- The parsed form and the rendered form must never disagree ---------------

def test_pins_reflects_a_repin_immediately():
    """Regression: repin used to rewrite only the text, so pins() stayed stale
    and the agent's update loop re-proposed the same change every pass."""
    rf = ReqFile.from_text("torch==1.12.0\n")
    rf.repin("torch", "2.0.0")
    assert rf.pins() == {"torch": "2.0.0"}
    assert rf.to_text() == "torch==2.0.0\n"
    assert next(rf.requirements()).pinned_version == "2.0.0"


def test_repin_is_idempotent_and_chainable():
    rf = ReqFile.from_text("torch==1.12.0  # note\n")
    rf.repin("torch", "1.13.0")
    rf.repin("torch", "2.0.0")
    assert rf.pins() == {"torch": "2.0.0"}
    assert rf.to_text() == "torch==2.0.0  # note\n"


def test_with_pins_result_reports_the_new_versions():
    rf = ReqFile.from_text("torch==1.12.0\nnumpy==1.22.0\n")
    assert rf.with_pins({"torch": "2.0.0"}).pins() == {"torch": "2.0.0", "numpy": "1.22.0"}
    assert rf.pins()["torch"] == "1.12.0"       # original untouched


def test_reset_restores_the_parsed_version_too():
    rf = ReqFile.from_text("torch==1.12.0\n")
    line = next(rf.requirements())
    line.repin("2.0.0")
    line.reset()
    assert line.pinned_version == "1.12.0"
    assert rf.to_text() == "torch==1.12.0\n"


def test_repin_after_dropping_constraints_still_works():
    rf = ReqFile.from_text('numpy==1.22.0 ; python_version >= "3.8"\n')
    rf.drop_all_constraints()
    assert rf.pins() == {}
    assert rf.repin("numpy", "1.26.4")
    assert rf.pins() == {"numpy": "1.26.4"}


def test_a_debian_pip_freeze_artefact_is_not_read_as_a_dependency():
    """pkg-resources==0.0.0 names no release on PyPI; a file carrying it can
    never install, and removing it changes nothing that does."""
    from aura.reqfile import ReqFile
    rf = ReqFile.from_text("Flask==1.0.2\npkg-resources==0.0.0\nrequests==2.21.0\n")
    assert rf.pins() == {"flask": "1.0.2", "requests": "2.21.0"}
    assert "pkg-resources" not in rf.to_text()
    # A real pin on the name, should one exist, is left alone.
    assert "pkg-resources" in ReqFile.from_text("pkg-resources==1.0\n").pins()
