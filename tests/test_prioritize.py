"""Tests for Architectural Spread and the ordering strategies."""

import textwrap
import pytest

from aura.prioritize import (
    Candidate, analyze_imports, architectural_spread, order_candidates,
)


@pytest.fixture
def repo(tmp_path):
    def write(rel, src):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))
    # torch imported 3x in ONE file -> spread 1, statements 3
    write("model.py", """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        import os
        import sys
    """)
    # numpy in two files -> spread 2
    write("train.py", """
        import numpy as np
        import cv2
        from sklearn.metrics import accuracy_score
        from . import model
    """)
    write("utils.py", """
        import numpy
        from PIL import Image
        import yaml
    """)
    # excluded directory must not contribute
    write("temp_venv/lib/thing.py", "import torch\nimport numpy\nimport scipy\n")
    return tmp_path


def test_spread_counts_files_not_statements(repo):
    files_by_module, statements = analyze_imports(repo)
    assert len(files_by_module["torch"]) == 1      # one file
    assert statements["torch"] == 3                # three statements
    assert len(files_by_module["numpy"]) == 2


def test_stdlib_is_ignored(repo):
    files_by_module, _ = analyze_imports(repo)
    assert "os" not in files_by_module
    assert "sys" not in files_by_module


def test_relative_imports_are_ignored(repo):
    files_by_module, _ = analyze_imports(repo)
    assert "model" not in files_by_module


def test_excluded_directories_are_skipped(repo):
    files_by_module, _ = analyze_imports(repo)
    assert "scipy" not in files_by_module        # only in temp_venv/
    assert len(files_by_module["torch"]) == 1    # not 2


def test_spread_is_keyed_by_distribution_name(repo):
    """The v1 bug: cv2/sklearn/PIL/yaml scored 0 because they were keyed by import name."""
    spread, _ = architectural_spread(repo)
    for dist in ("opencv-python", "scikit-learn", "pillow", "pyyaml"):
        assert spread.get(dist, 0) > 0, f"{dist} should have non-zero spread"
    assert spread["torch"] == 1
    assert spread["numpy"] == 2


def _cands():
    # torch: huge usage, patch bump. requests: tiny usage, major bump.
    return [
        Candidate("torch", "2.0.0", "2.0.1", spread=40, import_statements=90, is_primary=True),
        Candidate("requests", "1.0.0", "3.0.0", spread=1, import_statements=1, is_primary=True),
        Candidate("attrs", "22.1.0", "23.1.0", spread=0, import_statements=0, fanout=5),
    ]


def test_spread_ordering_puts_the_most_used_package_first():
    ordered = order_candidates(_cands(), "spread")
    assert [c.name for c in ordered] == ["torch", "requests", "attrs"]


def test_legacy_ordering_puts_the_biggest_version_jump_first():
    """Documents what v1 actually did: requests (major bump) outranks torch."""
    ordered = order_candidates(_cands(), "version_jump")
    # Both major bumps sort above torch's patch bump, despite torch being the
    # most-used package in the repo by a wide margin.
    assert ordered[-1].name == "torch"
    assert ordered[-1].version_jump == 1
    assert all(c.version_jump == 10 for c in ordered[:2])


def test_spread_and_legacy_orderings_genuinely_disagree():
    assert (
        [c.name for c in order_candidates(_cands(), "spread")]
        != [c.name for c in order_candidates(_cands(), "version_jump")]
    )


def test_random_ordering_is_reproducible_for_a_seed():
    a = [c.name for c in order_candidates(_cands(), "random", seed=7)]
    b = [c.name for c in order_candidates(_cands(), "random", seed=7)]
    c = [c.name for c in order_candidates(_cands(), "random", seed=8)]
    assert a == b
    assert sorted(a) == sorted(c)


def test_unknown_ordering_is_rejected():
    with pytest.raises(ValueError, match="unknown ordering"):
        order_candidates(_cands(), "nonsense")


@pytest.mark.parametrize("strategy", ["spread", "version_jump", "import_statements",
                                      "fanout", "alphabetical", "reverse_spread", "random"])
def test_every_ordering_is_a_permutation(strategy):
    ordered = order_candidates(_cands(), strategy)
    assert sorted(c.name for c in ordered) == ["attrs", "requests", "torch"]


def test_version_jump_classification():
    assert Candidate("x", "1.0.0", "2.0.0").version_jump == 10
    assert Candidate("x", "1.0.0", "1.1.0").version_jump == 5
    assert Candidate("x", "1.0.0", "1.0.1").version_jump == 1
    assert Candidate("x", "junk", "1.0.1").version_jump == 5   # unparseable -> neutral
