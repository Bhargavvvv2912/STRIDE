"""The committed registry must load: a stray field fails every cluster run."""
from pathlib import Path

from evaluation.subjects import Registry


def test_subjects_json_loads():
    reg = Registry.load(Path(__file__).resolve().parent.parent / "subjects.json")
    assert reg.included
