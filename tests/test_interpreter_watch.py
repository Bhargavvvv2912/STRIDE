import subprocess
import sys
import sysconfig
from pathlib import Path

from stride import guard


def test_planted_sitecustomize_is_caught_and_removed(tmp_path):
    venv = tmp_path / "v"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    python = str(venv / "bin" / "python")
    purelib = subprocess.run([python, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                             capture_output=True, text=True).stdout.strip()
    with guard.InterpreterWatch(python) as watch:
        (Path(purelib) / "sitecustomize.py").write_text("print('patched')\n")
    assert any("sitecustomize.py" in c for c in watch.changed)
    assert not (Path(purelib) / "sitecustomize.py").exists()


def test_quiet_call_reports_nothing(tmp_path):
    with guard.InterpreterWatch(sys.executable) as watch:
        pass
    assert watch.changed == []


def test_printing_hook_cannot_widen_the_watch(tmp_path):
    """Regression: a planted sitecustomize that prints must not shift the
    folders the watch compares (it once turned the stdlib into the prefix and
    deleted every file outside it)."""
    venv = tmp_path / "v"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    python = str(venv / "bin" / "python")
    keep = venv / "keep.txt"
    keep.write_text("x")
    purelib = guard._interpreter_paths(python)[1]
    with guard.InterpreterWatch(python) as watch:
        (Path(purelib) / "sitecustomize.py").write_text("print('noise')\nprint('more')\n")
        (Path(purelib) / "other.py").write_text("")
    assert guard._interpreter_paths(python) == watch.paths
    assert keep.exists() and (Path(purelib) / "other.py").exists()
    assert (venv / "bin" / "python").exists()
