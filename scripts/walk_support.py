"""Small, independently testable safeguards for the bounded walk checker."""
import json
import subprocess
from pathlib import Path


def clean_result(result):
    return result is not None and result[0] > 0 and result[1] == 0 and result[2] == 0


def runnable_baseline(result):
    """AURA's rule: the suite ran (pytest exit 0 or 1) and something passed;
    failures already present at baseline are tolerated."""
    return result is not None and result[0] > 0 and result[2] in (0, 1)


def acceptable(result, base):
    """No new failures and no lost passes relative to the baseline."""
    return (result is not None and result[2] in (0, 1)
            and result[0] >= base[0] and result[1] <= base[1])


def conflict_names(text, known):
    """Pinned packages pip's resolver names in a conflict report."""
    import re
    canon = lambda n: re.sub(r"[-_.]+", "-", n).lower()
    names = set()
    for line in text.splitlines():
        if re.search(r"depends on|requested|conflict is caused|requires|incompatible", line, re.I):
            for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", line):
                if canon(tok) in known:
                    names.add(canon(tok))
    return names


def reaches_target(state, target, version):
    return all(name in state and version(state[name]) >= version(value)
               for name, value in target.items())


class ProbeLogger:
    def __init__(self, root, cwd):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cwd = cwd
        self.state = {}
        self.sequence = 0
        self.interrupted = False

    def run(self, args, stage="command", **kwargs):
        self.sequence += 1
        stem = self.root / ("{:05d}-{}".format(self.sequence, stage))
        kwargs.pop("capture_output", None)
        kwargs.pop("text", None)
        check = kwargs.pop("check", False)
        kwargs.setdefault("cwd", self.cwd)
        kwargs.setdefault("timeout", 900)
        metadata = {"argv": [str(x) for x in args], "cwd": str(kwargs["cwd"]),
                    "state": dict(self.state), "timeout": kwargs["timeout"]}
        if "-r" in args:
            req = Path(args[args.index("-r") + 1])
            if req.is_file():
                stem.with_suffix(".requirements.txt").write_text(req.read_text())
        stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        # Stream logs to disk so a killed process leaves diagnostic evidence.
        with stem.with_suffix(".stdout.log").open("w") as out, stem.with_suffix(".stderr.log").open("w") as err:
            try:
                result = subprocess.run(args, stdout=out, stderr=err, text=True, **kwargs)
            except subprocess.TimeoutExpired:
                result = subprocess.CompletedProcess(args, 124)
                err.write("\nTIMEOUT\n")
            except OSError as exc:
                result = subprocess.CompletedProcess(args, 127)
                err.write(str(exc) + "\n")
        result.stdout = stem.with_suffix(".stdout.log").read_text(errors="replace")
        result.stderr = stem.with_suffix(".stderr.log").read_text(errors="replace")
        metadata["returncode"] = result.returncode
        if result.returncode in (124, 127) or result.returncode < 0:
            self.interrupted = True
        stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        if check:
            result.check_returncode()
        return result
