"""Code agents STRIDE can plug in. Each gets one task and a locked workspace.

Only mini-SWE-agent is wired up for the pilot. Others (OpenHands, Aider,
SWE-agent) implement the same ``CodeAgent.run`` and register in ``AGENTS``.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentRun:
    exit_status: str = ""
    steps: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trajectory: str = ""
    error: str = ""
    extra: dict = field(default_factory=dict)


class CodeAgent:
    name = "base"

    def run(self, task: str, workdir: Path, env: dict[str, str], out_dir: Path) -> AgentRun:
        raise NotImplementedError


def ollama_base() -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip().rstrip("/")
    return host if host.startswith("http") else f"http://{host}"


SYSTEM_EXTRA = """
Rules for this task:
- Do NOT edit, delete or add test files, test configuration, or requirements files.
  They are read-only; edits to them will be rejected.
- Do NOT install, upgrade or downgrade packages. The environment is fixed.
- Do NOT skip, xfail or silence tests, and do not weaken assertions.
- Change the project's own source code, as little as needed, keeping its behaviour.

Editing tips:
- For any change spanning more than one line, rewrite the whole file with a heredoc:
  cat > path/to/file.py <<'PYEOF'
  ...full new content...
  PYEOF
  Do not use sed for multi-line Python; it breaks indentation.
- After each edit, run `python -m py_compile path/to/file.py`, then the tests.
- If the same command fails twice, change approach instead of repeating it.
- If the error is raised inside an installed package (a path under site-packages),
  read that package's code to see what it expects, then adapt the project's code
  around it; you cannot change the package itself.
"""


def _git(workdir: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=workdir, capture_output=True,
                          text=True).stdout


def _state(workdir: Path) -> str:
    """A fingerprint of the working tree: changes only when some file does."""
    diff = _git(workdir, "diff", "HEAD") + _git(workdir, "ls-files", "--others",
                                                    "--exclude-standard")
    return hashlib.sha1(diff.encode()).hexdigest()


def _changed_py(workdir: Path) -> set[str]:
    out = _git(workdir, "status", "--porcelain", "--untracked-files=all")
    return {line[3:].strip() for line in out.splitlines() if line[3:].strip().endswith(".py")}


def guarded_agent_class():
    """DefaultAgent with two guards against what local models do most:
    - a command repeated while no file changed is not run again (its output
      would be the same); the agent is told to try something else;
    - an edit that leaves a Python file unparsable is undone at once, and the
      agent sees the syntax error."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.exceptions import LimitsExceeded

    class GuardedAgent(DefaultAgent):
        max_blocked = 8

        def setup_guards(self, workdir: Path, python: str):
            self.workdir, self.python = workdir, python
            self.seen: dict[tuple[str, str], int] = {}
            self.blocked = 0
            self.reverted = 0

        def _snapshot(self) -> dict[str, str | None]:
            snap = {}
            for rel in _changed_py(self.workdir):
                path = self.workdir / rel
                snap[rel] = path.read_text(errors="replace") if path.exists() else None
            return snap

        def _syntax_error(self, rel: str) -> str:
            out = subprocess.run([self.python, "-m", "py_compile", rel], cwd=self.workdir,
                                 capture_output=True, text=True)
            return "" if out.returncode == 0 else (out.stderr or out.stdout)[-800:]

        def execute_actions(self, message: dict) -> list[dict]:
            actions = message.get("extra", {}).get("actions", [])
            state = _state(self.workdir)
            repeats = [a for a in actions
                       if (" ".join(str(a.get("command", a)).split()), state) in self.seen]
            if actions and len(repeats) == len(actions):
                self.blocked += 1
                if self.blocked >= self.max_blocked:
                    raise LimitsExceeded({"role": "exit", "content": "RepeatedCommands",
                                          "extra": {"exit_status": "RepeatedCommands",
                                                    "submission": ""}})
                return self.add_messages(self.model.format_message(role="user", content=(
                    "You already ran exactly this command, and no file has changed since, "
                    "so its output would be the same. Do not repeat it. Use what you "
                    "already learned: make an edit, look somewhere new (for example the "
                    "installed package that raises the error), or run the tests.")))
            for a in actions:
                self.seen[(" ".join(str(a.get("command", a)).split()), state)] = 1
            before = self._snapshot()
            observations = super().execute_actions(message)
            notes = []
            for rel in sorted(_changed_py(self.workdir)):
                path = self.workdir / rel
                if not path.exists() or path.read_text(errors="replace") == before.get(rel):
                    continue
                error = self._syntax_error(rel)
                if not error:
                    continue
                previous = before.get(rel)
                if previous is None and rel not in before:   # was clean: back to HEAD
                    previous = _git(self.workdir, "show", f"HEAD:{rel}") or None
                if previous is None:
                    path.unlink()
                else:
                    path.write_text(previous)
                self.reverted += 1
                notes.append(f"Your edit to {rel} was undone because the file no longer "
                             f"compiles:\n{error}\nWrite the whole file again with a "
                             "heredoc (cat > file <<'PYEOF' ... PYEOF), keeping indentation.")
            if notes:
                observations += self.add_messages(
                    self.model.format_message(role="user", content="\n\n".join(notes)))
            return observations

    return GuardedAgent


class MiniSweAgent(CodeAgent):
    """mini-SWE-agent (text-based actions, works with local models via litellm)."""

    name = "mini-swe-agent"

    def __init__(self, model: str = "ollama_chat/qwen2.5-coder:14b", step_limit: int = 60,
                 wall_time_s: int = 3600, command_timeout_s: int = 900,
                 num_ctx: int = 65536, temperature: float = 0.0):
        self.model = model
        self.step_limit = step_limit
        self.wall_time_s = wall_time_s
        self.command_timeout_s = command_timeout_s
        self.num_ctx = num_ctx
        self.temperature = temperature

    def _config(self) -> dict:
        import yaml
        import minisweagent
        path = Path(minisweagent.__file__).parent / "config" / "mini_textbased.yaml"
        return yaml.safe_load(path.read_text())

    def run(self, task: str, workdir: Path, env: dict[str, str], out_dir: Path) -> AgentRun:
        os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
        os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

        cfg = self._config()
        agent_cfg = {k: v for k, v in cfg.get("agent", {}).items()
                     if k in ("system_template", "instance_template")}
        agent_cfg["system_template"] = agent_cfg["system_template"] + SYSTEM_EXTRA
        model_cfg = dict(cfg.get("model", {}))
        kwargs = dict(model_cfg.pop("model_kwargs", {}) or {})
        kwargs.update({"temperature": self.temperature})
        if self.model.startswith("ollama"):
            kwargs.update({"api_base": ollama_base(), "num_ctx": self.num_ctx})
        model = LitellmTextbasedModel(model_name=self.model, model_kwargs=kwargs,
                                      cost_tracking="ignore_errors", **model_cfg)
        environment = LocalEnvironment(cwd=str(workdir), env=env,
                                       timeout=self.command_timeout_s)
        out_dir.mkdir(parents=True, exist_ok=True)
        trajectory = out_dir / "trajectory.json"
        agent = guarded_agent_class()(model, environment, step_limit=self.step_limit,
                             cost_limit=0.0, wall_time_limit_seconds=self.wall_time_s,
                             output_path=trajectory, **agent_cfg)
        python = os.path.join(env.get("VIRTUAL_ENV", ""), "bin", "python")
        agent.setup_guards(workdir, python if os.path.exists(python) else "python3")
        started = time.time()
        run = AgentRun(trajectory=str(trajectory))
        try:
            info = agent.run(task) or {}
            run.exit_status = info.get("exit_status", "")
        except Exception as exc:  # the attempt failed; the controller rolls back
            run.exit_status = "crashed"
            run.error = f"{type(exc).__name__}: {exc}"
        run.seconds = round(time.time() - started, 1)
        run.steps = agent.n_calls
        run.extra = {"blocked_repeats": agent.blocked, "reverted_edits": agent.reverted}
        for message in agent.messages:
            usage = ((message.get("extra") or {}).get("response") or {}).get("usage") or {}
            run.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            run.completion_tokens += int(usage.get("completion_tokens") or 0)
        return run


AGENTS = {"mini-swe-agent": MiniSweAgent}


def build(name: str, **kwargs) -> CodeAgent:
    if name not in AGENTS:
        raise KeyError(f"unknown code agent {name!r}; known: {sorted(AGENTS)}")
    return AGENTS[name](**kwargs)
