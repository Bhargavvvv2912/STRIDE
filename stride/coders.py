"""Code agents STRIDE can plug in. Each gets one task and a locked workspace.

Only mini-SWE-agent is wired up for the pilot. Others (OpenHands, Aider,
SWE-agent) implement the same ``CodeAgent.run`` and register in ``AGENTS``.
"""
from __future__ import annotations

import hashlib
import os
import shutil
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

    @property
    def label(self) -> str:
        """Folder name for this agent's runs: agent and model."""
        model = getattr(self, "model", "")
        short = model.split("/")[-1].replace(":", "-") if model else ""
        return f"{self.name}__{short}" if short else self.name

    def run(self, task: str, workdir: Path, env: dict[str, str], out_dir: Path) -> AgentRun:
        raise NotImplementedError


def ollama_base() -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip().rstrip("/")
    return host if host.startswith("http") else f"http://{host}"


SYSTEM_EXTRA = """
Rules for this task:
- Requirements files are read-only. Do NOT install, upgrade or downgrade packages.
- You may change how tests are SET UP when the upgrade requires it: imports,
  fixtures, mocks/fakes, conftest.py, version checks in test setup.
- Do NOT change what tests CHECK: no removing or editing assertions or
  pytest.raises blocks, no deleting or renaming tests, no skip/xfail, no test
  selection in the pytest configuration. Such edits are detected and rejected.
- Do NOT make project code behave differently under tests (no checks for mocks or
  for pytest in source code).
- Change the project's own source code as little as needed, keeping its behaviour.

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
        # A hung model server: give up in about 15 minutes (3 tries of 300 s),
        # not mini-swe-agent's default 10 tries of 600 s (about 2 hours).
        os.environ.setdefault("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "3")
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
            kwargs.update({"api_base": ollama_base(), "num_ctx": self.num_ctx,
                           "timeout": 300})
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


# ------------------------------------------------------------------ CLI agents

def _ollama_model(model: str) -> str:
    """'ollama_chat/qwen3-coder:30b' -> 'qwen3-coder:30b'."""
    return model.split("/", 1)[1] if "/" in model else model


def _count_events(lines: list[str]) -> tuple[int, int, int]:
    """(tool calls, prompt tokens, completion tokens) from a JSONL event stream.
    Event shapes differ per tool and version, so this reads them loosely."""
    import json
    steps = prompt = completion = 0

    def walk(obj):
        nonlocal prompt, completion
        if isinstance(obj, dict):
            for key in ("usage", "tokens"):
                u = obj.get(key)
                if isinstance(u, dict):
                    prompt += int(u.get("input_tokens") or u.get("input") or
                                  u.get("prompt_tokens") or 0)
                    completion += int(u.get("output_tokens") or u.get("output") or
                                      u.get("completion_tokens") or 0)
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = str(event.get("type", ""))
        item = event.get("item") or event.get("part") or {}
        item_kind = str(item.get("type", "")) if isinstance(item, dict) else ""
        if (kind in ("item.completed", "tool_use") and
                (item_kind in ("command_execution", "file_change", "mcp_tool_call", "tool")
                 or kind == "tool_use")):
            steps += 1
        if kind in ("turn.completed", "step_finish"):
            walk(event)
    return steps, prompt, completion


class CliAgent(CodeAgent):
    """A coding agent run as its own program (Codex CLI, OpenCode), headless,
    against the job's Ollama server. Same task text and rules as mini-SWE-agent;
    the guard around it (locked tests, no pip, checks after) is the same too.
    These tools have no step limit, so the budget is wall-clock time."""

    binary_env = ""
    binary_name = ""

    def __init__(self, model: str = "ollama_chat/qwen3-coder:30b", step_limit: int = 60,
                 wall_time_s: int = 3600, num_ctx: int = 65536, **_):
        self.model = model
        self.step_limit = step_limit          # recorded only; see class docstring
        self.wall_time_s = wall_time_s
        self.num_ctx = num_ctx

    def binary(self) -> str:
        import shutil
        found = os.environ.get(self.binary_env) or shutil.which(self.binary_name)
        if not found:
            raise FileNotFoundError(f"{self.binary_name} not found: put it on PATH or set "
                                    f"{self.binary_env}")
        return found

    def command(self, task: str, workdir: Path, out_dir: Path) -> tuple[list[str], dict]:
        raise NotImplementedError

    def run(self, task: str, workdir: Path, env: dict[str, str], out_dir: Path) -> AgentRun:
        import signal
        out_dir.mkdir(parents=True, exist_ok=True)
        trajectory = out_dir / "trajectory.jsonl"
        run = AgentRun(trajectory=str(trajectory))
        prompt = task + "\n" + SYSTEM_EXTRA
        try:
            cmd, extra_env = self.command(prompt, workdir, out_dir)
        except Exception as exc:
            run.exit_status, run.error = "crashed", f"{type(exc).__name__}: {exc}"
            return run
        full_env = {**os.environ, **env, **extra_env}
        started = time.time()
        with open(trajectory, "w") as out, open(out_dir / "agent_stderr.txt", "w") as err:
            proc = subprocess.Popen(cmd, cwd=workdir, env=full_env, stdout=out, stderr=err,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                code = proc.wait(timeout=self.wall_time_s)
                run.exit_status = "Submitted" if code == 0 else f"exit_{code}"
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                run.exit_status = "TimeExceeded"
        run.seconds = round(time.time() - started, 1)
        # The agent's private home holds thousands of plugin and cache files;
        # its transcript is already in trajectory.jsonl, so drop the rest.
        for home in ("codex_home", "opencode_home"):
            shutil.rmtree(out_dir / home, ignore_errors=True)
        lines = trajectory.read_text(errors="replace").splitlines()
        run.steps, run.prompt_tokens, run.completion_tokens = _count_events(lines)
        if run.exit_status.startswith("exit_"):
            run.error = (out_dir / "agent_stderr.txt").read_text(errors="replace")[-1500:]
        return run


class CodexCli(CliAgent):
    """OpenAI Codex CLI (``codex exec``) on a local model through Ollama."""

    name = "codex"
    binary_env, binary_name = "STRIDE_CODEX_BIN", "codex"

    def command(self, task, workdir, out_dir):
        home = out_dir / "codex_home"
        home.mkdir(parents=True, exist_ok=True)
        provider = {
            "model_provider": '"stride"',
            "model_providers.stride.name": '"ollama"',
            "model_providers.stride.base_url": f'"{ollama_base()}/v1"',
            "model_providers.stride.wire_api": '"responses"',
            "shell_environment_policy.inherit": '"all"',
            "model_context_window": str(self.num_ctx),
        }
        cmd = [self.binary(), "exec", "--ignore-user-config", "--ignore-rules",
               "--skip-git-repo-check", "--ephemeral", "--json", "--color", "never",
               # The guard around the agent does the sandboxing (read-only tests,
               # pip blocked, checks afterwards), as for every agent.
               "--dangerously-bypass-approvals-and-sandbox",
               "-C", str(workdir), "-m", _ollama_model(self.model),
               "-o", str(out_dir / "last_message.txt")]
        for key, value in provider.items():
            cmd += ["-c", f"{key}={value}"]
        cmd.append(task)
        return cmd, {"CODEX_HOME": str(home), "NO_COLOR": "1"}


class OpenCode(CliAgent):
    """OpenCode (``opencode run``) on a local model through Ollama."""

    name = "opencode"
    binary_env, binary_name = "STRIDE_OPENCODE_BIN", "opencode"

    def command(self, task, workdir, out_dir):
        import json
        home = out_dir / "opencode_home"
        for sub in ("config", "data", "state"):
            (home / sub).mkdir(parents=True, exist_ok=True)
        model = _ollama_model(self.model)
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {"ollama": {
                "npm": "@ai-sdk/openai-compatible", "name": "Ollama",
                "options": {"baseURL": f"{ollama_base()}/v1"},
                "models": {model: {"name": model, "tools": True,
                                   "limit": {"context": self.num_ctx, "output": 8192}}}}},
            "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
            "share": "disabled", "autoupdate": False,
        }
        path = home / "opencode.json"
        path.write_text(json.dumps(config, indent=2))
        cmd = [self.binary(), "run", "--model", f"ollama/{model}", "--format", "json",
               "--auto", "--pure", "--dir", str(workdir), task]
        return cmd, {"OPENCODE_CONFIG": str(path),
                     "XDG_CONFIG_HOME": str(home / "config"),
                     "XDG_DATA_HOME": str(home / "data"),
                     "XDG_STATE_HOME": str(home / "state"),
                     "OPENCODE_DISABLE_AUTOUPDATE": "1"}


class NoopAgent(CodeAgent):
    """Changes nothing. For smoke tests of the harness without a model."""
    name = "noop"

    def __init__(self, **_):
        self.model = ""

    def run(self, task, workdir, env, out_dir) -> AgentRun:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trajectory.jsonl").write_text("")
        return AgentRun(exit_status="Submitted", trajectory=str(out_dir / "trajectory.jsonl"))


AGENTS = {"mini-swe-agent": MiniSweAgent, "codex": CodexCli, "opencode": OpenCode,
          "noop": NoopAgent}


def build(name: str, **kwargs) -> CodeAgent:
    if name not in AGENTS:
        raise KeyError(f"unknown code agent {name!r}; known: {sorted(AGENTS)}")
    return AGENTS[name](**kwargs)
