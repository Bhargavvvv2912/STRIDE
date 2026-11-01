"""Code agents STRIDE can plug in. Each gets one task and a locked workspace.

Only mini-SWE-agent is wired up for the pilot. Others (OpenHands, Aider,
SWE-agent) implement the same ``CodeAgent.run`` and register in ``AGENTS``.
"""
from __future__ import annotations

import os
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
"""


class MiniSweAgent(CodeAgent):
    """mini-SWE-agent (text-based actions, works with local models via litellm)."""

    name = "mini-swe-agent"

    def __init__(self, model: str = "ollama_chat/qwen2.5-coder:14b", step_limit: int = 60,
                 wall_time_s: int = 3600, command_timeout_s: int = 900,
                 num_ctx: int = 32768, temperature: float = 0.0):
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
        from minisweagent.agents.default import DefaultAgent
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
        agent = DefaultAgent(model, environment, step_limit=self.step_limit,
                             cost_limit=0.0, wall_time_limit_seconds=self.wall_time_s,
                             output_path=trajectory, **agent_cfg)
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
