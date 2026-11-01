"""Backend-agnostic LLM clients with token accounting and deterministic decoding.

Three things v1 could not do and this can:

1. **Prove an arm is LLM-free.** ``NullLLM`` raises on any call, so a run
   configured without the LLM tier cannot quietly reach the API. v1 called
   ``summarize_error`` from inside the deterministic tiers, so its published
   "no LLM" ablation still hit the network on every failure.
2. **Count tokens and cost.** v1 had no counter; the paper quoted list prices.
3. **Swap backends.** ``OpenAICompatibleLLM`` targets vLLM / Ollama / TGI and any
   hosted OpenAI-shaped endpoint, so open models can be compared against a
   hosted one on identical prompts.

Temperature defaults to 0 everywhere so a rerun reproduces the plan.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from aura.telemetry import LLMUsage


# The planner's reply schema. Backends that support constrained decoding are
# given this so malformed JSON becomes impossible; version *hallucination* is a
# separate problem that no grammar can prevent and that the index check catches.
PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "resolvable": {"type": "boolean"},
        "reason": {"type": "string"},
        "plausible": {"type": "boolean"},
        "proposed_plan": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["resolvable", "plausible", "proposed_plan"],
}


class LLMUnavailable(RuntimeError):
    """Raised when a run without the LLM tier attempts a model call."""


@dataclass
class LLMResponse:
    text: str
    usage: LLMUsage


class LLMClient(Protocol):
    model: str

    def generate(self, prompt: str, *, max_output_tokens: int = 2048,
                 schema: dict | None = None) -> LLMResponse: ...


def _approx_tokens(text: str) -> int:
    """Fallback when a backend reports no usage: ~4 characters per token."""
    return max(1, len(text) // 4)


class NullLLM:
    """The client used by deterministic-only arms. Any call is a hard error."""

    model = "none"

    def __init__(self, reason: str = "this arm runs without the LLM tier"):
        self.reason = reason
        self.calls_attempted = 0

    def generate(self, prompt: str, *, max_output_tokens: int = 2048,
                 schema: dict | None = None) -> LLMResponse:
        self.calls_attempted += 1
        raise LLMUnavailable(
            f"LLM call attempted but {self.reason}. This is a bug in the caller: "
            f"deterministic tiers must not depend on the model."
        )


class GeminiLLM:
    """google-genai backend, pinned to temperature 0 with real usage metadata."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash",
                 temperature: float = 0.0, timeout_s: int = 120):
        from google import genai  # imported lazily so no-LLM arms need no SDK

        self._genai = genai
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s

    def generate(self, prompt: str, *, max_output_tokens: int = 2048,
                 schema: dict | None = None) -> LLMResponse:
        from google.genai import types

        usage = LLMUsage(model=self.model)
        start = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=max_output_tokens,
                    # No tools, no retrieval, no web access: the model sees only
                    # the prompt. Reviewer B asked for this to be explicit.
                    tools=[],
                    **({"response_mime_type": "application/json",
                        "response_schema": schema} if schema else {}),
                ),
            )
            text = (response.text or "").strip()
            meta = getattr(response, "usage_metadata", None)
            usage.prompt_tokens = getattr(meta, "prompt_token_count", 0) or _approx_tokens(prompt)
            usage.completion_tokens = getattr(meta, "candidates_token_count", 0) or _approx_tokens(text)
        except Exception as exc:
            usage.ok, usage.error = False, f"{type(exc).__name__}: {exc}"
            text = ""
        usage.latency_s = round(time.time() - start, 3)
        return LLMResponse(text=text, usage=usage.finalize())


class OpenAICompatibleLLM:
    """Any OpenAI-shaped ``/v1/chat/completions`` endpoint.

    Covers vLLM, Ollama, text-generation-inference and hosted providers, which
    is how the open-model backend comparison is run on identical prompts.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed",
                 temperature: float = 0.0, timeout_s: int = 600, seed: int | None = 0,
                 structured_output: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.seed = seed
        self.structured_output = structured_output

    def generate(self, prompt: str, *, max_output_tokens: int = 2048,
                 schema: dict | None = None) -> LLMResponse:
        import urllib.error
        import urllib.request

        usage = LLMUsage(model=self.model)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_output_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if schema and self.structured_output:
            # Grammar-constrained decoding. Supported by llama.cpp, Ollama, vLLM
            # and MLX; small open models benefit from it far more than large
            # hosted ones, which is itself worth reporting.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "plan", "schema": schema, "strict": True},
            }

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        start = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read())
            text = body["choices"][0]["message"]["content"].strip()
            counts = body.get("usage") or {}
            usage.prompt_tokens = counts.get("prompt_tokens") or _approx_tokens(prompt)
            usage.completion_tokens = counts.get("completion_tokens") or _approx_tokens(text)
        except Exception as exc:
            usage.ok, usage.error = False, f"{type(exc).__name__}: {exc}"
            text = ""
        usage.latency_s = round(time.time() - start, 3)
        return LLMResponse(text=text, usage=usage.finalize())


class ScriptedLLM:
    """Deterministic stub for tests: replays canned responses in order."""

    def __init__(self, responses: list[str], model: str = "scripted"):
        self.responses = list(responses)
        self.model = model
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_output_tokens: int = 2048,
                 schema: dict | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else ""
        return LLMResponse(
            text=text,
            usage=LLMUsage(model=self.model, prompt_tokens=_approx_tokens(prompt),
                           completion_tokens=_approx_tokens(text), latency_s=0.0).finalize(),
        )


# Local servers that speak the OpenAI chat API. All run on Apple Silicon.
LOCAL_ENDPOINTS = {
    "ollama": "http://localhost:11434/v1",
    "llamacpp": "http://localhost:8080/v1",
    "lmstudio": "http://localhost:1234/v1",
    "mlx": "http://localhost:8080/v1",
}


def _local_endpoint(kind: str) -> str:
    """The server address for a local backend, honouring ``OLLAMA_HOST``.

    Two jobs on one cluster node cannot both serve Ollama on port 11434, so each
    job starts its own server on its own port and exports ``OLLAMA_HOST`` -- the
    same variable the ``ollama`` command itself reads.
    """
    if kind == "ollama" and os.environ.get("OLLAMA_HOST"):
        host = os.environ["OLLAMA_HOST"].strip().rstrip("/")
        if not host.startswith(("http://", "https://")):
            host = "http://" + host
        return host + "/v1"
    return LOCAL_ENDPOINTS[kind]


def build_client(spec: str, temperature: float = 0.0) -> LLMClient:
    """Construct a client from a config string.

    ``none``                          -> NullLLM (deterministic-only arms)
    ``gemini:<model>``                -> hosted Gemini
    ``openai:<base_url>:<model>``     -> any OpenAI-compatible server
    ``ollama:<model>``                -> local Ollama   (localhost:11434)
    ``llamacpp:<model>``              -> local llama.cpp server
    ``lmstudio:<model>``              -> local LM Studio
    ``mlx:<model>``                   -> local mlx_lm.server
    """
    if spec in ("", "none", "null"):
        return NullLLM()
    kind, _, rest = spec.partition(":")
    if kind == "gemini":
        return GeminiLLM(model=rest or "gemini-2.5-flash", temperature=temperature)
    if kind in LOCAL_ENDPOINTS:
        if not rest:
            raise ValueError(f"{kind} spec needs a model name, got {spec!r}")
        return OpenAICompatibleLLM(base_url=_local_endpoint(kind), model=rest,
                                   temperature=temperature)
    if kind == "openai":
        base_url, _, model = rest.rpartition(":")
        if not base_url:
            raise ValueError(f"openai spec needs base_url and model, got {spec!r}")
        return OpenAICompatibleLLM(base_url=base_url, model=model, temperature=temperature)
    raise ValueError(f"unknown LLM spec {spec!r}")
