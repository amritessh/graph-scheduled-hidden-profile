"""
OpenAI-compatible chat for local vLLM — conventions aligned with ``AI-GBS/llm_run.py``.

Use the same model strings you use there:

- ``vllm:8000/my-served-model-name`` → ``http://127.0.0.1:8000/v1``
- ``vllm/my-model`` → ``$VLLM_BASE_URL`` (default ``http://127.0.0.1:8000/v1``) + model id
- ``localhost:8000/my-model`` → same routing as ``vllm:8000/...`` (chat completions)

Ollama-style ``localhost:5001`` *without* a ``/model`` path uses a different API
(``/api/generate``) in AI-GBS; this module only implements **OpenAI-compatible**
``/v1/chat/completions`` (vLLM, TensorRT-LLM OpenAI server, etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


def parse_model_spec(model_string: str) -> tuple[str, str]:
    """
    Return ``(base_url, model_id)`` for OpenAI-compatible servers.

    ``base_url`` should be the OpenAI client base (…/v1, no trailing slash).
    """
    s = model_string.strip()
    if s.startswith("localhost:"):
        rest = s[len("localhost:") :]
        if "/" not in rest:
            raise ValueError(
                "For OpenAI-compatible local servers use localhost:PORT/MODEL_ID "
                "(e.g. localhost:8000/Qwen/Qwen3-8B). "
                "Bare localhost:PORT is Ollama-style in AI-GBS, not handled here."
            )
        port, model_id = rest.split("/", 1)
        return f"http://127.0.0.1:{port}/v1", model_id

    if s.startswith("vllm:"):
        rest = s[5:]
        if "/" not in rest:
            base = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
            return base, rest
        left, model_name = rest.split("/", 1)
        if ":" in left:
            host, port = left.rsplit(":", 1)
            base_url = f"http://{host}:{port}/v1"
        else:
            base_url = f"http://127.0.0.1:{left}/v1"
        return base_url, model_name

    if s.startswith("vllm/"):
        model_name = s[5:]
        base = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        return base, model_name

    raise ValueError(
        f"Unknown model spec: {model_string!r}. "
        "Expected vllm:PORT/model, vllm/model, or localhost:PORT/model."
    )


@dataclass
class OpenAICompatibleChat:
    """
    Sync chat client implementing :class:`gshp.session.LLMClient`.

    ``api_key``: vLLM often ignores it; use ``dummy`` or set ``OPENAI_API_KEY`` if your
    gateway requires it.
    """

    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        key = self.api_key if self.api_key is not None else os.environ.get("OPENAI_API_KEY", "dummy")
        self._client = OpenAI(base_url=self.base_url.rstrip("/"), api_key=key)

    @classmethod
    def from_model_spec(
        cls,
        model_string: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ) -> OpenAICompatibleChat:
        base, model = parse_model_spec(model_string)
        return cls(
            base_url=base,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        """``messages`` roles: user | assistant (no system — pass via ``system``)."""
        msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
        msgs.extend(messages)
        kwargs: dict = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        return (choice.content or "").strip()
