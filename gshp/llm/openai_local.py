"""
OpenAI-compatible chat: **one model string** picks the backend (like a unified ``chat(model, ...)`` router).

**Local / self-hosted** (OpenAI-compatible HTTP):

- ``vllm:8000/my-served-model-name`` → ``http://127.0.0.1:8000/v1``
- ``vllm/my-model`` → ``$VLLM_BASE_URL`` (default ``http://127.0.0.1:8000/v1``) + model id
- ``localhost:8000/my-model`` → ``http://127.0.0.1:8000/v1`` + model id

**Cloud APIs** (via :func:`make_llm_client`):

- ``gpt-4o-mini`` or any bare OpenAI model id → ``https://api.openai.com/v1`` + ``OPENAI_API_KEY``
- ``openai/gpt-4o-mini`` → same (optional explicit prefix)
- ``openrouter/organization/model`` → OpenRouter + ``OPENROUTER_API_KEY``

Ollama’s native ``/api/generate`` is a different API; use an OpenAI-compatible wrapper or extend separately.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

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
                "Bare localhost:PORT without /model is not handled here."
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


def make_llm_client(
    model_string: str,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 3,
) -> OpenAICompatibleChat:
    """
    Single entry point: **same CLI flag** for local vLLM, OpenAI cloud, or OpenRouter.

    Resolution order:

    1. ``openrouter/...`` → OpenRouter (needs ``OPENROUTER_API_KEY``).
    2. ``vllm:...``, ``vllm/...``, ``localhost:.../...`` → :func:`parse_model_spec`.
    3. Anything else → OpenAI official API (needs ``OPENAI_API_KEY``), after stripping optional ``openai/`` prefix.
    """
    s = model_string.strip()
    if s.startswith("openrouter/"):
        model_id = s[len("openrouter/") :].strip()
        if not model_id:
            raise ValueError("openrouter/ requires a model id, e.g. openrouter/qwen/qwen-2.5-72b-instruct")
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("Set OPENROUTER_API_KEY for openrouter/... model strings.")
        return OpenAICompatibleChat(
            base_url="https://openrouter.ai/api/v1",
            model=model_id,
            api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    try:
        return OpenAICompatibleChat.from_model_spec(
            s,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
    except ValueError:
        pass

    openai_model = s[7:] if s.startswith("openai/") else s
    if not openai_model:
        raise ValueError(f"Invalid model string: {model_string!r}")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            f"Could not parse {model_string!r} as a local spec (vllm:… or localhost:port/model). "
            "For OpenAI Cloud, set OPENAI_API_KEY and pass a model id such as gpt-4o-mini."
        )
    return OpenAICompatibleChat(
        base_url="https://api.openai.com/v1",
        model=openai_model,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


@dataclass
class OpenAICompatibleChat:
    """
    Sync chat client implementing :class:`gshp.session.LLMClient`.

    ``api_key``: vLLM often ignores it; use ``dummy`` or set ``OPENAI_API_KEY`` if your
    gateway requires it.

    ``last_completion`` is set after each successful API call (for optional raw dumps in
    :class:`gshp.llm.logging_client.LoggingLLMClient`).
    """

    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float = 120.0
    max_retries: int = 3
    last_completion: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        key = self.api_key if self.api_key is not None else os.environ.get("OPENAI_API_KEY", "dummy")
        self._client = OpenAI(
            base_url=self.base_url.rstrip("/"),
            api_key=key,
            timeout=self.timeout,
        )

    def clone(self) -> "OpenAICompatibleChat":
        """Return a new instance with the same config and a fresh underlying client.

        Use this when running dyads in parallel — each thread needs its own
        ``last_completion`` state so they don't race reading each other's response.
        """
        return OpenAICompatibleChat(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_model_spec(
        cls,
        model_string: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> OpenAICompatibleChat:
        """Local OpenAI-compatible servers only. For cloud routing use :func:`make_llm_client`."""
        base, model = parse_model_spec(model_string)
        return cls(
            base_url=base,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
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

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                self.last_completion = resp
                choice = resp.choices[0].message
                return (choice.content or "").strip()
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    wait = (2**attempt) + random.uniform(0, 0.5)
                    time.sleep(wait)
        assert last_err is not None
        raise last_err
