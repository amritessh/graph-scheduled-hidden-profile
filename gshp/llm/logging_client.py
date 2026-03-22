"""
Wrap any LLM client to record every ``complete()`` for experiment artifacts.

Same role as prompt/call capture in prior lab code: every API call is logged with
system prompt, message list, raw response text, and optional metadata (dyad, turn, …).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from gshp.session import LLMClient


class LoggingLLMClient:
    """
    Delegates to an inner client and appends one JSON-serializable dict per ``complete()``.

    Use ``set_call_meta(**kwargs)`` immediately before each ``complete()`` to attach
    fields like ``kind``, ``dyad_u``, ``turn``, ``agent_id``.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []
        self._pending: dict[str, Any] = {}

    def set_call_meta(self, **kwargs: Any) -> None:
        self._pending = dict(kwargs)

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        out = self._inner.complete(system, messages)
        rec: dict[str, Any] = {
            "seq": len(self.calls),
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "system": system,
            "messages": messages,
            "response": out,
        }
        rec.update(self._pending)
        self.calls.append(rec)
        self._pending = {}
        return out
