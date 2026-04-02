"""
Wrap any LLM client to record every ``complete()`` for experiment artifacts.

Same role as prompt/call capture in prior lab code: every API call is logged with
system prompt, message list, raw response text, and optional metadata (dyad, turn, …).
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

from gshp.session import LLMClient


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. Qwen3)."""
    stripped = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    return stripped if stripped else text


def _serialize_openai_completion(obj: object) -> dict[str, Any] | None:
    """Best-effort JSON-serializable dict (like AI-GBS ``raw_api_*.json`` payloads)."""
    if obj is None:
        return None
    md = getattr(obj, "model_dump", None)
    if callable(md):
        try:
            d = md(mode="json")
            if isinstance(d, dict):
                return d
        except TypeError:
            try:
                d2 = md()
                if isinstance(d2, dict):
                    return d2
            except Exception:
                pass
        except Exception:
            try:
                d2 = md()
                if isinstance(d2, dict):
                    return d2
            except Exception:
                pass
    return {"_note": "could not model_dump", "type": type(obj).__name__}


def _usage_snapshot(completion: object | None) -> dict[str, Any] | None:
    if completion is None:
        return None
    u = getattr(completion, "usage", None)
    if u is None:
        return None
    md = getattr(u, "model_dump", None)
    if callable(md):
        try:
            d = md(mode="json")
            if isinstance(d, dict):
                return d
        except TypeError:
            try:
                d2 = md()
                if isinstance(d2, dict):
                    return d2
            except Exception:
                pass
        except Exception:
            pass
    return None


class LoggingLLMClient:
    """
    Delegates to an inner client and appends one JSON-serializable dict per ``complete()``.

    Use ``set_call_meta(**kwargs)`` immediately before each ``complete()`` to attach
    fields like ``kind``, ``dyad_u``, ``turn``, ``agent_id``.
    """

    def __init__(self, inner: LLMClient, *, capture_raw_completion: bool = True) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []
        self._pending: dict[str, Any] = {}
        self.capture_raw_completion = capture_raw_completion

    def set_call_meta(self, **kwargs: Any) -> None:
        self._pending = dict(kwargs)

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        t0 = time.perf_counter()
        raw_out = self._inner.complete(system, messages)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        out = _strip_thinking(raw_out)
        raw = getattr(self._inner, "last_completion", None)
        rec: dict[str, Any] = {
            "seq": len(self.calls),
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "latency_ms": round(latency_ms, 3),
            "system": system,
            "messages": messages,
            "response": out,
            "raw_response": raw_out,  # keep full output with thinking for artifacts
        }
        rec.update(self._pending)
        usage = _usage_snapshot(raw)
        if usage is not None:
            rec["usage"] = usage
        if self.capture_raw_completion:
            serialized = _serialize_openai_completion(raw)
            if serialized is not None:
                rec["openai_completion"] = serialized
        self.calls.append(rec)
        self._pending = {}
        return out
