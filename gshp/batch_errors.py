"""Classify exceptions for structured batch logging."""

from __future__ import annotations

import json
from typing import Any


def classify_error(exc: BaseException) -> str:
    name = type(exc).__name__
    msg = str(exc).lower()

    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, OSError) and "timed out" in msg:
        return "timeout"

    try:
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return "timeout"
        if isinstance(exc, httpx.HTTPStatusError):
            return f"http_{exc.response.status_code}"
    except ImportError:
        pass

    try:
        import openai

        if isinstance(exc, openai.APITimeoutError):
            return "timeout"
        if isinstance(exc, openai.APIConnectionError):
            return "connection"
        if isinstance(exc, openai.RateLimitError):
            return "rate_limit"
        if isinstance(exc, openai.AuthenticationError):
            return "auth"
        if isinstance(exc, openai.APIStatusError):
            return f"api_status_{getattr(exc, 'status_code', 'unknown')}"
        if isinstance(exc, openai.OpenAIError):
            return "openai_error"
    except ImportError:
        pass

    if isinstance(exc, ValueError):
        return "value_error"
    if isinstance(exc, KeyError):
        return "key_error"
    if isinstance(exc, json.JSONDecodeError):
        return "json_decode"
    return f"exception:{name}"
