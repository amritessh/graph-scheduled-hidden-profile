"""Deterministic fake LLM for pipeline tests without a GPU."""

from __future__ import annotations

import json
import re


class StubLLM:
    """Returns canned dialogue and a valid final JSON decision."""

    def __init__(self, *, final_choice: str = "X", dyad_reply: str | None = None) -> None:
        self.final_choice = final_choice.upper()
        if self.final_choice not in ("X", "Y", "Z"):
            raise ValueError("final_choice must be X, Y, or Z")
        self.dyad_reply = dyad_reply or (
            "From what I know, X looks like the strongest hire; happy to hear your view."
        )

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        last = messages[-1]["content"] if messages else ""
        if "JSON object" in last or '"choice"' in last:
            return json.dumps(
                {
                    "choice": self.final_choice,
                    "justification": "Stub client: predetermined answer for testing.",
                }
            )
        return self.dyad_reply


def parse_choice_json(raw: str) -> tuple[str | None, str]:
    """
    Extract choice X|Y|Z and justification from model output.
    Returns (choice_or_none, raw_trimmed).
    """
    text = raw.strip()
    # Strip optional ```json fences
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        c = str(data.get("choice", "")).upper()
        if c in ("X", "Y", "Z"):
            return c, str(data.get("justification", ""))
    except json.JSONDecodeError:
        pass
    # Fallback: first occurrence of X, Y, Z as word
    m = re.search(r'\b([XYZ])\b', text.upper())
    if m:
        return m.group(1), text
    return None, text
