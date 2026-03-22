"""Dyadic conversation session — plug in real LLM here."""

from __future__ import annotations

from typing import Protocol

from gshp.types import DyadTranscript, Message


class LLMClient(Protocol):
    """OpenAI-compatible chat is enough for vLLM."""

    def complete(self, system: str, messages: list[dict[str, str]]) -> str: ...


def run_dyad_stub(
    u: int,
    v: int,
    *,
    round_index: int,
    round_label: str,
    turns: int = 4,
) -> DyadTranscript:
    """Placeholder dialogue until vLLM is wired."""
    msgs: list[Message] = []
    for t in range(turns):
        speaker = u if t % 2 == 0 else v
        msgs.append(
            Message(
                role=f"agent_{speaker}",
                content=f"[stub turn {t + 1}] no LLM yet",
                metadata={"turn": t},
            )
        )
    return DyadTranscript(
        u=min(u, v),
        v=max(u, v),
        round_index=round_index,
        round_label=round_label,
        messages=msgs,
    )


def run_dyad_llm(
    u: int,
    v: int,
    *,
    round_index: int,
    round_label: str,
    client: LLMClient,
    system_u: str,
    system_v: str,
    turns: int = 6,
) -> DyadTranscript:
    """Alternating dyad; each side gets system prompt + shared thread as user/assistant."""
    # Implement in a follow-up PR once you have prompts + OpenAI client
    raise NotImplementedError("Wire OpenAI-compatible client + prompts")
