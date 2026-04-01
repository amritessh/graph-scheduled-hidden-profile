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
    round_sub_index: int = 0,
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
        round_sub_index=round_sub_index,
        messages=msgs,
    )


def run_dyad_llm(
    u: int,
    v: int,
    *,
    round_index: int,
    round_label: str,
    round_sub_index: int = 0,
    client: LLMClient,
    system_u: str,
    system_v: str,
    turns: int = 6,
) -> DyadTranscript:
    """
    Alternating dyad. Each turn the **current speaker**'s system prompt is used; the user
    message carries the dialogue so far (plain text).
    """
    transcript = DyadTranscript(
        u=min(u, v),
        v=max(u, v),
        round_index=round_index,
        round_label=round_label,
        round_sub_index=round_sub_index,
        messages=[],
    )
    lines: list[str] = []

    for t in range(turns):
        speaker = u if t % 2 == 0 else v
        system = system_u if speaker == u else system_v
        so_far = "\n".join(lines) if lines else "(No messages yet — open the discussion.)"
        user_msg = (
            f"You are Agent {speaker}. You are speaking privately with one colleague.\n\n"
            f"Conversation so far:\n{so_far}\n\n"
            "Speak next. Reply in 1–3 sentences. Do not prefix with your agent number; "
            "just say what you want to say."
        )
        _meta = getattr(client, "set_call_meta", None)
        if callable(_meta):
            _meta(
                kind="dyad",
                round_index=round_index,
                round_label=round_label,
                round_sub_index=round_sub_index,
                dyad_u=u,
                dyad_v=v,
                turn=t,
                speaker=speaker,
            )
        content = client.complete(system, [{"role": "user", "content": user_msg}])
        line = f"Agent {speaker}: {content}"
        lines.append(line)
        transcript.messages.append(
            Message(
                role=f"agent_{speaker}",
                content=content.strip(),
                metadata={"turn": t},
            )
        )

    return transcript
