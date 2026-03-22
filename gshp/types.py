"""Shared types for transcripts and runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # "agent_3" or "system" / "user" depending on convention
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DyadTranscript(BaseModel):
    u: int
    v: int
    round_index: int
    round_label: str
    messages: list[Message] = Field(default_factory=list)


class RunManifest(BaseModel):
    schedule: str
    l: int
    k: int
    model: str | None = None
    seed: int | None = None


class ExperimentRun(BaseModel):
    manifest: RunManifest
    dyads: list[DyadTranscript] = Field(default_factory=list)
    # Filled later: per-agent private context keys, final votes, etc.
    notes: dict[str, Any] = Field(default_factory=dict)
