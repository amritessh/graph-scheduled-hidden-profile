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
    round_sub_index: int = Field(
        default=0,
        description="Parallel matching layer within round_index (0 if schedule not layer-expanded).",
    )
    messages: list[Message] = Field(default_factory=list)


class RunManifest(BaseModel):
    schedule: str
    l: int
    k: int
    model: str | None = None
    seed: int | None = None
    task_id: str = "hiring_v1"
    information_condition: str = "hidden_profile"
    tom_bridge: bool = False
    parallel_dyad_layers: bool = Field(
        default=False,
        description="If true, phases are split into parallel matching layers (see docs/algorithms.md).",
    )


class AgentDecision(BaseModel):
    agent_id: int
    choice: str | None = Field(
        default=None, description="X, Y, or Z if parsed; None if parse failed"
    )
    justification: str = ""
    raw_response: str = ""


class GroupDecision(BaseModel):
    """One agent's recommendation after seeing all individual decisions (deliberation round)."""

    agent_id: int
    choice: str | None = Field(default=None, description="X, Y, or Z if parsed; None if parse failed")
    justification: str = ""
    raw_response: str = ""


class GroupDeliberationResult(BaseModel):
    """Outcome of the all-agents broadcast round that follows individual decisions."""

    group_decisions: list[GroupDecision] = Field(default_factory=list)
    group_consensus: str | None = Field(
        default=None,
        description="Majority vote across group_decisions; None if no decisions recorded.",
    )
    unanimous: bool = False


class ExperimentRun(BaseModel):
    manifest: RunManifest
    dyads: list[DyadTranscript] = Field(default_factory=list)
    final_decisions: list[AgentDecision] = Field(default_factory=list)
    deliberation: GroupDeliberationResult | None = Field(
        default=None,
        description="Group deliberation round result; None if not run.",
    )
    notes: dict[str, Any] = Field(default_factory=dict)
