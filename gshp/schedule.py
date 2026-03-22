"""Communication schedules: order of intra- vs inter-community dyads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from gshp.graph.caveman import CavemanTopology


class ScheduleName(str, Enum):
    """Which communication phase comes first (Momennejad-style ordering lever)."""

    WITHIN_FIRST = "within_first"  # all intra-clique dyads, then inter-clique
    CROSS_FIRST = "cross_first"  # inter-clique dyads first, then intra


@dataclass(frozen=True)
class CommunicationRound:
    """One batch of simultaneous allowed dyads (non-overlapping nodes per sub-slot optional)."""

    index: int
    label: str
    """Undirected edges (u, v) with u < v to run in this round."""
    edges: tuple[tuple[int, int], ...]


def build_two_phase_schedule(
    topo: CavemanTopology,
    name: ScheduleName | Literal["within_first", "cross_first"],
) -> tuple[CommunicationRound, ...]:
    """
    Minimal v0 schedule: two phases — either intra then inter, or inter then intra.

    Within each phase, all edges of that type are listed in one round (your runner can
    later split into parallel non-conflicting matchings if you want strict simultaneity).
    """
    if isinstance(name, str):
        name = ScheduleName(name)

    intra = topo.edges_for_round_kind("intra")
    inter = topo.edges_for_round_kind("inter")

    if name == ScheduleName.WITHIN_FIRST:
        return (
            CommunicationRound(0, "intra_community", tuple(intra)),
            CommunicationRound(1, "inter_community", tuple(inter)),
        )
    if name == ScheduleName.CROSS_FIRST:
        return (
            CommunicationRound(0, "inter_community", tuple(inter)),
            CommunicationRound(1, "intra_community", tuple(intra)),
        )
    raise ValueError(name)
