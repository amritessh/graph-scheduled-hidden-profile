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
    """
    One batch of dyads for phase ``index`` (intra vs inter).

    ``edges`` are undirected (u, v) with u < v. ``sub_index`` numbers **parallel layers**
    when using :func:`expand_schedule_parallel_matchings` (each layer is a matching).
    """

    index: int
    label: str
    edges: tuple[tuple[int, int], ...]
    sub_index: int = 0


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

    intra = _augment_intra_edges_for_exposure(topo, topo.edges_for_round_kind("intra"))
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


def _augment_intra_edges_for_exposure(
    topo: CavemanTopology,
    intra_edges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Keep phase edge *types* intact while matching intended per-agent exposure.

    In the default 3x3 ring (l=3, k=3), bridge agents naturally have 4 total dyads
    (2 intra + 2 inter), while non-bridge agents only have 2 (intra only).
    To match the study protocol target (4 conversations per agent), we repeat the
    non-bridge pair inside each cluster twice. This adds +2 dyads for each non-bridge
    agent without introducing any extra inter-cluster links.
    """
    if topo.k != 3:
        return intra_edges

    edges = list(intra_edges)
    for cluster in topo.communities:
        bridge = max(cluster)  # ring bridge endpoint for this cluster
        non_bridge = sorted(n for n in cluster if n != bridge)
        if len(non_bridge) != 2:
            continue
        pair = (non_bridge[0], non_bridge[1])
        # Two repeats: each non-bridge gets +2 dyads (2 -> 4 total).
        edges.append(pair)
        edges.append(pair)
    return edges


def expand_schedule_parallel_matchings(
    rounds: tuple[CommunicationRound, ...],
) -> tuple[CommunicationRound, ...]:
    """
    Replace each phase round with several sub-rounds: edges partitioned into **matchings**
    (no shared endpoints within a sub-round). See ``docs/algorithms.md``.

    Dyads still run **sequentially** in the runner; layers document which conversations could
    run in parallel without agent double-booking.
    """
    from gshp.matching_schedule import partition_edges_into_matching_layers

    out: list[CommunicationRound] = []
    for r in rounds:
        if not r.edges:
            out.append(
                CommunicationRound(
                    index=r.index,
                    label=r.label,
                    edges=(),
                    sub_index=0,
                )
            )
            continue
        layers = partition_edges_into_matching_layers(r.edges)
        for j, layer_edges in enumerate(layers):
            out.append(
                CommunicationRound(
                    index=r.index,
                    label=r.label,
                    edges=tuple(sorted(layer_edges)),
                    sub_index=j,
                )
            )
    return tuple(out)
