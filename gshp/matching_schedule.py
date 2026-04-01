"""Partition communication edges into parallel dyad layers (graph matchings)."""

from __future__ import annotations

import networkx as nx


def _normalize_edge(e: tuple[int, int]) -> tuple[int, int]:
    u, v = e
    return (u, v) if u < v else (v, u)


def partition_edges_into_matching_layers(
    edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> list[tuple[tuple[int, int], ...]]:
    """
    Split ``edges`` into an ordered list of **layers**.

    Each layer is a **matching**: no vertex appears in more than one edge in that layer.
    Layers are built **greedily but optimally per step**: repeatedly take a *maximum*
    cardinality matching on the remaining subgraph, remove those edges, repeat until none
    are left.

    This yields a valid parallel schedule (each layer could run simultaneously). Layer count
    is modest on our caveman topologies (bounded by the edge count; often close to max degree).

    Empty input returns an empty list.
    """
    remaining: set[tuple[int, int]] = {_normalize_edge(e) for e in edges}
    layers: list[tuple[tuple[int, int], ...]] = []
    while remaining:
        g = nx.Graph()
        g.add_edges_from(remaining)
        # Unweighted maximum-cardinality matching (general graphs).
        raw = nx.max_weight_matching(g, maxcardinality=True)
        layer_set = {_normalize_edge((int(u), int(v))) for u, v in raw}
        if not layer_set:
            raise RuntimeError(
                "max_weight_matching returned empty while edges remain; "
                f"remaining={sorted(remaining)[:20]!s}..."
            )
        layers.append(tuple(sorted(layer_set)))
        remaining -= layer_set
    return layers
