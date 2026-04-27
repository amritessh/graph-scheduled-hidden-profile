"""Partition communication edges into parallel dyad layers (graph matchings)."""

from __future__ import annotations

from collections import Counter

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
    remaining: Counter[tuple[int, int]] = Counter(_normalize_edge(e) for e in edges)
    layers: list[tuple[tuple[int, int], ...]] = []
    while any(c > 0 for c in remaining.values()):
        g = nx.Graph()
        g.add_edges_from([e for e, c in remaining.items() if c > 0])
        # Unweighted maximum-cardinality matching (general graphs).
        raw = nx.max_weight_matching(g, maxcardinality=True)
        layer_set = {_normalize_edge((int(u), int(v))) for u, v in raw}
        if not layer_set:
            raise RuntimeError(
                "max_weight_matching returned empty while edges remain; "
                f"remaining={sorted([e for e, c in remaining.items() if c > 0])[:20]!s}..."
            )
        layers.append(tuple(sorted(layer_set)))
        for e in layer_set:
            remaining[e] -= 1
            if remaining[e] <= 0:
                del remaining[e]
    return layers
