"""Parallel dyad layers = graph matchings."""

from gshp.graph.caveman import CavemanTopology
from gshp.matching_schedule import partition_edges_into_matching_layers
from gshp.schedule import (
    ScheduleName,
    build_two_phase_schedule,
    expand_schedule_parallel_matchings,
)


def _assert_layer_is_matching(layer: tuple[tuple[int, int], ...]) -> None:
    seen: set[int] = set()
    for u, v in layer:
        assert u not in seen and v not in seen, layer
        seen.add(u)
        seen.add(v)


def test_partition_triangle_three_layers():
    layers = partition_edges_into_matching_layers([(0, 1), (1, 2), (0, 2)])
    assert len(layers) == 3
    for L in layers:
        assert len(L) == 1
        _assert_layer_is_matching(L)


def test_partition_path_two_layers():
    layers = partition_edges_into_matching_layers([(0, 1), (1, 2)])
    assert len(layers) == 2
    for L in layers:
        _assert_layer_is_matching(L)
    flat = [e for L in layers for e in L]
    assert sorted(flat) == [(0, 1), (1, 2)]


def test_expand_preserves_edge_multiset():
    topo = CavemanTopology.build(3, 3, kind="full_clique_ring")
    base = build_two_phase_schedule(topo, ScheduleName.WITHIN_FIRST)
    exp = expand_schedule_parallel_matchings(base)
    assert len(exp) >= len(base)

    def multiset(rounds):
        out = []
        for r in rounds:
            out.extend(r.edges)
        return sorted(out)

    assert multiset(base) == multiset(exp)

    for r in exp:
        _assert_layer_is_matching(r.edges)
