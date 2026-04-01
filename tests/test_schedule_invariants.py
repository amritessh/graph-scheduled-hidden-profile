"""Invariants: schedules only use the intended edge types per phase."""

from gshp.graph.caveman import CavemanTopology
from gshp.schedule import ScheduleName, build_two_phase_schedule


def _assert_edges_subset(edges: list[tuple[int, int]], allowed: set[tuple[int, int]]) -> None:
    for e in edges:
        assert e in allowed, f"edge {e} not in allowed set"


def test_within_first_round0_only_intra_round1_only_inter():
    topo = CavemanTopology.build(3, 3, kind="full_clique_ring")
    sched = build_two_phase_schedule(topo, ScheduleName.WITHIN_FIRST)
    assert len(sched) == 2
    intra, inter = topo.intra_edges, topo.inter_edges
    _assert_edges_subset(list(sched[0].edges), set(intra))
    _assert_edges_subset(list(sched[1].edges), set(inter))
    assert sched[0].label == "intra_community"
    assert sched[1].label == "inter_community"


def test_cross_first_round0_only_inter_round1_only_intra():
    topo = CavemanTopology.build(3, 3, kind="full_clique_ring")
    sched = build_two_phase_schedule(topo, ScheduleName.CROSS_FIRST)
    intra, inter = topo.intra_edges, topo.inter_edges
    _assert_edges_subset(list(sched[0].edges), set(inter))
    _assert_edges_subset(list(sched[1].edges), set(intra))


def test_connectors_are_endpoints_of_inter_edges():
    topo = CavemanTopology.build(3, 3, kind="full_clique_ring")
    ends: set[int] = set()
    for a, b in topo.inter_edges:
        ends.add(a)
        ends.add(b)
    assert ends == set(topo.connector_nodes)


def test_protocol_hash_stable_for_same_inputs():
    from gshp.protocol import canonical_protocol_dict, protocol_sha256

    topo = CavemanTopology.build(3, 3, kind="full_clique_ring")
    a = canonical_protocol_dict(
        topo,
        schedule="within_first",
        dyad_turns=6,
        information_condition="hidden_profile",
        tom_bridge=False,
        task_id="hiring_v1",
    )
    b = canonical_protocol_dict(
        topo,
        schedule="within_first",
        dyad_turns=6,
        information_condition="hidden_profile",
        tom_bridge=False,
        task_id="hiring_v1",
    )
    assert protocol_sha256(a) == protocol_sha256(b)
