"""Community graphs for experiments: caveman-style layouts + intra/inter edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Literal

import networkx as nx

TopologyKind = Literal[
    "full_clique_ring",
    "networkx_caveman",
    "networkx_connected_caveman",
]


def graph_full_clique_ring(l: int, k: int) -> nx.Graph:
    """
    `l` disjoint cliques K_k, plus one **bridge** edge between consecutive communities
    in a ring (last node of clique i ↔ first node of clique i+1, with wrap).

    This matches the usual “cavemen + between-cave ties” *picture* for experiments:
    dense within-community ties, sparse cross-community ties.

    Note: `networkx.caveman_graph(l, k)` (as of NX 3.6) is **l isolated cliques**
    with **no** inter-clique edges, so it is a poor default for cross-community
    scheduling. Use this builder unless you explicitly want disconnected cliques.
    """
    if l < 1 or k < 2:
        raise ValueError("Need l >= 1 and k >= 2")
    G = nx.Graph()
    n = l * k
    G.add_nodes_from(range(n))
    for i in range(l):
        nodes = list(range(i * k, (i + 1) * k))
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                G.add_edge(nodes[a], nodes[b])
    for i in range(l):
        last_i = (i + 1) * k - 1
        first_next = ((i + 1) % l) * k
        G.add_edge(last_i, first_next)
    return G


@dataclass(frozen=True)
class CavemanTopology:
    """Community layout: partition, intra/inter edges, connector (bridge) nodes."""

    graph: nx.Graph
    l: int
    k: int
    kind: TopologyKind
    communities: tuple[FrozenSet[int], ...]
    intra_edges: frozenset[tuple[int, int]]
    inter_edges: frozenset[tuple[int, int]]
    """Nodes that touch at least one inter-community edge (bridge endpoints)."""
    connector_nodes: frozenset[int]

    @staticmethod
    def build(
        l: int = 3,
        k: int = 3,
        *,
        kind: TopologyKind = "full_clique_ring",
    ) -> "CavemanTopology":
        if l < 1 or k < 2:
            raise ValueError("Need l >= 1 and k >= 2")
        if kind == "full_clique_ring":
            G = graph_full_clique_ring(l, k)
        elif kind == "networkx_caveman":
            G = nx.caveman_graph(l, k)
        elif kind == "networkx_connected_caveman":
            G = nx.connected_caveman_graph(l, k)
        else:
            raise ValueError(kind)
        communities: list[FrozenSet[int]] = []
        for i in range(l):
            nodes = frozenset(range(i * k, (i + 1) * k))
            communities.append(nodes)

        comm_of: dict[int, int] = {}
        for i, c in enumerate(communities):
            for n in c:
                comm_of[n] = i

        intra: set[tuple[int, int]] = set()
        inter: set[tuple[int, int]] = set()
        for u, v in G.edges():
            a, b = (u, v) if u < v else (v, u)
            if comm_of[u] == comm_of[v]:
                intra.add((a, b))
            else:
                inter.add((a, b))

        connectors: set[int] = set()
        for a, b in inter:
            connectors.add(a)
            connectors.add(b)

        return CavemanTopology(
            graph=G,
            l=l,
            k=k,
            kind=kind,
            communities=tuple(communities),
            intra_edges=frozenset(intra),
            inter_edges=frozenset(inter),
            connector_nodes=frozenset(connectors),
        )

    @property
    def nodes(self) -> list[int]:
        return sorted(self.graph.nodes())

    def edges_for_round_kind(self, kind: str) -> list[tuple[int, int]]:
        """kind: 'intra' | 'inter' — undirected edges as sorted pairs."""
        if kind == "intra":
            return sorted(self.intra_edges)
        if kind == "inter":
            return sorted(self.inter_edges)
        raise ValueError("kind must be 'intra' or 'inter'")
