"""
DV2 — Unique Information Utilization.

For each fact in the task, track three separable transmission failures:

  disclosure  — was the fact mentioned in *any* conversation?
  transmission — did it appear in at least one *inter-cluster* dyad (crossed a cluster boundary)?
  integration  — did it appear in any agent's final decision justification?

Only the conjunction of all three means a fact actually influenced the outcome.
Each stage can fail independently, which tells you exactly where the mechanism breaks.

Usage::

    records = analyze_fact_transmission(run, task, topo, condition)
    summary = fact_transmission_summary(records)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gshp.aho_corasick import AhoCorasickAutomaton
from gshp.task.hiring import HiringTaskSpec, InformationCondition, cluster_index_for_agent
from gshp.types import ExperimentRun


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FactRecord:
    """Per-fact transmission analysis."""

    fact_id: str
    fact_text: str
    category: str  # "shared" | "cluster" | "bridge"
    owner_agents: list[int]  # agents who initially held this fact
    owner_clusters: list[int]  # cluster indices of owners (may be all clusters for shared)

    # DV2 measures (populated by analyze_fact_transmission)
    disclosed: bool = False  # mentioned in any conversation message
    transmitted: bool = False  # mentioned in at least one inter-cluster dyad
    integrated_by: list[int] = field(default_factory=list)  # agent_ids in final justifications

    # Supporting detail
    dyads_disclosed_in: list[tuple[int, int]] = field(default_factory=list)
    dyads_transmitted_in: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "category": self.category,
            "owner_agents": self.owner_agents,
            "owner_clusters": self.owner_clusters,
            "disclosed": self.disclosed,
            "transmitted": self.transmitted,
            "integrated_by": self.integrated_by,
            "dyads_disclosed_in": [list(d) for d in self.dyads_disclosed_in],
            "dyads_transmitted_in": [list(d) for d in self.dyads_transmitted_in],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fact_owners(
    task: HiringTaskSpec,
    topo_l: int,
    topo_k: int,
) -> dict[str, tuple[str, list[int], list[int]]]:
    """
    Returns {fact_id: (category, owner_agent_ids, owner_cluster_ids)}.
    """
    n = topo_l * topo_k
    result: dict[str, tuple[str, list[int], list[int]]] = {}

    for fid in task.shared_fact_ids:
        agents = list(range(n))
        clusters = list(range(topo_l))
        result[fid] = ("shared", agents, clusters)

    for ci, fids in task.cluster_fact_ids.items():
        agents = [a for a in range(n) if cluster_index_for_agent(a, topo_k) == ci]
        for fid in fids:
            result[fid] = ("cluster", agents, [ci])

    for aid, fid in task.bridge_agent_fact_ids.items():
        ci = cluster_index_for_agent(int(aid), topo_k)
        result[fid] = ("bridge", [int(aid)], [ci])

    return result


def _build_ac(patterns: list[str]) -> AhoCorasickAutomaton | None:
    """Build an Aho-Corasick automaton from non-empty lowercased patterns."""
    cleaned = [p.strip().lower() for p in patterns if p.strip()]
    if not cleaned:
        return None
    return AhoCorasickAutomaton(cleaned)


def _mentions(ac: AhoCorasickAutomaton, text: str) -> set[int]:
    """Return the set of pattern indices that appear in text (lowercased)."""
    return set(ac.matching_pattern_indices(text.lower()))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_fact_transmission(
    run: ExperimentRun,
    task: HiringTaskSpec,
    topo_l: int,
    topo_k: int,
    condition: InformationCondition = InformationCondition.HIDDEN_PROFILE,
) -> dict[str, FactRecord]:
    """
    Analyse fact transmission across all dyads and final decisions.

    In SHARED_ONLY condition, unique facts were never distributed, so transmission
    and integration metrics for cluster/bridge facts will always be False.
    """
    owners = _fact_owners(task, topo_l, topo_k)

    # Build ordered lists so AC pattern indices map to fact_ids
    all_fids: list[str] = sorted(task.facts.keys())
    patterns: list[str] = [task.facts[fid] for fid in all_fids]
    ac = _build_ac(patterns)

    records: dict[str, FactRecord] = {}
    for fid in all_fids:
        cat, ags, cls = owners.get(fid, ("shared", [], []))
        records[fid] = FactRecord(
            fact_id=fid,
            fact_text=task.facts[fid],
            category=cat,
            owner_agents=ags,
            owner_clusters=cls,
        )

    if ac is None:
        return records

    # -- Dyad-level analysis --------------------------------------------------
    for dyad in run.dyads:
        u, v = dyad.u, dyad.v
        cu = cluster_index_for_agent(u, topo_k)
        cv = cluster_index_for_agent(v, topo_k)
        is_inter = cu != cv  # True for cross-cluster dyads

        # Combine all messages in this dyad
        dyad_text = "\n".join(m.content for m in dyad.messages)
        if not dyad_text.strip():
            continue

        hit_indices = _mentions(ac, dyad_text)
        for hi in hit_indices:
            fid = all_fids[hi]
            rec = records[fid]
            edge = (u, v)
            if not rec.disclosed:
                rec.disclosed = True
            rec.dyads_disclosed_in.append(edge)
            if is_inter:
                if not rec.transmitted:
                    rec.transmitted = True
                rec.dyads_transmitted_in.append(edge)

    # -- Integration: final decision justifications ---------------------------
    for decision in run.final_decisions:
        text = decision.justification
        if not text.strip():
            continue
        hit_indices = _mentions(ac, text)
        for hi in hit_indices:
            fid = all_fids[hi]
            records[fid].integrated_by.append(decision.agent_id)

    # Also check deliberation justifications if present
    if run.deliberation:
        for gdec in run.deliberation.group_decisions:
            text = gdec.justification
            if not text.strip():
                continue
            hit_indices = _mentions(ac, text)
            for hi in hit_indices:
                fid = all_fids[hi]
                aid = gdec.agent_id
                if aid not in records[fid].integrated_by:
                    records[fid].integrated_by.append(aid)

    return records


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


def fact_transmission_summary(
    records: dict[str, FactRecord],
) -> dict[str, Any]:
    """
    Aggregate DV2 rates across fact categories.

    Returns counts and rates separately for shared, cluster, and bridge facts.
    """
    cats = ["shared", "cluster", "bridge"]
    out: dict[str, Any] = {"per_fact": {fid: r.to_dict() for fid, r in records.items()}}

    for cat in cats:
        subset = [r for r in records.values() if r.category == cat]
        n = len(subset)
        if n == 0:
            out[cat] = {"n": 0}
            continue
        out[cat] = {
            "n": n,
            "disclosed": sum(r.disclosed for r in subset),
            "disclosed_rate": sum(r.disclosed for r in subset) / n,
            "transmitted": sum(r.transmitted for r in subset),
            "transmitted_rate": sum(r.transmitted for r in subset) / n,
            "integrated": sum(bool(r.integrated_by) for r in subset),
            "integrated_rate": sum(bool(r.integrated_by) for r in subset) / n,
        }

    # Overall (across all facts)
    all_recs = list(records.values())
    n_all = len(all_recs)
    out["overall"] = {
        "n": n_all,
        "disclosed_rate": sum(r.disclosed for r in all_recs) / n_all if n_all else 0,
        "transmitted_rate": sum(r.transmitted for r in all_recs) / n_all if n_all else 0,
        "integrated_rate": sum(bool(r.integrated_by) for r in all_recs) / n_all if n_all else 0,
    }

    return out
