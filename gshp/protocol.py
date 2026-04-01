"""Formal experiment protocol snapshot + stable hash (reproducibility / instrumentation)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from gshp.graph.caveman import CavemanTopology

PROTOCOL_VERSION = "1.1.0"


def canonical_protocol_dict(
    topo: CavemanTopology,
    *,
    schedule: str,
    dyad_turns: int,
    information_condition: str,
    tom_bridge: bool,
    task_id: str,
    parallel_dyad_layers: bool = False,
) -> dict[str, Any]:
    """JSON-serializable, order-stable description of the communication *instrument*."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task_id": task_id,
        "topology_kind": topo.kind,
        "l": topo.l,
        "k": topo.k,
        "intra_edges": [list(e) for e in sorted(topo.intra_edges)],
        "inter_edges": [list(e) for e in sorted(topo.inter_edges)],
        "connector_nodes": sorted(topo.connector_nodes),
        "schedule_name": schedule,
        "dyad_turns_per_conversation": dyad_turns,
        "information_condition": information_condition,
        "tom_bridge_prompt_on_designated_agents": tom_bridge,
        "parallel_dyad_layers": bool(parallel_dyad_layers),
    }


def protocol_sha256(canonical: dict[str, Any]) -> str:
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
