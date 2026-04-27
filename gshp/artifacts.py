"""
Write a results folder in the same spirit as typical lab experiment runs:

- ``config.json`` — run parameters + manifest
- ``task.json`` — full task spec (reproducibility)
- ``summary.json`` — outcomes / accuracy
- ``llm_calls.json`` — every LLM request/response (like consolidated prompt + API logs)
- ``game_log.txt`` — short human-readable trace
- ``dyad_NNN.json`` — one file per dyadic conversation
- ``run.json`` — full serialized :class:`ExperimentRun` (transcripts + decisions)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from gshp.audit import audit_metadata
from gshp.dv3 import convergence_alignment_by_cluster, convergence_alignment_metrics
from gshp.fact_tracker import analyze_fact_transmission, fact_transmission_summary
from gshp.graph.caveman import CavemanTopology
from gshp.info_gain import analyze_information_gain
from gshp.metrics import aggregate_llm_call_stats, fact_mention_rates
from gshp.protocol import canonical_protocol_dict, protocol_sha256
from gshp.task.hiring import HiringTaskSpec, InformationCondition
from gshp.types import ExperimentRun


def write_run_bundle(
    results_dir: str | Path,
    *,
    run: ExperimentRun,
    task: HiringTaskSpec,
    topo: CavemanTopology,
    llm_calls: list[dict[str, Any]],
    dyad_turns: int,
    tom_bridge: bool,
    extra_config: dict[str, Any] | None = None,
    bridge_codings: dict[str, Any] | None = None,
) -> Path:
    """
    Create ``results_dir`` and write all artifact files. Returns the resolved path.
    """
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)

    proto = canonical_protocol_dict(
        topo,
        schedule=run.manifest.schedule,
        dyad_turns=dyad_turns,
        information_condition=run.manifest.information_condition,
        tom_bridge=tom_bridge,
        task_id=task.task_id,
        parallel_dyad_layers=run.manifest.parallel_dyad_layers,
    )
    proto_hash = protocol_sha256(proto)
    audit = audit_metadata()
    llm_stats = aggregate_llm_call_stats(llm_calls)
    fact_stats = fact_mention_rates(dict(task.facts), llm_calls)

    config: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "artifact_policy": "full_capture",
        "writes_llm_calls_json": True,
        "embeds_raw_openai_completion_when_available": True,
        "protocol": proto,
        "protocol_sha256": proto_hash,
        "audit": audit,
        "topology": {
            "kind": topo.kind,
            "l": topo.l,
            "k": topo.k,
            "intra_edge_count": len(topo.intra_edges),
            "inter_edge_count": len(topo.inter_edges),
            "connector_nodes": sorted(topo.connector_nodes),
        },
        "manifest": run.manifest.model_dump(),
        "task_id": task.task_id,
        "correct_candidate": task.correct_candidate,
        "attractor_candidate": task.attractor_candidate,
        "num_llm_calls": len(llm_calls),
        "llm_aggregate": llm_stats,
        **(extra_config or {}),
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (root / "task.json").write_text(
        task.model_dump_json(indent=2),
        encoding="utf-8",
    )
    summary: dict[str, Any] = {
        "timestamp": config["timestamp"],
        "protocol_sha256": proto_hash,
        "notes": dict(run.notes),
        "manifest": run.manifest.model_dump(),
        "votes": [d.model_dump() for d in run.final_decisions],
        "llm_aggregate": llm_stats,
        "fact_mention_heuristic": {
            "facts_checked": fact_stats["facts_checked"],
            "facts_mentioned_anywhere": fact_stats["facts_mentioned_anywhere"],
        },
    }
    if run.deliberation is not None:
        summary["deliberation"] = {
            "group_consensus": run.deliberation.group_consensus,
            "unanimous": run.deliberation.unanimous,
            "group_votes": [d.model_dump() for d in run.deliberation.group_decisions],
        }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "protocol_sha256": proto_hash,
                "llm_aggregate": llm_stats,
                "fact_mentions": fact_stats,
                "notes": dict(run.notes),
                "manifest": run.manifest.model_dump(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "llm_calls.json").write_text(
        json.dumps({"calls": llm_calls}, indent=2),
        encoding="utf-8",
    )
    (root / "run.json").write_text(
        json.dumps(run.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    for i, dyad in enumerate(run.dyads):
        p = root / f"dyad_{i:03d}_{dyad.round_label}.json"
        p.write_text(json.dumps(dyad.model_dump(mode="json"), indent=2), encoding="utf-8")

    # DV2: fact transmission analysis
    condition = InformationCondition(run.manifest.information_condition)
    ft_records = analyze_fact_transmission(run, task, topo.l, topo.k, condition)
    ft_summary = fact_transmission_summary(ft_records)
    (root / "fact_transmission.json").write_text(
        json.dumps(ft_summary, indent=2),
        encoding="utf-8",
    )

    # Group deliberation
    if run.deliberation is not None:
        (root / "deliberation.json").write_text(
            json.dumps(run.deliberation.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

    # DV3: convergence/alignment dissociation metrics
    dv3_overall = convergence_alignment_metrics(
        run,
        correct_choice=task.correct_candidate,
    )
    dv3_clusters = convergence_alignment_by_cluster(
        run.final_decisions,
        topo_l=topo.l,
        topo_k=topo.k,
        correct_choice=task.correct_candidate,
    )
    (root / "dv3.json").write_text(
        json.dumps(
            {
                "overall": dv3_overall,
                "by_cluster": dv3_clusters,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Entropy-reduction information-gain analysis + Shapley attribution
    info_gain = analyze_information_gain(run, task)
    (root / "info_gain.json").write_text(
        json.dumps(info_gain, indent=2),
        encoding="utf-8",
    )

    # DV4: bridge coding (optional, provided externally)
    if bridge_codings is not None:
        (root / "bridge_coding.json").write_text(
            json.dumps(bridge_codings, indent=2),
            encoding="utf-8",
        )

    _write_game_log(root / "game_log.txt", run, task)
    return root


def _write_game_log(path: Path, run: ExperimentRun, task: HiringTaskSpec) -> None:
    lines: list[str] = []
    lines.append(f"# graph-scheduled-hidden-profile run log")
    lines.append(f"# task={task.task_id} correct={task.correct_candidate}")
    lines.append(f"# schedule={run.manifest.schedule} condition={run.manifest.information_condition}")
    lines.append("")
    for i, d in enumerate(run.dyads):
        sub = d.round_sub_index
        sub_s = f" sub={sub}" if sub else ""
        lines.append(
            f"dyad {i:03d} round={d.round_index} label={d.round_label}{sub_s} "
            f"agents={d.u}-{d.v} turns={len(d.messages)}"
        )
    lines.append("")
    lines.append("--- final decisions ---")
    for dec in run.final_decisions:
        lines.append(
            f"agent {dec.agent_id}: choice={dec.choice} | {dec.justification[:120]}"
        )
    lines.append("")
    lines.append("--- aggregate ---")
    for k, v in run.notes.items():
        lines.append(f"{k}={v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
