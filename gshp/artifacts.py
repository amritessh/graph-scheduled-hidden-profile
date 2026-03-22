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

from gshp.graph.caveman import CavemanTopology
from gshp.task.hiring import HiringTaskSpec
from gshp.types import ExperimentRun


def write_run_bundle(
    results_dir: str | Path,
    *,
    run: ExperimentRun,
    task: HiringTaskSpec,
    topo: CavemanTopology,
    llm_calls: list[dict[str, Any]],
    extra_config: dict[str, Any] | None = None,
) -> Path:
    """
    Create ``results_dir`` and write all artifact files. Returns the resolved path.
    """
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
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
        **(extra_config or {}),
    }
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (root / "task.json").write_text(
        task.model_dump_json(indent=2),
        encoding="utf-8",
    )
    summary = {
        "timestamp": config["timestamp"],
        "notes": dict(run.notes),
        "manifest": run.manifest.model_dump(),
        "votes": [d.model_dump() for d in run.final_decisions],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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

    _write_game_log(root / "game_log.txt", run, task)
    return root


def _write_game_log(path: Path, run: ExperimentRun, task: HiringTaskSpec) -> None:
    lines: list[str] = []
    lines.append(f"# graph-scheduled-hidden-profile run log")
    lines.append(f"# task={task.task_id} correct={task.correct_candidate}")
    lines.append(f"# schedule={run.manifest.schedule} condition={run.manifest.information_condition}")
    lines.append("")
    for i, d in enumerate(run.dyads):
        lines.append(
            f"dyad {i:03d} round={d.round_index} label={d.round_label} "
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
