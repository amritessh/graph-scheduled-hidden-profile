"""
Batch-level aggregation for factorial experiment outputs.

Consumes a batch root (folder containing index.csv + per-run artifacts) and writes:
  - batch_analysis.json (full machine-readable aggregates)
  - condition_summary.csv (one row per cell)
  - report.md (human-readable summary for quick review)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def analyze_batch_dir(batch_dir: str | Path) -> dict[str, Any]:
    root = Path(batch_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    index_path = root / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index.csv in {root}")

    rows = list(_read_index(index_path))
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        return {
            "batch_dir": str(root.resolve()),
            "n_index_rows": len(rows),
            "n_ok_runs": 0,
            "cells": {},
        }

    per_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok_rows:
        per_cell[r["cell_id"]].append(r)

    cell_metrics: dict[str, Any] = {}
    for cell_id, runs in per_cell.items():
        cell_metrics[cell_id] = _aggregate_cell(cell_id, runs)

    overall = _aggregate_overall(list(cell_metrics.values()))
    return {
        "batch_dir": str(root.resolve()),
        "n_index_rows": len(rows),
        "n_ok_runs": len(ok_rows),
        "overall": overall,
        "cells": cell_metrics,
    }


def write_batch_analysis(
    batch_dir: str | Path,
    *,
    out_json: str | Path | None = None,
    out_csv: str | Path | None = None,
    out_report: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(batch_dir)
    data = analyze_batch_dir(root)

    json_path = Path(out_json) if out_json else (root / "batch_analysis.json")
    csv_path = Path(out_csv) if out_csv else (root / "condition_summary.csv")
    report_path = Path(out_report) if out_report else (root / "report.md")

    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_condition_csv(csv_path, data)
    report_path.write_text(_render_report_md(data), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "report": report_path}


def _read_index(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _aggregate_cell(cell_id: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    task: dict[str, Any] = {
        "cell_id": cell_id,
        "n_runs": len(runs),
        "factors": {
            "schedule": runs[0].get("grid_schedule"),
            "condition": runs[0].get("grid_condition"),
            "tom_bridge": _to_bool(runs[0].get("grid_tom_bridge")),
        },
    }

    acc = []
    bits_total = []
    zero_rate = []
    entropy_final = []
    resolved = []
    bridge_bits = []
    non_bridge_bits = []
    unique_eff = []
    shared_eff = []
    synergy = []
    redundancy = []

    for r in runs:
        run_dir = Path(r["run_dir"])
        summary = _safe_json(run_dir / "summary.json")
        info = _safe_json(run_dir / "info_gain.json")
        dv3 = _safe_json(run_dir / "dv3.json")

        notes = (summary or {}).get("notes", {})
        if isinstance(notes.get("accuracy_agent_level"), (int, float)):
            acc.append(float(notes["accuracy_agent_level"]))

        if info:
            if isinstance(info.get("total_bits_resolved"), (int, float)):
                bits_total.append(float(info["total_bits_resolved"]))
            if isinstance(info.get("zero_bit_rate"), (int, float)):
                zero_rate.append(float(info["zero_bit_rate"]))
            if isinstance(info.get("final_entropy"), (int, float)):
                entropy_final.append(float(info["final_entropy"]))
            if isinstance(info.get("resolved_to_single_option"), bool):
                resolved.append(1.0 if info["resolved_to_single_option"] else 0.0)

            b_by_agent = info.get("bits_by_agent") or {}
            bridge = sum(
                float(v) for k, v in b_by_agent.items() if str(k) in {"2", "5", "8"}
            )
            non_bridge = sum(
                float(v) for k, v in b_by_agent.items() if str(k) not in {"2", "5", "8"}
            )
            bridge_bits.append(bridge)
            non_bridge_bits.append(non_bridge)

            avg_by_cat = info.get("avg_bits_per_event_by_category") or {}
            if isinstance(avg_by_cat.get("cluster"), (int, float)) or isinstance(avg_by_cat.get("bridge"), (int, float)):
                unique_eff.append(float(avg_by_cat.get("cluster", 0.0)) + float(avg_by_cat.get("bridge", 0.0)))
            if isinstance(avg_by_cat.get("shared"), (int, float)):
                shared_eff.append(float(avg_by_cat.get("shared")))

            shap = info.get("shapley") or {}
            if isinstance(shap.get("total_positive_pairwise_synergy_bits"), (int, float)):
                synergy.append(float(shap.get("total_positive_pairwise_synergy_bits")))
            if isinstance(shap.get("total_negative_pairwise_redundancy_bits"), (int, float)):
                redundancy.append(float(shap.get("total_negative_pairwise_redundancy_bits")))

        # Keep as optional metric in case needed downstream
        _ = dv3

    task["metrics"] = {
        "accuracy_agent_level_mean": _mean(acc),
        "total_bits_resolved_mean": _mean(bits_total),
        "zero_bit_rate_mean": _mean(zero_rate),
        "final_entropy_mean": _mean(entropy_final),
        "resolved_to_single_option_rate": _mean(resolved),
        "bridge_bits_mean": _mean(bridge_bits),
        "non_bridge_bits_mean": _mean(non_bridge_bits),
        "unique_bits_per_event_mean": _mean(unique_eff),
        "shared_bits_per_event_mean": _mean(shared_eff),
        "synergy_bits_mean": _mean(synergy),
        "redundancy_bits_mean": _mean(redundancy),
    }
    return task


def _aggregate_overall(cells: list[dict[str, Any]]) -> dict[str, Any]:
    if not cells:
        return {}
    m = [c["metrics"] for c in cells]
    return {
        "n_cells": len(cells),
        "accuracy_agent_level_mean": _mean([x.get("accuracy_agent_level_mean") for x in m]),
        "total_bits_resolved_mean": _mean([x.get("total_bits_resolved_mean") for x in m]),
        "zero_bit_rate_mean": _mean([x.get("zero_bit_rate_mean") for x in m]),
        "resolved_to_single_option_rate_mean": _mean([x.get("resolved_to_single_option_rate") for x in m]),
        "synergy_bits_mean": _mean([x.get("synergy_bits_mean") for x in m]),
    }


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _mean(vals: list[Any]) -> float | None:
    clean = [float(v) for v in vals if isinstance(v, (int, float))]
    if not clean:
        return None
    return mean(clean)


def _to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes"}:
            return True
        if s in {"0", "false", "no"}:
            return False
    return None


def _write_condition_csv(path: Path, data: dict[str, Any]) -> None:
    rows = []
    for cell in data.get("cells", {}).values():
        row = {
            "cell_id": cell.get("cell_id"),
            "n_runs": cell.get("n_runs"),
            "schedule": (cell.get("factors") or {}).get("schedule"),
            "condition": (cell.get("factors") or {}).get("condition"),
            "tom_bridge": (cell.get("factors") or {}).get("tom_bridge"),
            **(cell.get("metrics") or {}),
        }
        rows.append(row)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _render_report_md(data: dict[str, Any]) -> str:
    overall = data.get("overall", {})
    lines = [
        "# Batch Analysis Report",
        "",
        f"- Batch: `{data.get('batch_dir', '')}`",
        f"- Runs analyzed: `{data.get('n_ok_runs', 0)}` / index rows `{data.get('n_index_rows', 0)}`",
        "",
        "## Overall",
        "",
        f"- Mean agent-level accuracy: `{_fmt(overall.get('accuracy_agent_level_mean'))}`",
        f"- Mean total bits resolved: `{_fmt(overall.get('total_bits_resolved_mean'))}`",
        f"- Mean zero-bit rate: `{_fmt(overall.get('zero_bit_rate_mean'))}`",
        f"- Mean resolved-to-single-option rate: `{_fmt(overall.get('resolved_to_single_option_rate_mean'))}`",
        f"- Mean synergy bits: `{_fmt(overall.get('synergy_bits_mean'))}`",
        "",
        "## Per Condition (Topline)",
        "",
    ]

    cells = sorted(
        data.get("cells", {}).values(),
        key=lambda c: (
            (c.get("factors") or {}).get("schedule") or "",
            (c.get("factors") or {}).get("condition") or "",
            str((c.get("factors") or {}).get("tom_bridge")),
        ),
    )
    for c in cells:
        f = c.get("factors") or {}
        m = c.get("metrics") or {}
        lines.append(
            "- "
            f"`{f.get('schedule')}` / `{f.get('condition')}` / `tom={f.get('tom_bridge')}`: "
            f"acc={_fmt(m.get('accuracy_agent_level_mean'))}, "
            f"bits={_fmt(m.get('total_bits_resolved_mean'))}, "
            f"waste={_fmt(m.get('zero_bit_rate_mean'))}, "
            f"synergy={_fmt(m.get('synergy_bits_mean'))}"
        )
    lines.append("")
    lines.append("Detailed machine-readable outputs are in `batch_analysis.json` and `condition_summary.csv`.")
    return "\n".join(lines) + "\n"


def _fmt(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.4f}"
    return "NA"
