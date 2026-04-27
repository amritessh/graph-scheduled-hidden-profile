#!/usr/bin/env python3
"""
Paper-focused interpretation tables from a completed batch folder.

Outputs:
  - paper_tables.json
  - paper_report.md

Usage:
  python scripts/paper_tables.py results/qwen3_full_factorial/<batch_dir>
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class CellKey:
    schedule: str
    condition: str
    tom_bridge: bool

    def slug(self) -> str:
        return f"{self.schedule}|{self.condition}|tom={int(self.tom_bridge)}"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/paper_tables.py <batch_dir>")
    root = Path(sys.argv[1])
    if not root.is_dir():
        raise SystemExit(f"Batch dir not found: {root}")

    progress_path = root / "progress.json"
    if not progress_path.exists():
        raise SystemExit(f"Missing progress.json in {root}")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))

    run_dirs = [Path(x["run_dir"]) for x in progress.get("completed", []) if "run_dir" in x]
    if not run_dirs:
        raise SystemExit("No completed runs found in progress.json")

    by_cell: dict[CellKey, list[dict[str, float]]] = {}
    for run_dir in run_dirs:
        row = _load_run_metrics(run_dir)
        if row is None:
            continue
        key = CellKey(row["schedule"], row["condition"], bool(row["tom_bridge"]))
        by_cell.setdefault(key, []).append(row)

    cell_summary = {k.slug(): _summarize_runs(v) for k, v in by_cell.items()}
    contrasts = _build_contrasts(by_cell)
    report_md = _render_report(root, cell_summary, contrasts)

    out_json = root / "paper_tables.json"
    out_md = root / "paper_report.md"
    out_json.write_text(
        json.dumps(
            {
                "n_completed_runs": len(run_dirs),
                "n_cells": len(by_cell),
                "cell_summary": cell_summary,
                "contrasts": contrasts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out_md.write_text(report_md, encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


def _load_run_metrics(run_dir: Path) -> dict[str, Any] | None:
    summary_p = run_dir / "summary.json"
    info_p = run_dir / "info_gain.json"
    bridge_p = run_dir / "bridge_coding.json"
    if not summary_p.exists() or not info_p.exists():
        return None
    summary = json.loads(summary_p.read_text(encoding="utf-8"))
    info = json.loads(info_p.read_text(encoding="utf-8"))
    bridge = json.loads(bridge_p.read_text(encoding="utf-8")) if bridge_p.exists() else {}

    manifest = summary.get("manifest", {})
    notes = summary.get("notes", {})
    shap = info.get("shapley", {}) or {}

    return {
        "schedule": str(manifest.get("schedule")),
        "condition": str(manifest.get("information_condition")),
        "tom_bridge": bool(manifest.get("tom_bridge")),
        "accuracy": float(notes.get("accuracy_agent_level", 0.0)),
        "bits_total": float(info.get("total_bits_resolved", 0.0)),
        "zero_rate": float(info.get("zero_bit_rate", 0.0)),
        "synergy_bits": float(shap.get("total_positive_pairwise_synergy_bits", 0.0)),
        "redundancy_bits": float(shap.get("total_negative_pairwise_redundancy_bits", 0.0)),
        "translate_rate": float(bridge.get("translate_rate", 0.0)),
    }


def _summarize_runs(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ["accuracy", "bits_total", "zero_rate", "synergy_bits", "redundancy_bits", "translate_rate"]
    out: dict[str, float] = {"n": float(len(rows))}
    for k in keys:
        vals = [float(r[k]) for r in rows if isinstance(r.get(k), (int, float))]
        out[f"{k}_mean"] = mean(vals) if vals else 0.0
    return out


def _build_contrasts(by_cell: dict[CellKey, list[dict[str, float]]]) -> dict[str, Any]:
    # Primary hypotheses
    out: dict[str, Any] = {}

    # H1: timing effect under hidden profile (No ToM)
    out["timing_hidden_no_tom"] = _delta_with_ci(
        by_cell,
        CellKey("cross_first", "hidden_profile", False),
        CellKey("within_first", "hidden_profile", False),
        metric="accuracy",
    )

    # H2: ToM effect under bridge-first hidden profile
    out["tom_given_bridge_first_hidden"] = _delta_with_ci(
        by_cell,
        CellKey("cross_first", "hidden_profile", True),
        CellKey("cross_first", "hidden_profile", False),
        metric="accuracy",
    )

    # H3: hidden-profile necessity in bridge-first + ToM
    out["hidden_vs_shared_bridge_first_tom"] = _delta_with_ci(
        by_cell,
        CellKey("cross_first", "hidden_profile", True),
        CellKey("cross_first", "shared_only", True),
        metric="accuracy",
    )

    # Mechanism-focused deltas
    out["timing_hidden_no_tom_bits"] = _delta_with_ci(
        by_cell,
        CellKey("cross_first", "hidden_profile", False),
        CellKey("within_first", "hidden_profile", False),
        metric="bits_total",
    )
    out["tom_bridge_first_hidden_translate"] = _delta_with_ci(
        by_cell,
        CellKey("cross_first", "hidden_profile", True),
        CellKey("cross_first", "hidden_profile", False),
        metric="translate_rate",
    )
    out["synergy_peak_cell_vs_baseline"] = _delta_with_ci(
        by_cell,
        CellKey("cross_first", "hidden_profile", True),
        CellKey("within_first", "shared_only", False),
        metric="synergy_bits",
    )
    return out


def _delta_with_ci(
    by_cell: dict[CellKey, list[dict[str, float]]],
    a: CellKey,
    b: CellKey,
    *,
    metric: str,
    n_boot: int = 2000,
) -> dict[str, Any]:
    ra = by_cell.get(a, [])
    rb = by_cell.get(b, [])
    if not ra or not rb:
        return {"missing": True, "a": a.slug(), "b": b.slug(), "metric": metric}

    va = [float(r[metric]) for r in ra]
    vb = [float(r[metric]) for r in rb]
    obs = mean(va) - mean(vb)

    rng = random.Random(42)
    boots: list[float] = []
    for _ in range(n_boot):
        sa = [va[rng.randrange(len(va))] for _ in range(len(va))]
        sb = [vb[rng.randrange(len(vb))] for _ in range(len(vb))]
        boots.append(mean(sa) - mean(sb))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]

    return {
        "missing": False,
        "a": a.slug(),
        "b": b.slug(),
        "metric": metric,
        "a_mean": mean(va),
        "b_mean": mean(vb),
        "delta": obs,
        "ci95": [lo, hi],
        "n_a": len(va),
        "n_b": len(vb),
    }


def _render_report(root: Path, cell_summary: dict[str, Any], contrasts: dict[str, Any]) -> str:
    lines = [
        "# Paper Results Report",
        "",
        f"- Batch folder: `{root}`",
        f"- Cells found: `{len(cell_summary)}`",
        "",
        "## Key Contrasts (delta with 95% bootstrap CI)",
        "",
    ]
    for name, c in contrasts.items():
        if c.get("missing"):
            lines.append(f"- `{name}`: missing data (`{c.get('a')}` vs `{c.get('b')}`)")
            continue
        lines.append(
            f"- `{name}` [{c['metric']}]: "
            f"{_fmt(c['delta'])} (CI {_fmt(c['ci95'][0])}, {_fmt(c['ci95'][1])}); "
            f"A={_fmt(c['a_mean'])}, B={_fmt(c['b_mean'])}"
        )

    lines += [
        "",
        "## Cell Means",
        "",
    ]
    for slug, s in sorted(cell_summary.items()):
        lines.append(
            f"- `{slug}` (n={int(s['n'])}): "
            f"acc={_fmt(s['accuracy_mean'])}, bits={_fmt(s['bits_total_mean'])}, "
            f"zero={_fmt(s['zero_rate_mean'])}, synergy={_fmt(s['synergy_bits_mean'])}, "
            f"translate={_fmt(s['translate_rate_mean'])}"
        )

    lines += [
        "",
        "Interpretation hint: positive deltas on accuracy/bits/synergy/translate and negative deltas on zero_rate "
        "in theoretically favored contrasts support your causal mechanism.",
        "",
    ]
    return "\n".join(lines)


def _fmt(x: float) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NA"
    return f"{x:.4f}"


if __name__ == "__main__":
    main()

