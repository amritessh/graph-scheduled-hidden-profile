"""
Offline analysis of one artifact folder: reload JSON, recompute / extend metrics.

Usage: ``python -m gshp.cli analyze PATH`` or import ``analyze_run_dir``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gshp.metrics import aggregate_llm_call_stats, fact_mention_rates


def analyze_run_dir(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)

    task = json.loads((root / "task.json").read_text(encoding="utf-8"))
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    llm_raw = json.loads((root / "llm_calls.json").read_text(encoding="utf-8"))
    calls = llm_raw.get("calls") or []

    facts = task.get("facts") or {}
    if not isinstance(facts, dict):
        facts = {}

    stats = aggregate_llm_call_stats(calls)
    mentions = fact_mention_rates(facts, calls)
    dv3_path = root / "dv3.json"
    info_gain_path = root / "info_gain.json"
    dv3 = {}
    info_gain = {}
    if dv3_path.exists():
        dv3 = json.loads(dv3_path.read_text(encoding="utf-8"))
    if info_gain_path.exists():
        info_gain = json.loads(info_gain_path.read_text(encoding="utf-8"))

    correct = task.get("correct_candidate")
    votes = [d.get("choice") for d in run.get("final_decisions") or [] if d.get("choice")]
    out: dict[str, Any] = {
        "run_dir": str(root.resolve()),
        "correct_candidate": correct,
        "accuracy_agent_level": run.get("notes", {}).get("accuracy_agent_level"),
        "majority_vote": run.get("notes", {}).get("majority_vote"),
        "llm_aggregate": stats,
        "fact_mentions": mentions,
        "dv3": dv3,
        "info_gain": {
            "total_bits_resolved": info_gain.get("total_bits_resolved"),
            "zero_bit_rate": info_gain.get("zero_bit_rate"),
            "resolved_to_single_option": info_gain.get("resolved_to_single_option"),
            "final_feasible_options": info_gain.get("final_feasible_options"),
            "total_positive_pairwise_synergy_bits": (
                (info_gain.get("shapley") or {}).get("total_positive_pairwise_synergy_bits")
            ),
            "total_negative_pairwise_redundancy_bits": (
                (info_gain.get("shapley") or {}).get("total_negative_pairwise_redundancy_bits")
            ),
        },
        "vote_counts": _vote_counts(votes),
    }
    return out


def _vote_counts(votes: list[str]) -> dict[str, int]:
    from collections import Counter

    return dict(Counter(votes))


def write_metrics_json(run_dir: str | Path, *, path: str | Path | None = None) -> Path:
    data = analyze_run_dir(run_dir)
    out_path = Path(path) if path else Path(run_dir) / "metrics.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_path
