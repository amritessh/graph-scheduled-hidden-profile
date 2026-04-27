"""
DV3 — Convergence vs Alignment dissociation.

This module provides run-level metrics that separate:
  - convergence: how much agents agree with each other
  - alignment: how much agents agree with the correct answer

The study write-up references PID-style alignment analysis. That requires multi-run
time-series and dedicated information-theoretic tooling. Here we implement robust,
artifact-friendly run-level metrics that expose the key dissociation pattern.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from gshp.types import AgentDecision, ExperimentRun


def convergence_alignment_metrics(
    run: ExperimentRun,
    *,
    correct_choice: str,
) -> dict[str, Any]:
    """Compute DV3-style convergence/alignment metrics for one run."""
    votes = [d.choice for d in run.final_decisions if d.choice]
    n = len(votes)
    if n <= 1:
        return {
            "n_agents_with_vote": n,
            "pairwise_convergence": 0.0,
            "alignment_accuracy": 0.0,
            "majority_choice": votes[0] if votes else None,
            "majority_correct": bool(votes) and votes[0] == correct_choice,
            "dissociation_gap": 0.0,
        }

    pairwise_convergence = _pairwise_agreement(votes)
    alignment_accuracy = sum(1 for v in votes if v == correct_choice) / n
    majority = Counter(votes).most_common(1)[0][0]

    return {
        "n_agents_with_vote": n,
        "pairwise_convergence": pairwise_convergence,
        "alignment_accuracy": alignment_accuracy,
        "majority_choice": majority,
        "majority_correct": majority == correct_choice,
        # Positive => agreement outpaces correctness (spurious consensus signature).
        "dissociation_gap": pairwise_convergence - alignment_accuracy,
    }


def convergence_alignment_by_cluster(
    decisions: list[AgentDecision],
    *,
    topo_l: int,
    topo_k: int,
    correct_choice: str,
) -> dict[str, Any]:
    """Cluster-level convergence/alignment breakdown."""
    out: dict[str, Any] = {}
    for ci in range(topo_l):
        agent_ids = [ci * topo_k + j for j in range(topo_k)]
        votes = [d.choice for d in decisions if d.agent_id in agent_ids and d.choice]
        n = len(votes)
        if n == 0:
            out[str(ci)] = {"n_agents_with_vote": 0}
            continue
        pairwise = _pairwise_agreement(votes) if n > 1 else 0.0
        accuracy = sum(1 for v in votes if v == correct_choice) / n
        majority = Counter(votes).most_common(1)[0][0]
        out[str(ci)] = {
            "n_agents_with_vote": n,
            "pairwise_convergence": pairwise,
            "alignment_accuracy": accuracy,
            "majority_choice": majority,
            "majority_correct": majority == correct_choice,
            "dissociation_gap": pairwise - accuracy,
        }
    return out


def _pairwise_agreement(votes: list[str]) -> float:
    """Fraction of unordered agent pairs that end with identical choices."""
    n = len(votes)
    if n <= 1:
        return 0.0
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if votes[i] == votes[j]:
                agree += 1
    return agree / total if total else 0.0
