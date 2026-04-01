"""Aggregate metrics from logged LLM calls (latency, token usage)."""

from __future__ import annotations

from typing import Any

from gshp.aho_corasick import AhoCorasickAutomaton


def aggregate_llm_call_stats(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum latencies and OpenAI-style usage fields when present on ``openai_completion``."""
    wall_ms = sum(float(c.get("latency_ms") or 0) for c in calls)
    pt = ct = tt = 0
    n_usage = 0
    for c in calls:
        u = c.get("usage")
        if not isinstance(u, dict):
            oc = c.get("openai_completion")
            if isinstance(oc, dict):
                u = oc.get("usage")
        if not isinstance(u, dict):
            continue
        n_usage += 1
        pt += int(u.get("prompt_tokens") or 0)
        ct += int(u.get("completion_tokens") or 0)
        tt += int(u.get("total_tokens") or 0)
    return {
        "num_llm_calls": len(calls),
        "sum_latency_ms": round(wall_ms, 3),
        "calls_with_usage": n_usage,
        "usage_prompt_tokens": pt,
        "usage_completion_tokens": ct,
        "usage_total_tokens": tt,
    }


def fact_mention_rates(
    task_facts: dict[str, str],
    llm_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Disclosure heuristic: literal **substring** match of each fact's *text* (not only id) in
    all prompts + responses. Implemented with an **Aho–Corasick** scan over one lowercased blob
    (multi-pattern, linear in transcript length). See ``docs/algorithms.md`` for caveats.
    """
    blob_parts: list[str] = []
    for c in llm_calls:
        blob_parts.append(c.get("system") or "")
        for m in c.get("messages") or []:
            if isinstance(m, dict):
                blob_parts.append(m.get("content") or "")
        blob_parts.append(c.get("response") or "")
    blob = "\n".join(blob_parts).lower()

    # Stable order → reproducible pattern indices (sorted fact ids).
    per_fact: dict[str, bool] = {fid: False for fid in task_facts}
    pattern_fids: list[str] = []
    patterns: list[str] = []
    for fid in sorted(task_facts.keys()):
        needle = (task_facts[fid] or "").strip().lower()
        if needle:
            pattern_fids.append(fid)
            patterns.append(needle)

    if patterns:
        ac = AhoCorasickAutomaton(patterns)
        for hit_i in ac.matching_pattern_indices(blob):
            per_fact[pattern_fids[hit_i]] = True

    disclosed = sum(1 for v in per_fact.values() if v)
    return {
        "facts_checked": len(per_fact),
        "facts_mentioned_anywhere": disclosed,
        "per_fact_mentioned": per_fact,
        "matcher": "aho_corasick",
    }
