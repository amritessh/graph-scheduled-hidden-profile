"""
Hard-elimination information-gain analysis with Shapley attribution.

Given a task with:
  - options/candidates
  - fact_eliminates (single-fact eliminations)
  - interaction_eliminates (combination-only eliminations)

and a run transcript, this module computes:
  - entropy trajectory H_t over feasible answer set size
  - per-atomic-fact delta-H events
  - waste/redundancy metrics
  - per-agent and per-category bit contributions
  - Shapley attribution over disclosed facts
  - pairwise interaction index (positive = synergy, negative = redundancy)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from gshp.aho_corasick import AhoCorasickAutomaton
from gshp.types import ExperimentRun


@dataclass
class FactEvent:
    dyad_index: int
    round_label: str
    message_index: int
    speaker: int
    fact_id: str
    fact_category: str
    is_repeat: bool
    newly_triggered_interactions: list[str]
    eliminated: list[str]
    feasible_before: list[str]
    feasible_after: list[str]
    entropy_before: float
    entropy_after: float
    delta_h: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dyad_index": self.dyad_index,
            "round_label": self.round_label,
            "message_index": self.message_index,
            "speaker": self.speaker,
            "fact_id": self.fact_id,
            "fact_category": self.fact_category,
            "is_repeat": self.is_repeat,
            "newly_triggered_interactions": self.newly_triggered_interactions,
            "eliminated": self.eliminated,
            "feasible_before": self.feasible_before,
            "feasible_after": self.feasible_after,
            "entropy_before": self.entropy_before,
            "entropy_after": self.entropy_after,
            "delta_h": self.delta_h,
        }


def analyze_information_gain(
    run: ExperimentRun,
    task: Any,
) -> dict[str, Any]:
    options = _task_options(task)
    if len(options) < 2:
        return {"error": "Task must expose at least 2 options/candidates."}

    fact_text = dict(getattr(task, "facts", {}) or {})
    fact_elim = dict(getattr(task, "fact_eliminates", {}) or {})
    interaction_elim_raw = dict(getattr(task, "interaction_eliminates", {}) or {})
    interaction_rules = _normalize_interactions(interaction_elim_raw)

    for fid in fact_text:
        fact_elim.setdefault(fid, [])

    categories = _fact_categories(task)
    hits_by_message = _extract_facts_by_message(run, fact_text)

    feasible = set(options)
    seen_facts: set[str] = set()
    triggered_interactions: set[str] = set()
    events: list[FactEvent] = []
    entropy_path: list[dict[str, Any]] = [{"t": 0, "entropy": _entropy(len(feasible)), "feasible": sorted(feasible)}]
    t = 0

    for key in sorted(hits_by_message.keys()):
        dyad_i, msg_i = key
        hit_fids = hits_by_message[key]
        if not hit_fids:
            continue
        dyad = run.dyads[dyad_i]
        speaker = int(dyad.messages[msg_i].role.split("_")[-1])
        for fid in hit_fids:
            t += 1
            before = sorted(feasible)
            h_before = _entropy(len(feasible))
            is_repeat = fid in seen_facts
            eliminated: set[str] = set()

            if not is_repeat:
                eliminated.update(x for x in fact_elim.get(fid, []) if x in feasible)

            seen_plus = set(seen_facts)
            seen_plus.add(fid)
            new_interactions: list[str] = []
            for ikey, rule in interaction_rules.items():
                if ikey in triggered_interactions:
                    continue
                if not set(rule["facts"]).issubset(seen_plus):
                    continue
                local_elim = [x for x in rule["eliminates"] if x in feasible]
                if local_elim:
                    eliminated.update(local_elim)
                new_interactions.append(ikey)
                triggered_interactions.add(ikey)

            feasible -= eliminated
            seen_facts.add(fid)
            h_after = _entropy(len(feasible))
            event = FactEvent(
                dyad_index=dyad_i,
                round_label=dyad.round_label,
                message_index=msg_i,
                speaker=speaker,
                fact_id=fid,
                fact_category=categories.get(fid, "unknown"),
                is_repeat=is_repeat,
                newly_triggered_interactions=sorted(new_interactions),
                eliminated=sorted(eliminated),
                feasible_before=before,
                feasible_after=sorted(feasible),
                entropy_before=h_before,
                entropy_after=h_after,
                delta_h=max(0.0, h_before - h_after),
            )
            events.append(event)
            entropy_path.append({"t": t, "entropy": h_after, "feasible": sorted(feasible)})

    disclosed = sorted({e.fact_id for e in events})
    shapley = _shapley_summary(disclosed, options, fact_elim, interaction_rules)

    bits_total = sum(e.delta_h for e in events)
    zero_events = sum(1 for e in events if e.delta_h == 0)
    bits_by_agent = _sum_by_key(events, key_fn=lambda e: str(e.speaker))
    bits_by_category = _sum_by_key(events, key_fn=lambda e: e.fact_category)
    count_by_category = _count_by_key(events, key_fn=lambda e: e.fact_category)
    avg_bits_by_category = {
        k: (bits_by_category.get(k, 0.0) / count_by_category[k]) if count_by_category[k] else 0.0
        for k in count_by_category
    }

    return {
        "options": options,
        "initial_entropy": _entropy(len(options)),
        "final_entropy": _entropy(len(feasible)),
        "resolved_to_single_option": len(feasible) == 1,
        "final_feasible_options": sorted(feasible),
        "n_atomic_events": len(events),
        "n_zero_bit_events": zero_events,
        "zero_bit_rate": (zero_events / len(events)) if events else 0.0,
        "total_bits_resolved": bits_total,
        "entropy_trajectory": entropy_path,
        "events": [e.to_dict() for e in events],
        "bits_by_agent": bits_by_agent,
        "bits_by_fact_category": bits_by_category,
        "avg_bits_per_event_by_category": avg_bits_by_category,
        "message_count_by_category": count_by_category,
        "disclosed_fact_ids": disclosed,
        "shapley": shapley,
    }


def _task_options(task: Any) -> list[str]:
    opts = getattr(task, "options", None)
    if opts:
        return [str(x) for x in opts]
    cands = getattr(task, "candidates", None)
    if cands:
        return [str(x) for x in cands]
    return []


def _fact_categories(task: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for fid in getattr(task, "shared_fact_ids", []) or []:
        out[str(fid)] = "shared"
    for _, fids in (getattr(task, "cluster_fact_ids", {}) or {}).items():
        for fid in fids:
            out[str(fid)] = "cluster"
    for _, fid in (getattr(task, "bridge_agent_fact_ids", {}) or {}).items():
        out[str(fid)] = "bridge"
    return out


def _extract_facts_by_message(
    run: ExperimentRun,
    fact_text: dict[str, str],
) -> dict[tuple[int, int], list[str]]:
    fids = sorted(fact_text.keys())
    patterns = [fact_text[fid].strip().lower() for fid in fids if (fact_text[fid] or "").strip()]
    ac = AhoCorasickAutomaton(patterns) if patterns else None

    out: dict[tuple[int, int], list[str]] = {}
    for dyad_i, dyad in enumerate(run.dyads):
        for msg_i, msg in enumerate(dyad.messages):
            text = msg.content or ""
            low = text.lower()
            found: set[str] = set()

            # Match by fact-id mention.
            for fid in fids:
                if fid.lower() in low:
                    found.add(fid)

            # Match by exact fact text mention via AC indices.
            if ac is not None and text.strip():
                hit_idx = set(ac.matching_pattern_indices(low))
                non_empty_fids = [fid for fid in fids if (fact_text.get(fid) or "").strip()]
                for hi in hit_idx:
                    if 0 <= hi < len(non_empty_fids):
                        found.add(non_empty_fids[hi])

            # Direct substring fallback for robustness.
            for fid in fids:
                needle = (fact_text.get(fid) or "").strip().lower()
                if needle and needle in low:
                    found.add(fid)

            if not found:
                continue

            # Approximate split-order by first appearance position in message text.
            ranked = sorted(
                found,
                key=lambda fid: _first_pos(low, fid.lower(), (fact_text.get(fid) or "").strip().lower()),
            )
            out[(dyad_i, msg_i)] = ranked
    return out


def _first_pos(text: str, fid_needle: str, fact_needle: str) -> int:
    p1 = text.find(fid_needle) if fid_needle else -1
    p2 = text.find(fact_needle) if fact_needle else -1
    vals = [p for p in (p1, p2) if p >= 0]
    return min(vals) if vals else 10**9


def _normalize_interactions(raw: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, elim in raw.items():
        fids = [x.strip() for x in str(key).split("+") if x.strip()]
        if len(fids) < 2:
            continue
        ikey = "+".join(sorted(fids))
        out[ikey] = {"facts": tuple(sorted(fids)), "eliminates": [str(x) for x in (elim or [])]}
    return out


def _entropy(k: int) -> float:
    if k <= 1:
        return 0.0
    return math.log2(float(k))


def _sum_by_key(events: list[FactEvent], key_fn) -> dict[str, float]:
    out: dict[str, float] = {}
    for e in events:
        k = key_fn(e)
        out[k] = out.get(k, 0.0) + e.delta_h
    return out


def _count_by_key(events: list[FactEvent], key_fn) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in events:
        k = key_fn(e)
        out[k] = out.get(k, 0) + 1
    return out


def _value_of_subset(
    subset: set[str],
    options: list[str],
    fact_elim: dict[str, list[str]],
    interaction_rules: dict[str, dict[str, Any]],
) -> float:
    feasible = set(options)
    for fid in subset:
        feasible -= set(x for x in fact_elim.get(fid, []) if x in feasible)
    for rule in interaction_rules.values():
        if set(rule["facts"]).issubset(subset):
            feasible -= set(x for x in rule["eliminates"] if x in feasible)
    return _entropy(len(options)) - _entropy(len(feasible))


def _shapley_summary(
    facts: list[str],
    options: list[str],
    fact_elim: dict[str, list[str]],
    interaction_rules: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not facts:
        return {
            "total_value_bits": 0.0,
            "per_fact": {},
            "pairwise_interaction_bits": {},
            "total_positive_pairwise_synergy_bits": 0.0,
            "total_negative_pairwise_redundancy_bits": 0.0,
        }

    n = len(facts)
    idx = {f: i for i, f in enumerate(facts)}
    fact_set = set(facts)

    # Cache v(S) over all subsets.
    v_cache: dict[int, float] = {}
    for mask in range(1 << n):
        subset = {facts[i] for i in range(n) if (mask >> i) & 1}
        v_cache[mask] = _value_of_subset(subset, options, fact_elim, interaction_rules)

    import math as _m
    n_fact = _m.factorial(n)
    shapley: dict[str, float] = {f: 0.0 for f in facts}
    for f in facts:
        i = idx[f]
        for mask in range(1 << n):
            if (mask >> i) & 1:
                continue
            s_size = _popcount(mask)
            weight = (_m.factorial(s_size) * _m.factorial(n - s_size - 1)) / n_fact
            with_i = mask | (1 << i)
            marginal = v_cache[with_i] - v_cache[mask]
            shapley[f] += weight * marginal

    pairwise: dict[str, float] = {}
    pos = 0.0
    neg = 0.0
    for a, b in combinations(facts, 2):
        ma = 1 << idx[a]
        mb = 1 << idx[b]
        vab = v_cache[ma | mb]
        va = v_cache[ma]
        vb = v_cache[mb]
        inter = vab - va - vb  # >0 synergy, <0 redundancy
        key = f"{a}+{b}"
        pairwise[key] = inter
        if inter > 0:
            pos += inter
        elif inter < 0:
            neg += abs(inter)

    per_fact: dict[str, Any] = {}
    for f in facts:
        alone = v_cache[1 << idx[f]]
        per_fact[f] = {
            "shapley_bits": shapley[f],
            "standalone_unique_bits": alone,
            "context_dependent_bits": shapley[f] - alone,
        }

    return {
        "total_value_bits": v_cache[(1 << n) - 1],
        "n_facts": len(facts),
        "facts_considered": sorted(fact_set),
        "per_fact": per_fact,
        "pairwise_interaction_bits": pairwise,
        "total_positive_pairwise_synergy_bits": pos,
        "total_negative_pairwise_redundancy_bits": neg,
    }


def _popcount(x: int) -> int:
    return x.bit_count()
