# Algorithms: parallel dyad scheduling & fact detection

This note describes two **engineering** choices in `gshp`: (1) splitting each communication phase into **parallel-safe layers** using graph matchings, and (2) **multi-pattern substring search** (Aho–Corasick) for cheap fact-disclosure metrics over transcripts.

---

## 1. Parallel dyad layers (graph matchings)

### Problem

Each **phase** of the schedule lists many undirected dyads \((u, v)\). If we ever want **simultaneous** conversations (or simply want to reason about “who could talk at the same time?”), we need groups of dyads where **no agent appears in two dyads at once**. In graph terms, each group must be a **matching**: a set of edges with **pairwise disjoint endpoints**.

The default schedule still runs dyads **one after another** in code; layers are a **structured decomposition** of the phase. They are useful for:

- Future **async** execution without agent double-booking.
- Reporting **depth** of a phase (how many matching layers are needed).
- Clearer alignment with networked-experiment metaphors where **matching rounds** are standard.

### Method

Given a finite simple edge set \(E\) (our dyads as unordered pairs):

1. Build graph \(G = (V, E)\).
2. Compute a **maximum-cardinality matching** \(M \subseteq E\) (NetworkX: `max_weight_matching(..., maxcardinality=True)` on an unweighted graph).
3. Remove \(M\) from \(E\); append \(M\) as the next **layer**.
4. Repeat until \(E\) is empty.

Each layer is a matching by construction. Removing a maximum matching each time is a standard greedy **edge partition**; it is not guaranteed to minimize the **number** of layers over all possible edge colorings, but it is fast, deterministic (given the library implementation), and always valid.

**Complexity:** One maximum matching on a graph with \(|E|\) edges is polynomial (Edmonds’ algorithm in NetworkX). We repeat at most \(|E|\) times, so worst-case \(O(|E|^2 \cdot \mathrm{poly}(|V|))\) is loose; in practice \(|E|\) is small (dozens) on our topologies.

### Code map

| Piece | Location |
|--------|-----------|
| Partition edges → layers | `gshp/matching_schedule.py` → `partition_edges_into_matching_layers` |
| Expand two-phase schedule | `gshp/schedule.py` → `expand_schedule_parallel_matchings` |
| CLI / manifest flag | `--parallel-dyads`; `RunManifest.parallel_dyad_layers` |
| Runner / experiment | `gshp/runner.py`, `gshp/experiment.py` |
| Protocol field | `canonical_protocol_dict(..., parallel_dyad_layers=...)` in `gshp/protocol.py` |

### Comparison to “single list” scheduling

| Mode | Behavior |
|------|-----------|
| **Default** | One `CommunicationRound` per phase; `sub_index` is always `0`. Edges run in sorted tuple order. |
| **`--parallel-dyads`** | Each phase becomes several rounds with the same `index` / `label` but `sub_index = 0, 1, …`; each round’s `edges` form a matching. Total dyad count unchanged. |

Transcripts record `round_sub_index` on `DyadTranscript` for traceability.

---

## 2. Fact / evidence detection (Aho–Corasick)

### Problem

For each fact id we have a **literal text string** (the fact wording). A coarse **disclosure** metric asks: did that exact wording appear anywhere in the concatenated prompts and model outputs?

Naively, for \(k\) facts and a blob of length \(n\), repeated substring search is \(O(k \cdot n)\). With tens of facts and long logs, a **multi-pattern** scan is preferable.

### Method

We use a **pure-Python Aho–Corasick automaton** (`gshp/aho_corasick.py`):

- **Build:** Insert all non-empty fact strings (lowercased) into a trie; compute failure links and output sets in one BFS (standard construction).
- **Scan:** Walk the lowercased blob once; each step may emit one or more **pattern indices** that end at the current position.

**Complexity:** \(O(\sum_i |p_i| + n + \#\mathrm{matches})\) with alphabet in trie degree; memory \(O(\sum_i |p_i|)\).

Fact ids are mapped to patterns in **sorted fact-id order** so `per_fact_mentioned` is **reproducible** independent of dict insertion order.

`gshp/metrics.fact_mention_rates` sets `"matcher": "aho_corasick"` in its return dict (for provenance in `metrics.json`).

### Limits (important)

| Issue | Consequence |
|-------|-------------|
| **Literal strings only** | Paraphrases, negation, or coreference (“that finding”) are **not** counted. |
| **Case / Unicode** | We lowercase the blob and fact text; no aggressive normalization (hyphens, digits, stemming). |
| **Substring false positives** | A short fact phrase can appear inside unrelated words; longer, distinctive fact texts reduce this. |
| **Empty fact text** | Treated as never mentioned. |

### Suggested upgrades (research path)

1. **Calibration:** Human or LLM-coded gold on a sample of turns; compare precision/recall of substring vs improved rules.
2. **LLM-as-judge** on spans (expensive, richer).
3. **Normalized forms** (strip punctuation, optional lemma) before AC — keep a second channel so raw-literal and normalized metrics don’t conflate.

---

## 3. Tests

- Matching layers: `tests/test_matching_schedule.py`
- AC + parity with naive substring check: `tests/test_aho_corasick.py`

---

## 4. Protocol version

`parallel_dyad_layers` is part of the canonical protocol dict (`protocol_version` **1.1.0**). Toggling it changes **`protocol_sha256`**, as intended for reproducibility.
