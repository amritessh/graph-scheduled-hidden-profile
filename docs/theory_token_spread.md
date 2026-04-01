# Theory Layer: Scheduled Token Spread on Ring-of-Cliques

This document describes the graph-theoretic contribution that runs alongside the LLM experiment.
The theory and the experiment are independent — the experiment can run and produce results before
the theorems are proved. They meet in the paper's results section.

---

## What the experiment is, formally

**Graph** `G = (V, E)`: 9 nodes arranged as a ring of 3 cliques of size 3 (the `full_clique_ring`
topology). Edges are either *intra* (within a clique) or *inter* (between bridge nodes across
cliques).

**Tokens**: Each agent starts with a subset of labeled facts. Facts are heterogeneous — a
cluster-1 unique fact is different from a cluster-2 unique fact and cannot substitute for it.

**Communication**: Each round, a *matching* M ⊆ E is activated (no agent appears in more than
one dyad per round). When edge (u,v) activates, u and v exchange tokens.

**Two-phase constraint**: The experiment enforces either:
- *Intra-first*: all intra-edges must activate before any inter-edge
- *Inter-first*: all inter-edges must activate before any intra-edge

The manipulation is *only the ordering of edge activation*. The graph, the tokens, and the
agents are identical across conditions.

---

## The core theory question

> On a ring of `l` cliques of size `k` with bridge-to-bridge inter-edges, what is the minimum
> number of parallel matching rounds for a single private token — held by one agent in one
> cluster — to reach **all** agents, under intra-first vs inter-first vs optimal interleaving?

For the experiment's fixed parameters (`l=3`, `k=3`), the answer can be computed exactly.
For general `(l, k)`, the goal is a Θ(·) bound.

---

## Why this matters for the experiment

The theorem would show that inter-first scheduling **strictly reduces** the number of rounds
needed for a unique fact to reach all agents, and that the gap grows with `k` (clique size).

This gives the experiment a **theoretical prediction to validate**: the empirical accuracy gap
between bridge-first and cluster-first conditions should track the round-complexity gap the
theorem predicts. If LLM behavior tracks the structural prediction, that's a strong result.
If it deviates (e.g., ToM closes the gap even under cluster-first), that deviation is itself
a meaningful finding — behavioral/cognitive factors compensate for structural disadvantage.

---

## The entrenchment mechanism, formally

Under intra-first scheduling, agents complete all within-cluster conversations before any
bridge conversation. By the time the bridge activates, every non-bridge agent has already
formed a preliminary belief from shared information alone (which points to the wrong answer,
X). The bridge then has to transmit a unique fact *against* an already-formed belief.

Under inter-first, the bridge activates before within-cluster consensus forms. Unique facts
reach the bridge node first, then spread within clusters — so within-cluster conversations
happen *after* agents have already heard cross-cluster information.

Formally: under intra-first, the token from cluster 1 cannot reach any agent in cluster 2
until round `≥ ⌈log₂(k)⌉ + 1` (rounds to cover the clique, plus one inter-round). Under
inter-first, it can reach cluster 2's bridge in round 1.

---

## The starter theorem (target formulation)

**Family**: Ring of `l` cliques of size `k`. Bridge nodes = last agent of each clique.
Inter-edges form a complete graph on bridge nodes (your `full_clique_ring`).

**Token**: One private fact held by one non-bridge agent in cluster 0.

**Question**: Under optimal scheduling within the two-phase constraint, what is:
- `T_intra(l, k)` — minimum rounds under intra-first
- `T_inter(l, k)` — minimum rounds under inter-first

**Conjecture**: `T_inter(l, k) < T_intra(l, k)` for all `k ≥ 2`, with the gap `Θ(k)`.

For `l=3, k=3` (the experiment topology):
- Intra-first: token must spread within clique 0 (≥1 round), then cross to bridge B2 via A2
  (1 inter-round), then spread within clique 1 (≥1 round), then cross to clique 2 — total ≥4
  rounds just for reachability, more for full coverage.
- Inter-first: bridge A2 activates immediately; unique fact reaches B2 and C2 in round 1;
  within-cluster spread happens in rounds 2+. Full coverage reachable in fewer total rounds.

---

## Connection to the literature

| This problem | Standard name | Key references |
|---|---|---|
| Min rounds for all agents to learn a token | Gossip / broadcasting | Hedetniemi et al. 1988; Fraigniaud & Lazard 1994 |
| Sequence of matchings on a graph | Round-based gossip / telephone model | Tijdeman 1971; Knödel 1975 |
| Edge activation restricted by time intervals | Temporal graphs | Kempe, Kleinberg & Kumar 2000; Holme & Saramäki 2012 |
| Optimal ordering of two edge sets | Two-phase scheduling / interleaving | Open — this is the novel contribution |
| Heterogeneous tokens (facts ≠ interchangeable) | Structured rumor mixing | Largely open |

The most directly related work is the **telephone gossip model** (each round = one call per
person, minimize calls until everyone knows everything). Your setting is more constrained:
edges are partitioned into two phases, and you're comparing phase orderings, not minimizing
total rounds globally.

---

## How to develop this

1. **Fix `l=3, k=3`**: Prove the exact round counts for intra-first and inter-first by
   explicit construction of optimal schedules. This is tractable by hand.

2. **Generalize to `(l, k)`**: Express `T_intra` and `T_inter` as functions of the parameters.
   Identify the regime where the gap is largest.

3. **Interleaving lower bound**: Show that no interleaving of intra and inter matchings can
   beat inter-first by more than a constant. (Or find a case where it does — that would be
   the interesting finding.)

4. **Connect to experiment**: Plot the theoretical round-complexity gap alongside the
   empirical accuracy gap across conditions. If they correlate, the structural model explains
   the behavioral data.

---

## Pitch to the professor (one paragraph)

> The experiment is an instance of scheduled token spread on a temporal graph — specifically,
> a ring of cliques under a two-phase edge activation constraint. I want to prove the
> round-complexity bounds for this topology (minimum matching rounds for a unique fact to reach
> all agents under intra-first vs inter-first), then use the LLM results to test whether
> behavior tracks the theoretical prediction. This turns the empirical study into an empirical
> validation of a structural prediction: the accuracy gap between conditions should scale with
> the theoretical round-complexity gap. The theory layer doesn't change the experiment at all —
> it provides the formal justification for why the manipulation works.
