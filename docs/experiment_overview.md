# The Experiment: What It Is and What It Tests

## The Core Question

When a group of agents needs to make a decision together, does the *order* in which they talk to each other determine whether they get the right answer?

This sounds simple. It isn't. The experiment is designed to prove that three things — the order of conversations, genuinely distributed information, and a specific cognitive ability at network bottlenecks — must be present *simultaneously* for a group to reach correct collective decisions. Each condition alone is insufficient. Only the conjunction works.

---

## The Setup

### The network

Nine agents arranged in three groups of three (cliques), connected by bridge nodes:

```
  Cluster A         Cluster B         Cluster C
[A0 - A1 - A2] -- [B0 - B1 - B2] -- [C0 - C1 - C2]
          |_________________________________|
               (ring: A2 ↔ B2 ↔ C2 ↔ A2)
```

- Within each cluster: all three agents can talk to each other
- Between clusters: only the bridge nodes (A2, B2, C2) are connected
- Bridge nodes are the **only path** for information to cross cluster boundaries

This is a "caveman graph": dense within, sparse between.

### The task

All nine agents are on a hiring panel. They must choose one of three candidates: **X, Y, or Z**.

The correct answer is **Y**. But:

- **Shared information** (all agents see this): four strong positive facts about X, one weak fact about Y, one weak fact about Z. If an agent only had the shared information, they would rationally choose X.
- **Cluster-unique information** (one cluster each): two facts per cluster, each making Y look much stronger
- **Bridge-unique information** (one bridge agent each): one "linking" fact that connects two clusters' information

The key design principle: **shared information creates a strong attractor toward X (wrong); unique information is required to identify Y (right).** An agent who never receives unique facts from other clusters will keep choosing X, even after talking to peers.

This is the **hidden profile paradigm** (Stasser & Titus, 1985): groups systematically fail because they keep discussing what everyone already knows rather than the unique information that would change the answer.

---

## The Three Experimental Factors

### Factor 1: Communication Timing

**Within-first**: Each cluster has all its internal conversations before bridge nodes talk across clusters.

**Bridge-first**: Bridge nodes talk across clusters first, before internal cluster discussions happen.

Why does this matter? If clusters discuss internally first, they form a shared view — a local consensus — before the bridge transmits information. By the time cross-cluster information arrives, the cluster has already "locked in" to a position. The new information has to fight entrenched beliefs.

If bridges talk first, the unique information travels before anyone has formed a strong opinion. When the cluster then discusses internally, they process a richer information set.

### Factor 2: Information Structure

**Hidden profile**: As described above — shared info favors X, unique info reveals Y.

**Shared only**: Remove all unique and bridge-unique information. Every agent has the same six shared facts. There is no "right answer" that requires pooling unique information — only coordination.

Why include the shared-only condition? Because it's a **control**. If the timing effect (Factor 1) exists even in shared-only conditions, then timing is just a coordination effect — it doesn't require information asymmetry. If timing only matters in the hidden profile condition, then information asymmetry is a necessary condition for the effect to operate. This lets us isolate the mechanism.

### Factor 3: Theory of Mind at Bridge Nodes

**No ToM**: Bridge agents get a standard prompt — share what you know, work toward a good decision.

**ToM**: Bridge agents get an additional instruction: *Before each conversation, consider what the other person likely knows and doesn't know. Frame your information based on their specific knowledge gaps.*

Why only at bridge nodes? Because the theoretical claim is that ToM at the *structural bottleneck* is what converts access to unique information into actual information transfer. A bridge can hold linking facts but still fail to transmit them usefully if it just relays what it heard ("Agent A said Y is strong because X") without framing it for the recipient's knowledge state ("You know Y's technical scores are exceptional — I can tell you that those are precisely the skills Agent A's cluster flagged as critical").

---

## The Eight Conditions (2 × 2 × 2)

| # | Timing | Info Structure | ToM | Expected Result |
|---|--------|---------------|-----|-----------------|
| 1 | Within-first | Shared-only | No | High convergence, wrong answer (X) — pure shared-info attractor |
| 2 | Within-first | Hidden profile | No | High convergence, still wrong — unique info suppressed before bridge activates |
| 3 | Bridge-first | Shared-only | No | Moderate convergence, wrong — timing preserved diversity but nothing unique to transmit |
| 4 | Bridge-first | Hidden profile | No | Moderate convergence, partially correct — unique info reaches bridge but relay mode loses it |
| 5 | Within-first | Shared-only | ToM | Same as #1 — ToM has nothing to translate |
| 6 | Within-first | Hidden profile | ToM | Moderate — ToM corrects some entrenchment but faces resistance |
| 7 | Bridge-first | Shared-only | ToM | Same as #3 — timing + ToM without unique info produces only coordination |
| **8** | **Bridge-first** | **Hidden profile** | **ToM** | **High convergence, correct answer (Y) — the conjunction works** |

The two "null results" (conditions 1 and 7 being identical in outcome) are not failures — they are the key theoretical control. They show that timing and ToM effects require real epistemic content to operate.

---

## What We Measure

### DV1: Decision Accuracy (the main outcome)
- Did individual agents choose Y?
- Did clusters reach correct consensus?
- Did the network-level group decision get it right?

This is what Momennejad (2019) couldn't measure — their experiment had no correct answer, only convergence. We can distinguish "everyone agrees on X" from "everyone agrees on Y."

### DV2: Fact Transmission (the mechanism check)
For each unique fact in the task, we track three stages independently:

1. **Disclosure**: Was the fact mentioned in any conversation at all?
2. **Transmission**: Did it appear in a conversation that crossed a cluster boundary?
3. **Integration**: Did it appear in any agent's final decision justification?

Three separate failure modes: the agent never shares it → they share it but it doesn't cross the boundary → it crosses the boundary but nobody uses it. The timing × ToM interaction should show up strongest in the transmission rate — specifically, whether unique facts cross the bridge before or after local consensus forms.

### DV3: Convergence vs. Alignment
- **Convergence**: Do agents agree with each other? (standard measure, Momennejad-style)
- **Alignment**: Are they agreeing on the *right* answer?

High convergence + low alignment = "spurious consensus" — everyone agrees on X. This is the failure mode that prior work couldn't detect. The critical prediction: cluster-first + no-ToM produces high convergence but low alignment. Bridge-first + ToM produces both.

### DV4: Bridge Communication Quality
For each message a bridge agent sends in a cross-cluster conversation, an LLM judge classifies it as:

- **Relay**: "Agent A said Y is strong" — just passing along information
- **Filter**: Selectively shares some things without adapting the framing
- **Translate**: "You mentioned concern about X's technical depth — I can tell you that the problem Y solved in Cluster A is exactly the technical domain you identified as critical"

The prediction: ToM-prompted bridges operate in translate mode. Translate mode → unique info crosses boundary (DV2 transmission). Transmission → correct decision (DV1). This is the full causal chain.

---

## The Causal Chain Being Tested

```
  Timing              Info structure        ToM at bridge
  (bridge-first)      (hidden profile)      (translate mode)
       |                    |                     |
       ↓                    ↓                     ↓
  Diversity          Unique info           Epistemic framing
  preserved          worth sharing         (not just relay)
  before             exists
  entrenchment
       \                    |                    /
        \                   ↓                   /
         ──────→  Unique info crosses bridge ←───
                            |
                            ↓
                  Unique info in final reasoning
                            |
                            ↓
                   Correct decision (Y wins)
                   NOT just convergence
```

Every arrow is testable. Every arrow can fail independently. That's what makes this falsifiable rather than just confirmatory.

---

## Scale and Cost

Each "run" = one complete experiment with 9 agents:
- 12 dyadic conversations (within + cross cluster)
- 9 individual decisions
- Optional: 9 group deliberation decisions

At 50 runs × 8 conditions = **400 total runs**, each at ~5–10 minutes on a local GPU running Qwen3 at 600 tokens/second. The full factorial takes a weekend of compute.

---

## How the Code Implements This

| Study design element | Code location |
|---------------------|---------------|
| Caveman topology | `gshp/graph/caveman.py` |
| Information distribution (facts per agent) | `gshp/task/hiring.py` |
| Within-first / bridge-first schedule | `gshp/schedule.py` |
| Dyadic conversations | `gshp/session.py` |
| Agent system prompts + ToM block | `gshp/prompts.py` |
| Main experiment loop | `gshp/experiment.py` |
| Group deliberation round | `gshp/deliberation.py` |
| DV2 fact transmission | `gshp/fact_tracker.py` |
| DV4 bridge mode judge | `gshp/bridge_coder.py` |
| Full artifact capture | `gshp/artifacts.py` |
| Batch factorial runner | `gshp/batch.py` |
| CLI | `gshp/cli.py` |

A complete run with all features:
```bash
python -m gshp.cli run \
  --model vllm:8000/Qwen/Qwen3-8B \
  --schedule cross_first \          # bridge-first
  --condition hidden_profile \
  --tom-bridge \
  --parallel-dyads --workers 8 \    # parallel dyads within each matching layer
  --group-deliberation \            # group deliberation round after individual decisions
  --artifact-dir results/run_001
```

---

## What This Study Can Claim That Prior Work Cannot

| Claim | Momennejad (2019) | This study |
|-------|------------------|------------|
| Timing of bridge conversations affects collective outcomes | ✓ | ✓ |
| The effect requires genuinely distributed information | ✗ not tested | ✓ |
| ToM at bridge nodes is the cognitive mechanism | ✗ not tested | ✓ |
| Convergence ≠ alignment; can dissociate them | ✗ | ✓ |
| The mechanism operates via translate mode at the bridge | ✗ | ✓ |
| Unique info suppression is the within-cluster failure mode | ✗ asserted | ✓ measured |

---

## One CS-Heavy Addition: The Adaptive Orchestrator

See `docs/adaptive_orchestrator.md` for the full idea. Short version:

The experiment currently compares two **fixed** schedules (bridge-first vs. cluster-first). A natural CS extension is to replace the fixed schedule with an **orchestrator agent** that dynamically decides which dyad to activate next. This turns the scheduling question from an experimental manipulation into an algorithmic problem:

> *Can an intelligent scheduler, observing only the current epistemic state of the network, learn to sequence conversations in a way that outperforms both fixed protocols?*

This shifts the paper from "we found that bridge-first works" to "we built a system that learns *why* bridge-first works and exploits it adaptively." That's a CS contribution.
