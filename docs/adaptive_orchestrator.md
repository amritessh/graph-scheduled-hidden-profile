# The CS Addition: Adaptive Orchestrator Agent

## The Gap in the Current Design

The study compares two fixed schedules:
- **Bridge-first**: always activate cross-cluster conversations before within-cluster
- **Cluster-first**: always activate within-cluster before cross-cluster

This is clean experimentally. But it treats the scheduling problem as a binary choice made by the experimenter. A more CS-heavy framing asks: *can an agent learn the optimal communication order dynamically, without being pre-committed to either protocol?*

If yes — and if the adaptive scheduler learns to do something like bridge-first — that's not just a replication of the timing effect. It's a mechanistic explanation encoded as an algorithm. You've built a system that "understands" why timing matters.

---

## The Idea

Add a fourth experimental condition: **adaptive scheduling**.

Instead of a fixed two-phase schedule, an **orchestrator agent** observes the current state of the network and decides, at each step, which dyad to activate next.

```
                     ┌─────────────────────────┐
                     │     ORCHESTRATOR         │
                     │                          │
                     │  Observes: epistemic      │
                     │  state estimates          │
                     │                          │
                     │  Decides: which dyad     │
                     │  to activate next         │
                     └────────────┬─────────────┘
                                  │  "activate (A2, B2)"
                          ┌───────▼───────┐
                          │  Conversation  │
                          │  A2 ↔ B2      │
                          └───────┬───────┘
                                  │  transcript
                     ┌────────────▼─────────────┐
                     │  Updated epistemic state  │
                     │  fed back to orchestrator │
                     └──────────────────────────┘
```

The orchestrator does not participate in conversations — it only decides *who* talks *when*.

---

## Two Versions (Increasing Sophistication)

### Version A: Greedy Information-Theoretic Scheduler

**No LLM required for the orchestrator.**

After each conversation, estimate how much unique information each possible next dyad would likely propagate. Activate the dyad with the highest expected information gain.

The information gain estimate: for each possible dyad (u, v), estimate the size of the *symmetric difference* between u's known facts and v's known facts. The dyad that would exchange the most new information is activated next.

```python
def choose_next_dyad(
    remaining_edges: list[tuple[int, int]],
    agent_believed_knowledge: dict[int, set[str]],  # estimated per-agent knowledge
) -> tuple[int, int]:
    scores = {}
    for u, v in remaining_edges:
        ku = agent_believed_knowledge[u]
        kv = agent_believed_knowledge[v]
        # symmetric difference = what each would learn from the other
        score = len(ku.symmetric_difference(kv))
        scores[(u, v)] = score
    return max(scores, key=scores.get)
```

The orchestrator maintains an estimated knowledge state per agent, updated after each conversation's transcript is observed. This is an approximation — the orchestrator doesn't know exactly what was said, but can estimate based on the conversation transcript.

**Why this is interesting:** If the greedy scheduler converges to a bridge-first order, it explains *why* bridge-first works — because bridge conversations have the highest information gain early on. If it doesn't, you learn something new.

### Version B: LLM Orchestrator (Planner Agent)

The orchestrator is itself an LLM given a "meta" system prompt:

> You are an orchestrator managing a 9-agent hiring panel. Your job is to decide which pair of agents should have a conversation next. You have access to a summary of what each agent currently knows (based on their conversations so far). Your goal is to maximize the group's chance of reaching the correct hiring decision by sequencing conversations optimally.
>
> Remaining possible conversations: [list]
> Current knowledge summaries: [per-agent summaries]
>
> Which conversation should happen next? Reply with JSON: {"dyad": [u, v], "reasoning": "..."}

This is where the agentic AI angle becomes explicit: the orchestrator is itself an agent with a meta-task (optimize the collective decision process), and it reasons over the state of other agents.

**Why this is more interesting than Version A:** The orchestrator's reasoning trace is observable. You can analyze *why* it chooses each dyad, whether its reasoning is consistent with the information-theoretic optimum, and whether it learns to recognize the hidden profile structure. Its behavior is interpretable in a way that a fixed schedule is not.

---

## What This Adds to the Paper

### A third scheduling condition
The 2×2×2 design becomes 3×2×2 (or you add "adaptive" as a fourth timing condition):

| Timing | Info | ToM | Prediction |
|--------|------|-----|-----------|
| Within-first | Hidden profile | No | Low accuracy |
| Bridge-first | Hidden profile | No | Moderate accuracy |
| **Adaptive** | **Hidden profile** | **No** | **≥ Bridge-first** |

If adaptive ≥ bridge-first: the orchestrator learned the bridge-first principle. If adaptive > bridge-first: it found something even better. If adaptive < bridge-first: the orchestrator failed — interesting, why?

### A mechanistic explanation
The greedy scheduler's dyad-choice sequence can be analyzed: does it consistently activate bridge conversations early? Does the pattern match bridge-first, or does it do something more nuanced (e.g., alternating: bridge → intra → bridge → intra)?

### A contribution to multi-agent orchestration
Most orchestration work in AI systems uses fixed, pre-programmed communication topologies. This shows what an intelligent scheduler can recover from the task structure alone — without being told "use bridges first." That's relevant to anyone building agentic pipelines.

### A connection to mechanism design / active sensing
The greedy scheduler is equivalent to an **active sensing** agent: it observes the network state and takes actions (activating dyads) to maximize information gain. This connects the work to a large CS literature on adaptive information gathering, Bayesian experimental design, and influence maximization in networks.

---

## Practical Implementation

The orchestrator needs to run between dyads. In the current codebase:

1. After each `run_dyad_llm()` call, extract a knowledge summary from the transcript
2. Pass the updated summaries to the orchestrator
3. The orchestrator selects the next edge from `remaining_edges`
4. Repeat until all conversations have run

The schedule is no longer fixed upfront — it's built dynamically as the experiment runs.

```
Remaining edges: all intra + inter edges
agent_knowledge_estimate: {i: set() for i in range(9)}

while remaining_edges:
    next_dyad = orchestrator.choose(remaining_edges, agent_knowledge_estimate)
    transcript = run_dyad_llm(*next_dyad, ...)
    agent_knowledge_estimate = update_estimates(transcript, agent_knowledge_estimate)
    remaining_edges.remove(next_dyad)
```

This is a ~100 line change to `experiment.py`, a new `orchestrator.py` module, and a new `ScheduleName.ADAPTIVE` value in `schedule.py`.

---

## Why This Is the Right CS Angle

The study is fundamentally about information flow in networks. The CS lens is: *can we build a system that optimizes information flow without hardcoded rules?*

The adaptive orchestrator does this. It's:
- **Agentic**: an agent with a meta-task that reasons over the state of other agents
- **Algorithmic**: the greedy version has a clear information-theoretic justification
- **Measurable**: you can compare adaptive to both fixed baselines and explain the differences
- **Generalizable**: the orchestrator logic is independent of the specific task — it would work for any hidden profile problem on any graph

This makes the paper relevant to both the network science audience (does the causal story hold?) and the CS/AI audience (can systems learn to exploit network structure?).

The network science result says: *given* the right schedule, the conjunction works.
The CS result says: *an agent can learn the right schedule from the task structure.*

Together: the conjunction works, and it's learnable. That's a complete story.

---

## One Concern and How to Address It

**Concern**: The adaptive orchestrator has "god's eye view" access to agent knowledge summaries. Real agents don't have this — the orchestrator would need to observe only public information (the conversation transcripts) and infer knowledge states.

**Response**: This is a feature, not a bug. The paper can position this as an idealized upper bound — what would perfect scheduling look like? Then compare it to both fixed protocols and to a version where the orchestrator only sees partial information. The gap between perfect and partial-information scheduling tells you how much of the timing effect is recoverable from observable signals.

This frames it as a fundamental result about the limits of orchestration, not just an engineering exercise.
