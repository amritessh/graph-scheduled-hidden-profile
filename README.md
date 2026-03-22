# graph-scheduled-hidden-profile

**LLM agents, hidden information, and a community-structured network** — with **who talks when** as a controlled, causal lever.

This repository implements the **software** for a multi-agent experiment: agents hold **shared** and **private** facts (hidden-profile paradigm), exchange messages only along **allowed edges** of a **community graph**, and do so in a **fixed schedule** that varies *within-community-first* vs *cross-community-first* communication. The goal is to study **when agents converge, whether they converge to the *correct* answer**, and how **information crosses community boundaries**.

---

## Table of contents

- [Scientific background](#scientific-background)
- [What this experiment tests](#what-this-experiment-tests)
- [Design (high level)](#design-high-level)
- [What is implemented today](#what-is-implemented-today)
- [Roadmap](#roadmap)
- [Installation & quick start](#installation--quick-start)
- [CLI reference](#cli-reference)
- [Project layout](#project-layout)
- [Local LLM (vLLM)](#local-llm-vllm)
- [References](#references)

---

## Scientific background

### Hidden profile

In a **hidden-profile** task, some evidence is **known to everyone** and other evidence is **distributed** across people. Often the **shared** evidence favors a **wrong** decision, while the **correct** decision only appears once **unique** pieces are combined. Groups frequently **fail to pool** private information and settle on the shared-information attractor — a classic finding in group decision research (e.g. Stasser & Titus–style paradigms).

**Why use it with LLM agents?** It gives a **verifiable ground truth**, asymmetric information, and a clear **failure mode** (consensus without accuracy) that we can trace in **transcripts**.

### Networks and communication *order*

Work on **humans** in networked communication (e.g. Momennejad *et al.*, *Nature Communications* 2019) shows that **structure** and **the order in which ties are used** can affect collective outcomes (e.g. what is remembered or reinforced). This project brings that **ordering idea** into a **hidden-profile** setting on a **small community graph**: *when* cross-community links are used relative to within-community discussion is a **causal** manipulation implemented as an explicit **schedule** in code.

### Community structure (“caveman” picture)

We use a graph of **dense within-community ties** and **sparse between-community ties** — the usual “caves + bridges” picture. Agents are partitioned into **cliques** (communities). **Intra-community** edges support local discussion; **inter-community** edges are the bottleneck through which **novel** facts must pass to reach other groups.

---

## What this experiment tests

At the broadest level:

| Question | Intuition |
|----------|-----------|
| Does **schedule** matter? | If cross-community dyads run **before** heavy within-clique discussion, unique information may stay **salient** longer; if within-clique runs first, groups may **entrench** on shared evidence before bridges activate. |
| Does **information structure** matter? | **Shared-only** vs **hidden profile** separates “pure coordination” from “integration of distributed evidence.” |
| (Planned) Do **bottleneck prompts** matter? | Interventions at **connector** nodes (e.g. theory-of-mind–style instructions) may change *how* information is relayed or framed across communities. |

The full factorial and dependent variables (accuracy, unique-fact disclosure/transmission, convergence vs alignment, etc.) are specified in the project design doc shared by the PI; **this repo’s job** is to make those manipulations **reproducible** and **loggable**.

---

## Design (high level)

### Topology

- **`l`** communities (cliques), each of **`k`** agents → **`l × k`** agents total. Default narrative: **`l = 3`, `k = 3`** → 9 agents.
- **Default builder: `full_clique_ring`**  
  - Every community is a **complete graph** \(K_k\) (all within-group dyads allowed).  
  - One **bridge** edge connects consecutive communities in a **ring** (last node of clique \(i\) ↔ first node of next clique, with wrap).  
- **Connector nodes** are endpoints of **inter-community** edges — natural targets for bottleneck-only prompts in later conditions.

> **Note (NetworkX):** In recent NetworkX (e.g. 3.6), `networkx.caveman_graph(l, k)` produces **`l` disconnected cliques** with **no** inter-clique edges. For experiments that need **bridges**, use our default **`full_clique_ring`**, or `--kind networkx_connected_caveman` if you want the library’s connected caveman variant (different wiring than full cliques).

### Communication schedule (main manipulation)

Two-phase schedules (Momennejad-style **ordering**):

| Schedule | Phase 1 | Phase 2 |
|----------|---------|---------|
| **`within_first`** | All **intra-community** dyads (one round listing every such edge) | All **inter-community** (bridge) dyads |
| **`cross_first`** | Inter-community dyads first | Intra-community dyads |

Within each phase, dyads are executed **sequentially** in a fixed order (see [Roadmap](#roadmap) for optional parallel matchings).

### Task (conceptual)

Each run will attach a **task instance**: short scenario, **shared facts**, **private facts per agent/cluster**, and a **correct discrete choice** (e.g. hire candidate **Y**). Agents only see their own information plus what others **choose to say** in scheduled conversations.

*Implementation of the generator and LLM dyads is in progress; the runner already walks the graph and schedule.*

---

## What is implemented today

| Component | Status |
|-----------|--------|
| **Community graph** | `CavemanTopology`: `full_clique_ring`, optional NX `caveman` / `connected_caveman` |
| **Intra vs inter edges** | Computed from community partition |
| **Connector nodes** | Identified for bridge endpoints |
| **Schedules** | `within_first`, `cross_first` |
| **Run orchestration** | Rounds → dyads → stub dialogue → JSON-serializable `ExperimentRun` |
| **CLI** | `inspect`, `dry-run` |
| **vLLM client (sync)** | `OpenAICompatibleChat` — OpenAI-compatible `/v1/chat/completions` |
| **Hidden-profile story generator** | Not yet |
| **Real LLM dyadic dialogue** | Not yet (stub only) |
| **Final vote / scoring** | Not yet |

---

## Roadmap

1. **Hidden-profile task generator** — Parameterized facts, who knows what, ground-truth label, optional “shared-only” ablation.  
2. **Dyadic LLM sessions** — System + private context per agent, alternating turns, transcripts in `DyadTranscript`.  
3. **Final decision step** — Structured output (choice + justification) per agent.  
4. **Batch runner** — Seeds × schedules × conditions; artifacts per run.  
5. **Metrics** — Accuracy, fact mention / transmission heuristics, convergence.  
6. **Optional** — ToM (or similar) prompts **only on connector nodes**; parallel dyad batches via graph matchings; async client + retries (patterns from `AI-GBS`).

---

## Installation & quick start

**Requirements:** Python ≥ 3.11

```bash
git clone <your-remote> graph-scheduled-hidden-profile
cd graph-scheduled-hidden-profile
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[llm]"     # core + OpenAI client for vLLM; use pip install -e . for graph-only
```

**Sanity check (no GPU / no API):**

```bash
python -m gshp.cli inspect
python -m gshp.cli dry-run --schedule within_first
python -m gshp.cli dry-run --schedule cross_first --out run.json
```

---

## CLI reference

```text
python -m gshp.cli inspect [--l N] [--k N] [--kind KIND]
python -m gshp.cli dry-run [--l N] [--k N] [--kind KIND] [--schedule within_first|cross_first] [--out PATH]
```

- **`--kind`**: `full_clique_ring` (default) | `networkx_caveman` | `networkx_connected_caveman`  
- **`--schedule`**: `within_first` | `cross_first`

---

## Project layout

```text
gshp/
  graph/
    caveman.py      # Topology: communities, intra/inter edges, connectors
  schedule.py       # Two-phase communication schedules
  runner.py         # One experiment run over a topology + schedule
  session.py        # Dyadic sessions (stub; LLM hook pending)
  types.py          # Pydantic models (transcripts, run manifest)
  llm/
    openai_local.py # vLLM / OpenAI-compatible sync client
  cli.py            # inspect + dry-run
```

---

## Local LLM (vLLM)

vLLM exposes an **OpenAI-compatible** HTTP API. This repo uses the same **model string conventions** as the related project **`AI-GBS`** (`llm_run.py`):

| Pattern | Behavior |
|---------|----------|
| `vllm:PORT/model-id` | `http://127.0.0.1:PORT/v1` + chat completions |
| `vllm/model-id` | `$VLLM_BASE_URL` (default `http://127.0.0.1:8000/v1`) |
| `localhost:PORT/model-id` | Same as vLLM on that port |

Example:

```python
from gshp.llm import OpenAICompatibleChat

client = OpenAICompatibleChat.from_model_spec("vllm:8000/Qwen/Qwen3-8B")
reply = client.complete("You are a helpful assistant.", [{"role": "user", "content": "Hello."}])
```

For **async**, **retries**, **OpenRouter**, or **Ollama’s `/api/generate`**, reuse or adapt `~/AI-GBS/llm_run.py`.

---

## References

- Momennejad, A. *et al.* Collective memory and social tie structure in networked communication. *Nat. Commun.* (2019). [DOI `10.1038/s41467-019-09452-y`](https://doi.org/10.1038/s41467-019-09452-y)  
- Hidden profile / group decision making: Stasser & Titus; subsequent hidden-profile literature in organizational and social psychology.  
- **NetworkX** caveman generators: [`caveman_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.caveman_graph.html), `connected_caveman_graph`.

---

## License / collaboration

Experiment design and priorities are set with the research collaborator; this README describes the **engineering** surface. Add a `LICENSE` and citation text when you publish code or a paper.
