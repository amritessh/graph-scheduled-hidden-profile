# graph-scheduled-hidden-profile

**LLM agents, hidden information, and a community-structured network** — with **who talks when** as a controlled, causal lever.

This repository implements the **software** for a multi-agent experiment: agents hold **shared** and **private** facts (hidden-profile paradigm), exchange messages only along **allowed edges** of a **community graph**, and do so in a **fixed schedule** that varies *within-community-first* vs *cross-community-first* communication. The goal is to study **when agents converge, whether they converge to the *correct* answer**, and how **information crosses community boundaries**.

---

## Table of contents

- [graph-scheduled-hidden-profile](#graph-scheduled-hidden-profile)
  - [Table of contents](#table-of-contents)
  - [Scientific background](#scientific-background)
    - [Hidden profile](#hidden-profile)
    - [Networks and communication *order*](#networks-and-communication-order)
    - [Community structure (“caveman” picture)](#community-structure-caveman-picture)
  - [What this experiment tests](#what-this-experiment-tests)
  - [Design (high level)](#design-high-level)
    - [Topology](#topology)
    - [Communication schedule (main manipulation)](#communication-schedule-main-manipulation)
    - [Task (conceptual)](#task-conceptual)
  - [Roadmap](#roadmap)
  - [Installation \& quick start](#installation--quick-start)
  - [CLI reference](#cli-reference)
  - [Project layout](#project-layout)
  - [Local LLM (vLLM)](#local-llm-vllm)
  - [References](#references)
  - [License / collaboration](#license--collaboration)

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

---

## Roadmap

**Done (v0):** hiring **task spec** (`HiringTaskSpec`, shared / cluster / bridge facts, `shared_only` vs `hidden_profile`), **LLM dyads** + **per-agent final JSON vote**, **`run` CLI** (stub or vLLM), basic **accuracy** in `run.notes`.

**Next:**

1. **Parameterized task generator** — Beyond the single default hiring instance; template slots and validation.  
2. **Batch runner** — Many runs (seeds × schedules × factorial cells); directory layout + `progress.json`.  
3. **Metrics** — Fact disclosure / transmission / integration (string or LLM-judge coding).  
4. **Optional** — Parallel dyad matchings per timestep; async + retries; PID / alignment metrics from the design doc.

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

**Full hiring run (stub LLM, no API):**

```bash
python -m gshp.cli run --stub --schedule within_first --condition hidden_profile
```

**Full run against local vLLM** (OpenAI-compatible server):

```bash
python -m gshp.cli run --model vllm:8000/YourModelName --schedule cross_first --out run.json
```

Use `--condition shared_only` for the control (everyone sees only the six shared facts).  
`--tom-bridge` adds the perspective-taking block for agents **2, 5, 8** (default bridge fact holders).

**Save a full results bundle** (recommended for real runs):

```bash
python -m gshp.cli run --stub --save-artifacts
# or
python -m gshp.cli run --model vllm:8000/MODEL --artifact-dir path/to/my_run
```

---

## Run artifacts (what gets saved)

When you pass **`--save-artifacts`** or **`--artifact-dir DIR`**, the run is wrapped in a logger and the following files are written (same *idea* as classic experiment folders: config + per-step JSON + prompt/API log + summary):

| File | Contents |
|------|-----------|
| **`config.json`** | Timestamp, topology summary, manifest (schedule, condition, model, …), task ids |
| **`task.json`** | Full `HiringTaskSpec` (all fact texts and assignments) for reproduction |
| **`summary.json`** | `run.notes` (accuracy, majority, …) + list of final agent votes |
| **`llm_calls.json`** | Every LLM call: `system`, `messages`, `response`, plus metadata (`kind`: `dyad` / `final_decision`, `speaker`, `turn`, `round_label`, …) |
| **`dyad_NNN_*.json`** | One transcript per scheduled dyad |
| **`run.json`** | Full `ExperimentRun` (all dyads + decisions) |
| **`game_log.txt`** | Short human-readable index of dyads and final choices |

`results/` is gitignored by default.

---

## CLI reference

```text
python -m gshp.cli inspect [--l N] [--k N] [--kind KIND]
python -m gshp.cli dry-run [--l N] [--k N] [--kind KIND] [--schedule within_first|cross_first] [--out PATH]
python -m gshp.cli run [--stub | --model SPEC] [--schedule ...] [--condition hidden_profile|shared_only]
                       [--tom-bridge] [--dyad-turns N] [--out PATH] [--save-artifacts] [--artifact-dir DIR]
```

- **`--kind`**: `full_clique_ring` (default) | `networkx_caveman` | `networkx_connected_caveman`  
- **`--schedule`**: `within_first` | `cross_first`

---

## Project layout

```text
gshp/
  graph/
    caveman.py      # Topology: communities, intra/inter edges, connectors
  task/
    hiring.py       # HiringTaskSpec + build_default_hiring_task()
  schedule.py       # Two-phase communication schedules
  runner.py         # Topology + schedule only (stub dyads; for scaffolding)
  experiment.py     # Full pipeline: task + LLM dyads + final decisions
  session.py        # run_dyad_stub, run_dyad_llm (alternating speakers)
  prompts.py        # Per-agent system prompts + optional ToM on 2,5,8
  types.py          # Transcripts, manifest, AgentDecision, ExperimentRun
  llm/
    openai_local.py   # vLLM / OpenAI-compatible sync client
    logging_client.py # record every complete() for llm_calls.json
    stub_client.py    # StubLLM + JSON choice parser (tests / CI)
  artifacts.py      # write config / summary / llm_calls / dyad files / game_log
  cli.py            # inspect, dry-run, run
```

---

## One `--model` string (local vs API)

The CLI matches the usual lab pattern: **you mostly change one argument** — `--model` — to switch backends. Optional **`--stub`** skips the network entirely.

| `--model` pattern | Backend | Auth |
|-------------------|---------|------|
| `vllm:PORT/model-id` | Local vLLM | Often `api_key` ignored (`dummy`) |
| `vllm/model-id` | vLLM at `$VLLM_BASE_URL` | same |
| `localhost:PORT/model-id` | Any OpenAI-compatible server on that port | optional |
| `gpt-4o-mini` (or `openai/gpt-4o-mini`) | OpenAI cloud | `OPENAI_API_KEY` |
| `openrouter/vendor/model-id` | OpenRouter | `OPENROUTER_API_KEY` |

**Experiment variant** is the combination of flags you already have: `--schedule`, `--condition`, `--tom-bridge`, `--kind`, etc. **Where results go** is `--artifact-dir` or `--save-artifacts` (not a separate “experiment name” registry unless you encode it in the path, e.g. `--artifact-dir results/bridge_first_hp`).

Example (Python):

```python
from gshp.llm import make_llm_client

local = make_llm_client("vllm:8000/Qwen/Qwen3-8B")
cloud = make_llm_client("gpt-4o-mini")
```

**Note:** Ollama’s native **`/api/generate`** API is not the same as `/v1/chat/completions`; use a compatible gateway or extend the router.

---

## References

- Momennejad, A. *et al.* Collective memory and social tie structure in networked communication. *Nat. Commun.* (2019). [DOI `10.1038/s41467-019-09452-y`](https://doi.org/10.1038/s41467-019-09452-y)  
- Hidden profile / group decision making: Stasser & Titus; subsequent hidden-profile literature in organizational and social psychology.  
- **NetworkX** caveman generators: [`caveman_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.caveman_graph.html), `connected_caveman_graph`.

---

## License / collaboration

Experiment design and priorities are set with the research collaborator; this README describes the **engineering** surface. Add a `LICENSE` and citation text when you publish code or a paper.
