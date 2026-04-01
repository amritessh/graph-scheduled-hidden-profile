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
    - [Algorithms (parallel layers & fact matching)](docs/algorithms.md)
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

Within each phase, dyads are executed **sequentially** in code. Optional **`--parallel-dyads`** (or batch field **`parallel_dyad_layers`**) **splits** each phase into **matching layers**: every layer is a set of dyads with **no shared agent**, i.e. a graph **matching**—the groups you could run in parallel without double-booking anyone. See **[docs/algorithms.md](docs/algorithms.md)** for the algorithm and protocol details.

### Task (conceptual)

Each run will attach a **task instance**: short scenario, **shared facts**, **private facts per agent/cluster**, and a **correct discrete choice** (e.g. hire candidate **Y**). Agents only see their own information plus what others **choose to say** in scheduled conversations.

---

## Roadmap

**Done (v0):** hiring **task spec** (`HiringTaskSpec`, shared / cluster / bridge facts, `shared_only` vs `hidden_profile`), **LLM dyads** + **per-agent final JSON vote**, **`run` CLI** (stub or vLLM), basic **accuracy** in `run.notes`.

**Next:**

1. **Parameterized task generator** — Beyond the single default hiring instance; template slots and validation.  
2. **Batch runner** — Many runs (seeds × schedules × factorial cells); directory layout + `progress.json`. *(Implemented; see below.)*  
3. **Metrics** — Fact disclosure uses **Aho–Corasick** multi-pattern literals in `metrics.py`; LLM-judge / calibration still optional.  
4. **Optional** — Actually **parallel matching layers** are implemented (`--parallel-dyads`); async concurrent dyads, richer retries, PID / alignment metrics remain future work.

---

## Installation & quick start

**Requirements:** Python ≥ 3.11

```bash
git clone <your-remote> graph-scheduled-hidden-profile
cd graph-scheduled-hidden-profile
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[llm]"     # core + OpenAI client for vLLM; use pip install -e . for graph-only
pip install -e ".[llm,dev]" # + pytest for invariant tests
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
python -m gshp.cli run --model vllm:8000/YourModelName --schedule cross_first
```

Every **`run`** always writes a **full artifact folder** (default `results/run_YYYYMMDD_HHMMSS`, or **`--artifact-dir DIR`**). Optional **`--out extra.json`** also copies the combined `run.json` to another path.

Use `--condition shared_only` for the control (everyone sees only the six shared facts).  
`--tom-bridge` adds the perspective-taking block for agents **2, 5, 8** (default bridge fact holders).

---

## Run artifacts (always saved)

Each **`run`** produces the same style of folder as typical lab code: config + per-step JSON + **every** prompt/response log + summary. There is **no** “lightweight” mode—everything is kept for analysis.

| File | Contents |
|------|-----------|
| **`config.json`** | **`protocol`** (formal instrument: edges, schedule, `dyad_turns`, condition, ToM flag, **`parallel_dyad_layers`**) + **`protocol_sha256`**, **`audit`** (git commit, package versions, `prompt_template_id`), topology, manifest |
| **`task.json`** | Full `HiringTaskSpec` (all fact texts and assignments) for reproduction |
| **`summary.json`** | Votes, `run.notes`, **`protocol_sha256`**, **`llm_aggregate`** (call count, sum latency, token totals when API reports usage), coarse **fact-mention** counts |
| **`metrics.json`** | Same-style analysis blob for offline tools (`analyze` command refreshes it) |
| **`llm_calls.json`** | Every LLM call: `latency_ms`, optional top-level **`usage`**, `system`, `messages`, `response`, metadata (`kind`, `speaker`, `turn`, …), optional **`openai_completion`** |
| **`dyad_NNN_*.json`** | One transcript per scheduled dyad |
| **`run.json`** | Full `ExperimentRun` (all dyads + decisions) |
| **`game_log.txt`** | Short human-readable index of dyads and final choices |

`results/` is gitignored by default.

**Every LLM turn** is one row in **`llm_calls.json`**. When the backend is **`OpenAICompatibleChat`**, rows include **`openai_completion`** (SDK dump) and extracted **`usage`** when present. **Stub** backends omit API-shaped fields.

**Re-analyze** a folder (e.g. after changing heuristics):

```bash
python -m gshp.cli analyze results/run_YYYYMMDD_HHMMSS
```

---

## Batch / factorial runs

JSON config → Cartesian product of **`grid`** keys → one folder per **cell** × **`runs_per_cell`**. Each **`run_NNN/`** gets the **same full bundle** as a single CLI **`run`** (full capture always on).

```bash
python -m gshp.cli batch --config examples/batch_stub_tiny.json
python -m gshp.cli batch --config examples/batch_stub.json --dry-run
python -m gshp.cli batch --config examples/batch_stub_tiny.json --resume   # skip runs that already have summary.json
```

**Batch folder layout:**

| Path | Purpose |
|------|---------|
| **`batch_config.json`** | Copy of the input config + metadata |
| **`progress.json`** | `completed` / `failed` (with **`error_type`**: timeout, rate_limit, api_status_*, …) / **`skipped`** (when using **`--resume`**) |
| **`index.csv`** | One row per run: `cell_id`, grid columns, `run_dir`, `status`, `accuracy`, `majority`, … |
| **`<cell_id>/run_001/`** | Full per-run bundle (`config.json`, `task.json`, `llm_calls.json`, `dyad_*.json`, …) |

**Batch JSON fields (common):**

- **`base_dir`** — output root  
- **`grid`** — object whose values are **lists** (e.g. `schedule`, `condition`, `tom_bridge`)  
- **`runs_per_cell`**, **`seed_base`**, **`increment_seed_per_run`**  
- **`stub`** / **`stub_final`** or **`model`** (+ optional **`temperature`**, **`max_tokens`**, **`timeout`**, **`max_retries`**)  
- **`topology`**: `{ "l", "k", "kind" }`  
- **`dyad_turns`**  
- **`parallel_dyad_layers`** — optional bool (default `false`); per-cell override via a **`grid`** key of the same name if you factorial it

See **`examples/batch_stub.json`** (small factorial) and **`examples/batch_stub_tiny.json`** (one cell).

---

## CLI reference

```text
python -m gshp.cli inspect [--l N] [--k N] [--kind KIND]
python -m gshp.cli dry-run [--l N] [--k N] [--kind KIND] [--schedule within_first|cross_first] [--parallel-dyads] [--out PATH]
python -m gshp.cli run [--stub | --model SPEC] [--schedule ...] [--condition hidden_profile|shared_only]
                       [--tom-bridge] [--parallel-dyads] [--dyad-turns N] [--out PATH] [--artifact-dir DIR]
python -m gshp.cli batch --config PATH.json [--dry-run] [--resume]
python -m gshp.cli analyze RUN_DIR [--out PATH]
```

- **`--kind`**: `full_clique_ring` (default) | `networkx_caveman` | `networkx_connected_caveman`  
- **`--schedule`**: `within_first` | `cross_first`  
- **`--parallel-dyads`**: expand each phase into **matching layers** (see [docs/algorithms.md](docs/algorithms.md))

---

## Project layout

```text
gshp/
  graph/
    caveman.py      # Topology: communities, intra/inter edges, connectors
  task/
    hiring.py       # HiringTaskSpec + build_default_hiring_task()
  schedule.py       # Two-phase schedules + expand_schedule_parallel_matchings
  matching_schedule.py  # Partition phase edges into matching layers (NetworkX)
  aho_corasick.py   # Multi-pattern substring search for fact metrics
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
  batch.py          # factorial batch from JSON (progress.json, index.csv)
  batch_errors.py   # map exceptions → error_type for progress.json
  protocol.py       # canonical protocol dict + SHA-256
  audit.py          # git head, package versions, prompt_template_id
  metrics.py        # aggregate latency/tokens; fact-mention heuristic
  analyze_run.py    # offline metrics from artifacts
  cli.py            # inspect, dry-run, run, batch, analyze
tests/
  test_schedule_invariants.py  # schedule edge-types + protocol hash stability
  test_matching_schedule.py    # matching layers + expand_schedule invariants
  test_aho_corasick.py          # AC + fact_mention_rates vs naive substring
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

**Experiment variant** is the combination of flags you already have: `--schedule`, `--condition`, `--tom-bridge`, `--kind`, etc. **Where results go** defaults to **`results/run_<timestamp>`** or set **`--artifact-dir`** (e.g. `results/bridge_first_hp`).

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
