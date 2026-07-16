# graph-scheduled-hidden-profile

**LLM agents, hidden information, and a community-structured network** — with **who talks when** as a controlled, causal lever.

A 2 × 2 × 2 factorial experiment using LLM agents in a 3-cluster, 3-agent-per-cluster caveman graph.
The three factors are conversation timing (cluster-first vs bridge-first), information structure
(shared-only vs hidden profile), and Theory of Mind at bridge nodes.

---

## Table of contents

- [Scientific background](#scientific-background)
- [What this experiment tests](#what-this-experiment-tests)
- [Design](#design)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Batch / factorial runs](#batch--factorial-runs)
- [Run artifacts](#run-artifacts)
- [Local LLM (vLLM)](#local-llm-vllm)
- [Project layout](#project-layout)
- [Theory layer](#theory-layer)
- [References](#references)

---

## Scientific background

**Hidden profile** (Stasser & Titus 1985): shared evidence favors the wrong answer; the correct
answer only emerges when unique private facts are pooled. Groups routinely fail to pool private
information and lock onto the shared-info attractor.

**Network structure + order** (Momennejad et al., *Nat. Commun.* 2019): in networked communication,
*when* cross-community links are used relative to within-community discussion affects collective
outcomes. This project brings that ordering idea into a hidden-profile setting with a verifiable
ground truth and a full factorial design.

---

## What this experiment tests

| Factor | Level 0 | Level 1 |
|--------|---------|---------|
| **Timing** | Cluster-first (within → bridge) | Bridge-first (bridge → within) |
| **Information structure** | Shared-only (control) | Hidden profile (unique info distributed) |
| **ToM at bridge nodes** | No ToM | ToM prompt on agents 2, 5, 8 |

**Four dependent variables:**

- **DV1** — Decision accuracy: proportion choosing the correct candidate (Y)
- **DV2** — Unique-fact utilization: disclosure / transmission / integration rates (Aho-Corasick)
- **DV3** — Convergence vs alignment + entropy-reduction information gain (hard elimination + Shapley decomposition)
- **DV4** — Bridge communication quality: relay / filter / translate coding (LLM judge)

---

## Design

### Topology

```
     CLUSTER 0          CLUSTER 1          CLUSTER 2
   0 ── 1 ── 2        3 ── 4 ── 5        6 ── 7 ── 8
          |                  |                  |
          2 ◄──────────────► 5 ◄──────────────► 8
       (bridge)           (bridge)           (bridge)
          └──────────────────────────────────────┘
                        (ring of bridges)
```

- Default: `l=3` clusters, `k=3` agents each → 9 agents total
- Bridge nodes: `{2, 5, 8}` — last agent of each clique, connected in a triangle
- Topology held constant across all conditions; only edge activation order varies

### Communication schedule

| Schedule | Phase 1 | Phase 2 |
|----------|---------|---------|
| `within_first` (cluster-first) | All intra-community dyads | All bridge dyads |
| `cross_first` (bridge-first) | Bridge dyads first | Intra-community dyads |

Each phase can be split into **matching layers** (`--parallel-dyads`): disjoint sets of dyads
with no shared agent, safe to run in parallel. See [docs/algorithms.md](docs/algorithms.md).

### Information distribution

- **Shared facts** (all 9 agents): 4 facts supporting the wrong candidate X + 1 weak fact each for Y and Z
- **Cluster-unique facts** (2 per cluster): critical facts supporting the correct candidate Y
- **Bridge-unique facts** (1 per bridge node): linking facts connecting two clusters' information

Shared-only condition: remove all unique and bridge facts. Every agent has the same 6 shared facts.

---

## Installation

**Requirements:** Python ≥ 3.11

```bash
git clone https://github.com/amritessh/graph-scheduled-hidden-profile.git
cd graph-scheduled-hidden-profile
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[llm]"            # core + OpenAI client for vLLM
pip install -e ".[llm,dev]"        # + pytest
```

---

## Quick start

**Sanity check (no GPU, no API):**

```bash
python -m gshp.cli inspect
python -m gshp.cli dry-run --schedule within_first
```

**Stub run (no LLM calls):**

```bash
python -m gshp.cli run --stub --schedule within_first --condition hidden_profile
```

**Single real run against local vLLM:**

```bash
python -m gshp.cli run \
  --model vllm:8000/Qwen/Qwen3-8B \
  --schedule cross_first \
  --condition hidden_profile \
  --tom-bridge \
  --group-deliberation
```

**Full 2×2×2 factorial batch (400 runs):**

```bash
# Simplest — no config file needed, uses built-in defaults
python -m gshp.cli batch --model vllm:8000/qwen3-30b --runs 50 --concurrent 10

# Dry run first
python -m gshp.cli batch --model vllm:8000/qwen3-30b --dry-run

# Resume after interruption (skips runs that already have summary.json)
python -m gshp.cli batch --model vllm:8000/qwen3-30b --resume

# Or with explicit config file
python -m gshp.cli batch --config examples/batch_qwen3_full.json
```

Results land in `results/{model_slug}_{YYYYMMDD_HHMMSS}/`.

**Post-experiment analysis (DV2 + DV4):**

```bash
# DV2: semantic fact detection across saved transcripts (runs on saved results)
python -m gshp.cli classify-facts results/<batch_dir> \
  --model vllm:8000/qwen3-30b --workers 10

# DV4: bridge communication quality coding
python -m gshp.cli code-bridges results/<batch_dir> \
  --model vllm:8000/qwen3-30b --force --workers 8 --max-tokens 1024
```

`code-bridges` runs the judge calls concurrently (`--workers`, default 8) and needs
a generous token budget (`--max-tokens`, default 1024) — a judge capped too low
will burn its budget on unstructured reasoning and return empty responses instead
of a classification. Use `--force` to re-code runs that already have `bridge_coding.json`
(e.g. after fixing the judge prompt or raising the token budget).

Both commands are resumable — skip runs that already have output files.

---

## CLI reference

```text
python -m gshp.cli inspect        [--l N] [--k N] [--kind KIND]
python -m gshp.cli dry-run        [--l N] [--k N] [--schedule S] [--parallel-dyads] [--out PATH]
python -m gshp.cli run            [--stub | --model SPEC]
                                  [--schedule within_first|cross_first]
                                  [--condition hidden_profile|shared_only]
                                  [--tom-bridge]
                                  [--dyad-turns N]          # turns per dyad (default 6)
                                  [--parallel-dyads]        # split phases into matching layers
                                  [--workers N]             # threads per matching layer
                                  [--group-deliberation]    # final group consensus round
                                  [--task-file PATH]        # use a generated task JSON
                                  [--verbose]               # print each agent turn as it arrives
                                  [--seed N]
                                  [--temperature F]
                                  [--max-tokens N]
                                  [--artifact-dir DIR]
                                  [--out PATH]
python -m gshp.cli batch          [--config PATH.json]      # optional; uses built-in defaults
                                  [--model SPEC]            # override model
                                  [--runs N]                # override runs_per_cell
                                  [--concurrent N]          # override concurrent_runs
                                  [--dry-run] [--resume]
python -m gshp.cli classify-facts BATCH_DIR                 # DV2: LLM semantic fact classifier
                                  --model SPEC
                                  [--workers N]
                                  [--force]
python -m gshp.cli code-bridges   BATCH_DIR                 # DV4: relay/filter/translate bridge coder
                                  --model SPEC
                                  [--force]
                                  [--workers N]             # concurrent judge calls per run (default 8)
                                  [--max-tokens N]          # judge completion budget (default 1024)
python -m gshp.cli generate-task  --domain "medical diagnosis"
                                  [--model SPEC | --stub]
                                  [--temperature F]
                                  [--options "A,B,C"]
                                  [--task-id ID]
                                  [--out PATH]
python -m gshp.cli analyze        RUN_DIR [--out PATH]
python -m gshp.cli analyze-batch  BATCH_DIR
                                  [--out-json PATH]
                                  [--out-csv PATH]
                                  [--out-report PATH]
python -m gshp.cli run-reasoning  [--stub | --model SPEC]  # single reasoning-task run
                                  [--problem ID] [--schedule S] [--deliberation]
python -m gshp.cli batch-reasoning [--config PATH.json]    # reasoning-task factorial batch
                                  [--model SPEC] [--runs N] [--concurrent N] [--resume]
```

**`--model` string format:**

| Pattern | Backend | Auth |
|---------|---------|------|
| `vllm:PORT/model-id` | Local vLLM | no key needed |
| `localhost:PORT/model-id` | Any OpenAI-compatible server | optional |
| `gpt-4o-mini` / `openai/gpt-4o-mini` | OpenAI cloud | `OPENAI_API_KEY` |
| `openrouter/vendor/model-id` | OpenRouter | `OPENROUTER_API_KEY` |

---

## Batch / factorial runs

A batch config is a JSON file with a `grid` object (keys → lists of values). The runner takes
the Cartesian product and runs `runs_per_cell` replicates per cell.

```json
{
  "base_dir": "results/qwen3_full_factorial",
  "model": "vllm:8000/Qwen/Qwen3-8B",
  "temperature": 0.7,
  "max_tokens": 512,
  "timeout": 120.0,
  "max_retries": 3,

  "runs_per_cell": 50,
  "seed_base": 0,
  "increment_seed_per_run": true,
  "concurrent_runs": 30,

  "dyad_turns": 6,
  "parallel_dyad_layers": true,
  "max_workers": 1,
  "group_deliberation": true,

  "topology": { "l": 3, "k": 3, "kind": "full_clique_ring" },

  "grid": {
    "schedule":   ["within_first", "cross_first"],
    "condition":  ["hidden_profile", "shared_only"],
    "tom_bridge": [false, true]
  }
}
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `base_dir` | Output root. A `{YYYYMMDD_HHMMSS}` subfolder is appended automatically |
| `concurrent_runs` | Number of experiment runs in parallel (each gets its own LLM client) |
| `group_deliberation` | Run a final group consensus round after individual decisions |
| `task_file` | Path to a generated task JSON (overrides the default hiring task) |
| `parallel_dyad_layers` | Split each phase into matching layers |
| `max_workers` | Threads per matching layer (requires `parallel_dyad_layers: true`) |

**Batch output layout:**

```
results/qwen3_full_factorial/
  20260331_214411/               ← timestamped run
    batch_config.json
    progress.json                ← live: completed / failed / skipped
    index.csv                    ← one row per run, all grid columns + accuracy
    condition=hidden_profile__schedule=cross_first__tom_bridge=1/
      run_001/  run_002/  ...    ← full artifact bundle per run
```

---

## Run artifacts

Every run (single or batch) writes the same full bundle — no lightweight mode.

| File | Contents |
|------|----------|
| `config.json` | Protocol, topology, manifest, protocol SHA-256, audit (git hash, versions) |
| `task.json` | Full task spec (all fact texts and assignments) for reproducibility |
| `summary.json` | Votes, accuracy, majority, unanimous_correct, deliberation outcome |
| `metrics.json` | LLM aggregate stats, fact mention rates, notes |
| `llm_calls.json` | Every LLM call: prompt, response, raw_response (pre-stripping), latency, usage |
| `run.json` | Full serialized `ExperimentRun` (all dyads + decisions) |
| `dyad_NNN_label.json` | One transcript per dyadic conversation |
| `fact_transmission.json` | DV2 (keyword): disclosure / transmission / integration rates by fact category |
| `fact_classification.json` | DV2 (semantic): LLM-classified disclosure / transmission / integration (written by `classify-facts`) |
| `bridge_coding.json` | DV4: relay/filter/translate counts + translate_rate per bridge agent (written by `code-bridges`) |
| `deliberation.json` | Group deliberation round (if `--group-deliberation`) |
| `game_log.txt` | Human-readable trace: dyad index, agents, round, final decisions |
| `error.txt` | Stack trace (failed runs only) |

**Batch-level files** (in the timestamped batch folder):

| File | Contents |
|------|----------|
| `index.csv` | One row per run: grid columns, accuracy, majority vote, group_accuracy, seed, status |
| `progress.json` | Live progress: completed / failed / skipped counts |
| `batch_config.json` | Resolved config used for this batch |

**Re-analyze a folder** (e.g. after changing heuristics):

```bash
python -m gshp.cli analyze results/hidden_profile_experiment_vllm_8000_Qwen_Qwen3-8B_20260331_214411
```

**Aggregate a full batch (condition-level report):**

```bash
python -m gshp.cli analyze-batch results/qwen3_full_factorial/vllm_8000_Qwen_Qwen3-8B_20260426_120000
```

This writes:
- `batch_analysis.json` (full aggregate)
- `condition_summary.csv` (one row per factorial cell)
- `report.md` (quick meeting-ready summary)
- `paper_tables.json` (key factorial contrasts + bootstrap CIs)
- `paper_report.md` (interpretation-oriented summary)

Convenience wrapper (auto-picks latest batch folder if omitted):

```bash
bash scripts/post_run_analysis.sh
# or:
bash scripts/post_run_analysis.sh results/qwen3_full_factorial/<batch_folder>
```

---

## Local LLM (vLLM)

```bash
# Basic
vllm serve Qwen/Qwen3-8B --port 8000 --api-key dummy

# With custom served name (use this name in --model)
docker run -d --gpus '"device=0"' --name vllm-qwen3 -p 8000:8000 \
  vllm/vllm-openai:latest Qwen/Qwen3-30B \
  --served-model-name qwen3-30b \
  --tensor-parallel-size 1

# Check what model id is being served
curl http://localhost:8000/v1/models | python3 -m json.tool
```

Then use `--model vllm:8000/<id>` where `<id>` matches the `id` field from the models endpoint.

**SSH tunnel for remote vLLM:**

```bash
ssh -L 8000:localhost:8000 user@remote-server
# then use --model vllm:8000/<id> locally
```

**Qwen3 note:** Qwen3 emits `<think>...</think>` reasoning blocks before responses.
The `LoggingLLMClient` automatically strips these before downstream processing
and preserves the raw output in `raw_response` for artifact analysis.

---

## Project layout

```
gshp/
  graph/
    caveman.py          # Topology: communities, intra/inter edges, connector nodes
  task/
    hiring.py           # HiringTaskSpec + build_default_hiring_task()
    generator.py        # generate_hidden_profile_task(): arbitrary domain → task JSON
  schedule.py           # Two-phase schedule builder
  matching_schedule.py  # Partition phase edges into matching layers (graph matchings)
  aho_corasick.py       # Multi-pattern substring search for fact tracking
  runner.py             # Stub runner (topology + schedule only, no LLM)
  experiment.py         # Full pipeline: task + LLM dyads + decisions + deliberation
  session.py            # run_dyad_llm: alternating-speaker dyad execution
  prompts.py            # Per-agent system prompts + ToM block for bridge nodes
  deliberation.py       # Group deliberation round after individual decisions
  fact_tracker.py       # DV2 keyword: disclosure / transmission / integration analysis
  fact_classifier.py    # DV2 semantic: LLM-based fact detection (liberal matching)
  bridge_coder.py       # DV4: relay/filter/translate LLM judge per bridge turn
  dv3.py                # DV3: convergence/alignment dissociation metrics (simple proxy)
  info_gain.py          # DV3: entropy-reduction + Shapley synergy over ground-truth
                        #   fact eliminations (keyword-detected, no LLM calls)
  types.py              # Pydantic models: ExperimentRun, DyadTranscript, AgentDecision, …
  llm/
    openai_local.py     # vLLM / OpenAI-compatible sync client + clone()
    logging_client.py   # Wraps any client; strips <think> blocks; records raw_response
    stub_client.py      # StubLLM for tests / dry runs
  artifacts.py          # Write full artifact bundle
  batch.py              # Factorial batch runner (concurrent_runs, progress.json, index.csv)
  batch_errors.py       # Map exceptions → error_type
  protocol.py           # Canonical protocol dict + SHA-256
  audit.py              # Git head, package versions, prompt_template_id
  metrics.py            # LLM call stats, fact mention heuristic
  analyze_run.py        # Offline metrics recomputation from artifact folder
  batch_reasoning.py    # Factorial batch runner for the reasoning contrast task
  experiment_reasoning.py  # run_reasoning_experiment(): pipeline for the reasoning task
  prompts_reasoning.py  # Per-agent prompts for the reasoning task (no info asymmetry)
  task/
    reasoning.py         # ReasoningTaskSpec: 10 math/probability problems + answer checking
  cli.py                # All subcommands: inspect, dry-run, run, batch, classify-facts,
                        #   code-bridges, generate-task, analyze, analyze-batch,
                        #   run-reasoning, batch-reasoning
docs/
  experiment_overview.md    # Plain-language experiment explanation
  algorithms.md             # Matching layers + parallel dyad algorithm
  theory_token_spread.md    # Graph-theory layer: scheduled token spread on ring-of-cliques
examples/
  batch_qwen3_full.json     # 2×2×2 factorial, 50 runs/cell, 30 concurrent
  batch_reasoning_pilot.json  # Reasoning-task contrast batch config
  batch_stub.json           # Same grid with stub LLM (testing)
  batch_stub_tiny.json      # Single cell, 2 runs (CI / quick check)
tests/
  test_schedule_invariants.py   # Schedule edge types + protocol hash stability
  test_matching_schedule.py     # Matching layers + expand_schedule invariants
  test_aho_corasick.py          # AC automaton + fact_mention_rates
```

---

## Theory layer

The experiment is an instance of **scheduled token spread on a temporal graph** — a ring of
cliques under a two-phase edge activation constraint. A companion theory result proves the
round-complexity bounds for this topology: minimum matching rounds for a unique fact to reach
all agents under intra-first vs inter-first scheduling.

This gives the empirical results a structural baseline: the accuracy gap between conditions
should track the theoretical round-complexity gap. See [docs/theory_token_spread.md](docs/theory_token_spread.md).

---

## References

- Momennejad *et al.* Collective memory and social tie structure in networked communication. *Nat. Commun.* 2019. [DOI 10.1038/s41467-019-09452-y](https://doi.org/10.1038/s41467-019-09452-y)
- Stasser & Titus. Pooling of unshared information in group decision making. *JPSP* 1985.
- Partial information decomposition for collective intelligence *(see references for the specific method)*

---

## License / collaboration

Add a `LICENSE` and citation when publishing code or paper.
