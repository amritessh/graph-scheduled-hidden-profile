# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (Python ≥ 3.11 required)
pip install -e ".[llm]"       # core + OpenAI-compatible client
pip install -e ".[llm,dev]"   # + pytest

# Sanity checks (no LLM calls)
python -m gshp.cli inspect
python -m gshp.cli dry-run --schedule within_first

# Stub run (no API required)
python -m gshp.cli run --stub --schedule within_first --condition hidden_profile

# Real run against local vLLM
python -m gshp.cli run \
  --model vllm:8000/Qwen/Qwen3-8B \
  --schedule cross_first \
  --condition hidden_profile \
  --tom-bridge \
  --group-deliberation

# Batch: 2×2×2 factorial
python -m gshp.cli batch --config examples/batch_qwen3_full.json
python -m gshp.cli batch --config examples/batch_qwen3_full.json --resume

# Re-analyze saved results
python -m gshp.cli analyze results/<run_dir>

# Tests
pytest tests/
pytest tests/test_schedule_invariants.py   # single file
```

## Architecture

This is a scientific experiment framework studying how **communication scheduling** and **information structure** affect collective decision-making in networked LLM agents.

### Core experimental setup

- **Topology**: 3-cluster caveman graph (3 agents × 3 clusters = 9 total). Clusters are fully connected internally; clusters connect in a ring via bridge agents (nodes 2, 5, 8).
- **Task**: Hidden profile hiring decision. Shared facts (visible to all) favor the wrong candidate (X). Cluster-unique and bridge-unique facts support the correct candidate (Y). The design forces inter-cluster communication to solve the task correctly.
- **Factorial design**: 2×2×2 — `schedule` × `condition` × `tom_bridge`.

### Two-phase schedule

`schedule.py` builds a `CommunicationRound` list. `ScheduleName` is either `WITHIN_FIRST` (intra → inter) or `CROSS_FIRST` (inter → intra). Each round contains a set of dyad edges. `matching_schedule.py` partitions edges within a phase into **graph matching layers** (no agent appears twice in a layer), which defines safe parallel execution boundaries.

### Per-dyad execution

`session.py:run_dyad_llm()` drives alternating dialogue: each turn, the current speaker's system prompt + transcript-so-far goes to the LLM. Prompts are built by `prompts.py:agent_system_prompt()`, which conditions on which facts the agent can see (`InformationCondition`) and optionally adds a Theory-of-Mind block for bridge agents.

### Experiment pipeline (`experiment.py`)

```
Task + Topology → build_two_phase_schedule()
  → for each dyad: run_dyad_llm() → update agent memory
  → query each agent for JSON decision
  → [optional] run_group_deliberation()
  → analyze_fact_transmission()   # DV2: disclosure / transmission / integration
  → write_run_bundle()            # full artifact capture
```

### LLM clients (`llm/`)

`make_llm_client(model_string)` selects backend from a single string:
- `vllm:8000/...` or `localhost:PORT/...` → local OpenAI-compatible server
- `gpt-4o-mini` / `openai/...` → OpenAI cloud (`OPENAI_API_KEY`)
- `openrouter/vendor/model` → OpenRouter (`OPENROUTER_API_KEY`)

`LoggingLLMClient` wraps any client transparently, recording every call (prompt, response, usage, latency) for `llm_calls.json`. `StubLLM` returns canned dialogue (no API calls) for testing.

### Batch processing (`batch.py`)

Reads a JSON config with a `grid` (keys → lists), takes the Cartesian product, and runs `concurrent_runs` experiments in parallel via `ThreadPoolExecutor`. Progress is written to `progress.json` after each run (enables `--resume`). Output: `index.csv` with one row per run.

### Fact transmission analysis (`fact_tracker.py`)

Uses an Aho-Corasick automaton (`aho_corasick.py`) to efficiently multi-pattern search all conversation text. Tracks three stages per fact: **disclosure** (mentioned anywhere), **transmission** (crossed a cluster boundary), **integration** (appeared in a final decision justification).

### Artifact bundle (`artifacts.py`)

Every run writes a results folder with: `config.json` (protocol + SHA-256 hash), `task.json`, `summary.json`, `metrics.json`, `llm_calls.json`, per-dyad `dyad_NNN.json`, `game_log.txt`, `fact_transmission.json`, optionally `deliberation.json`, and `run.json` (full `ExperimentRun` Pydantic model).

## Key modules

| File | Role |
|------|------|
| `gshp/cli.py` | Entry point; all subcommands |
| `gshp/experiment.py` | `run_hidden_profile_hiring()` — full pipeline |
| `gshp/session.py` | `run_dyad_llm()` — single dyad dialogue |
| `gshp/schedule.py` | `build_two_phase_schedule()` |
| `gshp/matching_schedule.py` | Partition edges into matching layers |
| `gshp/prompts.py` | Per-agent system prompts + ToM block |
| `gshp/task/hiring.py` | `HiringTaskSpec`, `InformationCondition`, default facts |
| `gshp/llm/openai_local.py` | `make_llm_client()` factory |
| `gshp/llm/logging_client.py` | `LoggingLLMClient` wrapper |
| `gshp/batch.py` | Factorial batch runner |
| `gshp/fact_tracker.py` | Aho-Corasick fact transmission analysis |
| `gshp/artifacts.py` | `write_run_bundle()` |
| `gshp/bridge_coder.py` | LLM judge for bridge communication quality (DV4, post-hoc) |

## Design notes

- **Schedule as data**: `CommunicationRound` list is built upfront, enabling dry-run and schedule variants without touching experiment logic.
- **Matching layers**: Computed via NetworkX max-weight-matching. Dyads in the same layer have no shared agents and can run concurrently. `parallel_dyad_layers=True` in batch config enables this.
- **JSON decisions**: Final agent choices are extracted from structured JSON. `parse_choice_json()` in `stub_client.py` has regex fallback.
- **`bridge_coder.py` is post-hoc**: The LLM judge for bridge communication quality (DV4) runs offline against saved artifacts, not during experiments.
- **Qwen3 `<think>` stripping**: The LLM client strips `<think>...</think>` blocks from Qwen3 reasoning model output before processing.
