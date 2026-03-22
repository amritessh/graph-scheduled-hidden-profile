# graph-scheduled-hidden-profile

LLM multi-agent experiments: **hidden-profile** information + **caveman community graph** + **round-based communication schedules** (within-clique first vs cross-community first).

## Setup

```bash
cd /Users/amriteshanand/multi-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[llm]"   # or pip install -e .  without LLM extras
```

## Quick check (no API)

```bash
python -m gshp.cli inspect
python -m gshp.cli dry-run --schedule within_first
python -m gshp.cli dry-run --schedule cross_first
```

**Topology default:** `full_clique_ring` — three (or `l`) full cliques plus one bridge per adjacent pair in a ring.  
`networkx.caveman_graph` in NetworkX 3.6 is **only isolated cliques** (no cross edges); use `--kind networkx_connected_caveman` or the NX docs if you need the library generator verbatim.

## Layout

- `gshp/graph/caveman.py` — build `caveman_graph(l, k)`, intra vs inter edges, connector nodes
- `gshp/schedule.py` — which edges are active in which round
- `gshp/runner.py` — one experiment run: rounds → dyads → (stub) dialogue → transcripts
- `gshp/cli.py` — CLI for dry-runs

Next: hidden-profile **task generator** + **vLLM/OpenAI-compatible** client plugged into `DyadicSession`.
