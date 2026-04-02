"""CLI: inspect topology, dry-run schedules, full hiring experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from gshp.analyze_run import analyze_run_dir, write_metrics_json
from gshp.artifacts import write_run_bundle
from gshp.experiment import run_hidden_profile_hiring
from gshp.graph.caveman import CavemanTopology
from gshp.llm.logging_client import LoggingLLMClient
from gshp.llm.openai_local import make_llm_client
from gshp.llm.stub_client import StubLLM
from gshp.runner import run_experiment
from gshp.schedule import (
    ScheduleName,
    build_two_phase_schedule,
    expand_schedule_parallel_matchings,
)
from gshp.batch import run_batch_from_config
from gshp.task.hiring import InformationCondition, build_default_hiring_task
from gshp.task.generator import generate_hidden_profile_task, save_task, load_task, HiddenProfileTaskSpec


def cmd_inspect(args: argparse.Namespace) -> None:
    topo = CavemanTopology.build(l=args.l, k=args.k, kind=args.kind)
    print(f"Kind: {topo.kind}")
    print(f"Nodes: {topo.nodes}")
    print(f"Communities ({topo.l} × {topo.k}): {[sorted(c) for c in topo.communities]}")
    print(f"Intra edges ({len(topo.intra_edges)}): {sorted(topo.intra_edges)}")
    print(f"Inter edges ({len(topo.inter_edges)}): {sorted(topo.inter_edges)}")
    print(f"Connector nodes: {sorted(topo.connector_nodes)}")


def cmd_dry_run(args: argparse.Namespace) -> None:
    topo = CavemanTopology.build(l=args.l, k=args.k, kind=args.kind)
    sched = ScheduleName(args.schedule)
    rounds = build_two_phase_schedule(topo, sched)
    if args.parallel_dyads:
        rounds = expand_schedule_parallel_matchings(rounds)
    print(f"Schedule: {sched.value}")
    for r in rounds:
        layer = f" L{r.sub_index}" if r.sub_index else ""
        print(f"  Round {r.index} [{r.label}]{layer}: {len(r.edges)} dyads")
    run = run_experiment(topo, sched, parallel_dyad_layers=args.parallel_dyads)
    out = Path(args.out) if args.out else None
    payload = run.model_dump(mode="json")
    if out:
        out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out}")
    else:
        print(json.dumps(payload, indent=2)[:2000] + ("..." if len(json.dumps(payload)) > 2000 else ""))


def cmd_generate_task(args: argparse.Namespace) -> None:
    """Generate an arbitrary hidden profile task and save it to JSON."""
    if args.stub:
        from gshp.llm.stub_client import StubLLM as _Stub
        client = _Stub()
    else:
        if not args.model:
            raise SystemExit("Provide --model or --stub for task generation.")
        client = make_llm_client(args.model, temperature=args.temperature)

    print(f"Generating hidden profile task: domain='{args.domain}' ...")
    task = generate_hidden_profile_task(
        args.domain,
        client,
        options=args.options.split(",") if args.options else None,
        task_id=args.task_id or None,
    )
    out = args.out or f"task_{task.task_id}.json"
    save_task(task, out)
    print(f"Saved: {out}")
    print(f"  Scenario: {task.scenario[:120]}...")
    print(f"  Options: {task.options}  correct={task.correct_option}  attractor={task.attractor_option}")
    print(f"  Facts: {len(task.facts)} total ({len(task.shared_fact_ids)} shared, "
          f"{sum(len(v) for v in task.cluster_fact_ids.values())} cluster, "
          f"{len(task.bridge_agent_fact_ids)} bridge)")


def _model_slug(model: str) -> str:
    """Convert model string to filesystem-safe slug (mirrors AI-GBS convention)."""
    import re
    slug = model.replace("://", "_").replace("/", "_").replace(":", "_")
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", slug).strip("_")[:80] or "model"


def cmd_run(args: argparse.Namespace) -> None:
    topo = CavemanTopology.build(l=args.l, k=args.k, kind=args.kind)

    # Load task: explicit file > default hiring task
    if args.task_file:
        task = load_task(args.task_file)
        print(f"Loaded task from {args.task_file}: domain='{task.domain}'")
    else:
        task = build_default_hiring_task()

    cond = InformationCondition(args.condition)

    if args.stub:
        base_client: StubLLM | object = StubLLM(final_choice=args.stub_final)
        model_label = f"stub:{args.stub_final}"
    else:
        if not args.model:
            raise SystemExit(
                "Provide --model (e.g. vllm:8000/MODEL, localhost:8000/MODEL, gpt-4o-mini, openrouter/org/model) "
                "or use --stub"
            )
        try:
            base_client = make_llm_client(
                args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except ValueError as e:
            raise SystemExit(str(e)) from e
        model_label = args.model

    if args.artifact_dir:
        artifact_path = Path(args.artifact_dir)
    else:
        slug = _model_slug(model_label)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_path = Path("results") / f"hidden_profile_experiment_{slug}_{ts}"

    client: LoggingLLMClient = LoggingLLMClient(base_client, capture_raw_completion=True)

    run = run_hidden_profile_hiring(
        topo,
        args.schedule,
        task,
        client,
        condition=cond,
        tom_bridge=args.tom_bridge,
        dyad_turns=args.dyad_turns,
        model_label=model_label,
        seed=args.seed,
        parallel_dyad_layers=args.parallel_dyads,
        max_workers=args.workers,
        group_deliberation=args.group_deliberation,
        verbose=args.verbose,
    )

    out = Path(args.out) if args.out else None
    payload = run.model_dump(mode="json")
    if out:
        out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out}")

    root = write_run_bundle(
        artifact_path,
        run=run,
        task=task,
        topo=topo,
        llm_calls=client.calls,
        dyad_turns=args.dyad_turns,
        tom_bridge=args.tom_bridge,
        extra_config={"capture_raw_completion": True},
    )
    artifacts = "config, task, summary, metrics.json, fact_transmission.json, llm_calls, dyad_*.json, game_log.txt, run.json"
    if run.deliberation is not None:
        artifacts += ", deliberation.json"
    print(f"Artifact bundle: {root} ({artifacts})")

    print("--- summary ---")
    print(f"accuracy (agent-level): {run.notes.get('accuracy_agent_level')}")
    print(f"majority: {run.notes.get('majority_vote')} (correct={task.correct_candidate})")
    print(f"unanimous correct: {run.notes.get('unanimous_correct')}")
    for d in run.final_decisions:
        tail = d.justification[:80] + ("..." if len(d.justification) > 80 else "")
        print(f"  Agent {d.agent_id}: {d.choice} — {tail}")
    if run.deliberation is not None:
        print(f"--- group deliberation ---")
        print(f"group consensus: {run.notes.get('group_consensus')}  accuracy: {run.notes.get('group_accuracy')}")


def cmd_analyze(args: argparse.Namespace) -> None:
    p = Path(args.run_dir)
    outp = Path(args.out) if args.out else None
    written = write_metrics_json(p, path=outp)
    data = analyze_run_dir(p)
    print(f"Wrote {written}")
    txt = json.dumps(data, indent=2)
    print(txt[:4000] + ("..." if len(txt) > 4000 else ""))


def cmd_batch(args: argparse.Namespace) -> None:
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.runs is not None:
        overrides["runs_per_cell"] = args.runs
    if args.concurrent is not None:
        overrides["concurrent_runs"] = args.concurrent
    config_path = args.config or None
    root = run_batch_from_config(config_path, dry_run=args.dry_run, resume=args.resume, overrides=overrides)
    print(f"Batch base_dir: {root}")
    if args.dry_run:
        print("(dry run: no LLM calls, index.csv + progress show planned runs)")


def main() -> None:
    p = argparse.ArgumentParser(description="graph-scheduled-hidden-profile")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="Print caveman topology")
    pi.add_argument("--l", type=int, default=3, help="number of cliques")
    pi.add_argument("--k", type=int, default=3, help="clique size")
    pi.add_argument(
        "--kind",
        choices=["full_clique_ring", "networkx_caveman", "networkx_connected_caveman"],
        default="full_clique_ring",
        help="full_clique_ring = dense cliques + ring bridges (default); nx caveman = isolated cliques in NX 3.6",
    )
    pi.set_defaults(func=cmd_inspect)

    pr = sub.add_parser("dry-run", help="Run stub dyads through schedule")
    pr.add_argument("--l", type=int, default=3)
    pr.add_argument("--k", type=int, default=3)
    pr.add_argument(
        "--kind",
        choices=["full_clique_ring", "networkx_caveman", "networkx_connected_caveman"],
        default="full_clique_ring",
    )
    pr.add_argument(
        "--schedule",
        choices=[s.value for s in ScheduleName],
        default=ScheduleName.WITHIN_FIRST.value,
    )
    pr.add_argument("--out", type=str, default="", help="optional JSON path")
    pr.add_argument(
        "--parallel-dyads",
        action="store_true",
        help="Split each phase into matching layers (parallel-safe dyad groups); see docs/algorithms.md",
    )
    pr.set_defaults(func=cmd_dry_run)

    px = sub.add_parser("run", help="Full hiring hidden-profile run (LLM or stub)")
    px.add_argument("--l", type=int, default=3)
    px.add_argument("--k", type=int, default=3)
    px.add_argument(
        "--kind",
        choices=["full_clique_ring", "networkx_caveman", "networkx_connected_caveman"],
        default="full_clique_ring",
    )
    px.add_argument(
        "--schedule",
        choices=[s.value for s in ScheduleName],
        default=ScheduleName.WITHIN_FIRST.value,
    )
    px.add_argument(
        "--condition",
        choices=[c.value for c in InformationCondition],
        default=InformationCondition.HIDDEN_PROFILE.value,
    )
    px.add_argument("--tom-bridge", action="store_true", help="ToM-style prompt on agents 2,5,8")
    px.add_argument("--dyad-turns", type=int, default=6)
    px.add_argument("--stub", action="store_true", help="Use stub LLM (no API)")
    px.add_argument(
        "--stub-final",
        choices=["X", "Y", "Z"],
        default="X",
        help="Stub final JSON choice when --stub",
    )
    px.add_argument(
        "--model",
        type=str,
        default="",
        help=(
            "One string selects the backend: vllm:PORT/id, localhost:PORT/id (local); "
            "gpt-4o-mini or openai/gpt-4o-mini (OpenAI API key); openrouter/vendor/model (OpenRouter key)"
        ),
    )
    px.add_argument("--temperature", type=float, default=0.0)
    px.add_argument("--max-tokens", type=int, default=None)
    px.add_argument("--seed", type=int, default=None)
    px.add_argument(
        "--parallel-dyads",
        action="store_true",
        help="Split each phase into matching layers (parallel-safe dyad groups); see docs/algorithms.md",
    )
    px.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel threads for dyads within a matching layer. "
            "Requires --parallel-dyads. Each thread gets a cloned inner client. "
            "Use up to the number of independent dyads per layer (typically l=3 → 1 inter, "
            "up to k*(k-1)/2 intra per layer)."
        ),
    )
    px.add_argument(
        "--group-deliberation",
        action="store_true",
        help=(
            "After individual decisions, run a group deliberation round: all agents see "
            "the full panel's choices and provide a final group recommendation. "
            "Writes deliberation.json to the artifact bundle."
        ),
    )
    px.add_argument(
        "--task-file",
        type=str,
        default="",
        help="Path to a generated task JSON (from 'generate-task'). Overrides the default hiring task.",
    )
    px.add_argument("--verbose", action="store_true", help="print each agent turn as it arrives")
    px.add_argument("--out", type=str, default="", help="write full JSON run artifact")
    px.add_argument(
        "--artifact-dir",
        type=str,
        default="",
        help=(
            "results folder (always written): config, task, llm_calls, dyad_*.json, game_log, run.json. "
            "Default: results/run_YYYYMMDD_HHMMSS"
        ),
    )
    px.set_defaults(func=cmd_run)

    pb = sub.add_parser("batch", help="Run factorial batch from JSON config (progress + index.csv)")
    pb.add_argument("--config", type=str, default="", help="path to batch JSON (optional; uses built-in defaults if omitted)")
    pb.add_argument("--model", type=str, default="", help="override model in config (e.g. vllm:8000/Qwen/Qwen3-8B)")
    pb.add_argument("--runs", type=int, default=None, help="override runs_per_cell in config")
    pb.add_argument("--concurrent", type=int, default=None, help="override concurrent_runs in config")
    pb.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned grid only; write index.csv and progress without LLM",
    )
    pb.add_argument(
        "--resume",
        action="store_true",
        help="skip runs whose folder already has summary.json (keep completed work)",
    )
    pb.set_defaults(func=cmd_batch)

    pg = sub.add_parser("generate-task", help="Generate an arbitrary hidden profile task JSON")
    pg.add_argument("--domain", type=str, required=True,
                    help="Decision domain, e.g. 'medical diagnosis', 'policy decision', 'hiring'")
    pg.add_argument("--model", type=str, default="",
                    help="LLM for generation (vllm:PORT/id, gpt-4o-mini, openrouter/…)")
    pg.add_argument("--stub", action="store_true", help="Use stub LLM (produces a dummy task for testing)")
    pg.add_argument("--temperature", type=float, default=0.8,
                    help="Higher temperature = more varied tasks (default 0.8)")
    pg.add_argument("--options", type=str, default="",
                    help="Comma-separated option labels, e.g. 'X,Y,Z' (default A,B,C)")
    pg.add_argument("--task-id", type=str, default="", help="Custom task ID string")
    pg.add_argument("--out", type=str, default="", help="Output JSON path (default: task_<id>.json)")
    pg.set_defaults(func=cmd_generate_task)

    pa = sub.add_parser("analyze", help="Recompute metrics.json from an artifact folder")
    pa.add_argument("run_dir", type=str, help="path to results/run_* or batch cell run_001")
    pa.add_argument("--out", type=str, default="", help="write metrics JSON path (default: run_dir/metrics.json)")
    pa.set_defaults(func=cmd_analyze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
