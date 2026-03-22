"""CLI: inspect topology, dry-run schedules, full hiring experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from gshp.artifacts import write_run_bundle
from gshp.experiment import run_hidden_profile_hiring
from gshp.graph.caveman import CavemanTopology
from gshp.llm.logging_client import LoggingLLMClient
from gshp.llm.openai_local import make_llm_client
from gshp.llm.stub_client import StubLLM
from gshp.runner import run_experiment
from gshp.schedule import ScheduleName, build_two_phase_schedule
from gshp.task.hiring import InformationCondition, build_default_hiring_task


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
    print(f"Schedule: {sched.value}")
    for r in rounds:
        print(f"  Round {r.index} [{r.label}]: {len(r.edges)} dyads")
    run = run_experiment(topo, sched)
    out = Path(args.out) if args.out else None
    payload = run.model_dump(mode="json")
    if out:
        out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out}")
    else:
        print(json.dumps(payload, indent=2)[:2000] + ("..." if len(json.dumps(payload)) > 2000 else ""))


def cmd_run(args: argparse.Namespace) -> None:
    topo = CavemanTopology.build(l=args.l, k=args.k, kind=args.kind)
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

    artifact_path: Path | None = None
    if args.artifact_dir:
        artifact_path = Path(args.artifact_dir)
    elif args.save_artifacts:
        artifact_path = Path("results") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    client: StubLLM | LoggingLLMClient | object = base_client
    if artifact_path is not None:
        client = LoggingLLMClient(base_client)

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
    )

    out = Path(args.out) if args.out else None
    payload = run.model_dump(mode="json")
    if out:
        out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out}")

    if artifact_path is not None:
        assert isinstance(client, LoggingLLMClient)
        root = write_run_bundle(
            artifact_path,
            run=run,
            task=task,
            topo=topo,
            llm_calls=client.calls,
        )
        print(f"Artifact bundle: {root} (config, task, summary, llm_calls, dyad_*.json, game_log.txt, run.json)")

    print("--- summary ---")
    print(f"accuracy (agent-level): {run.notes.get('accuracy_agent_level')}")
    print(f"majority: {run.notes.get('majority_vote')} (correct={task.correct_candidate})")
    print(f"unanimous correct: {run.notes.get('unanimous_correct')}")
    for d in run.final_decisions:
        tail = d.justification[:80] + ("..." if len(d.justification) > 80 else "")
        print(f"  Agent {d.agent_id}: {d.choice} — {tail}")


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
    px.add_argument("--out", type=str, default="", help="write full JSON run artifact")
    px.add_argument(
        "--artifact-dir",
        type=str,
        default="",
        help="write full results folder (config, task, llm_calls, per-dyad JSON, game_log, run.json)",
    )
    px.add_argument(
        "--save-artifacts",
        action="store_true",
        help="same as --artifact-dir results/run_YYYYMMDD_HHMMSS",
    )
    px.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
