"""CLI: inspect topology and dry-run schedules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gshp.graph.caveman import CavemanTopology
from gshp.runner import run_experiment
from gshp.schedule import ScheduleName, build_two_phase_schedule


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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
