"""Orchestrate one full experiment run over a schedule."""

from __future__ import annotations

from collections.abc import Callable

from gshp.graph.caveman import CavemanTopology
from gshp.schedule import (
    ScheduleName,
    build_two_phase_schedule,
    expand_schedule_parallel_matchings,
)
from gshp.session import run_dyad_stub
from gshp.types import DyadTranscript, ExperimentRun, RunManifest


DyadFn = Callable[..., DyadTranscript]


def run_experiment(
    topo: CavemanTopology,
    schedule_name: ScheduleName | str,
    *,
    dyad_fn: DyadFn | None = None,
    manifest_extras: dict | None = None,
    parallel_dyad_layers: bool = False,
) -> ExperimentRun:
    """
    Execute every dyad in each communication round (sequential order within the round).

    v0: stub dyads. Later: pass dyad_fn that calls vLLM with per-agent prompts + task state.
    """
    if isinstance(schedule_name, str):
        schedule_name = ScheduleName(schedule_name)

    schedule = build_two_phase_schedule(topo, schedule_name)
    if parallel_dyad_layers:
        schedule = expand_schedule_parallel_matchings(schedule)
    dyad_fn = dyad_fn or run_dyad_stub

    manifest = RunManifest(
        schedule=schedule_name.value,
        l=topo.l,
        k=topo.k,
        parallel_dyad_layers=parallel_dyad_layers,
        **(manifest_extras or {}),
    )
    run = ExperimentRun(manifest=manifest)

    for rnd in schedule:
        for u, v in rnd.edges:
            t = dyad_fn(
                u,
                v,
                round_index=rnd.index,
                round_label=rnd.label,
                round_sub_index=rnd.sub_index,
            )
            run.dyads.append(t)

    run.notes["num_dyads"] = len(run.dyads)
    run.notes["connector_nodes"] = sorted(topo.connector_nodes)
    return run
