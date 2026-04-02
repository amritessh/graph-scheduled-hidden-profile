"""End-to-end run: topology + schedule + hiring task + LLM dyads + final JSON votes."""

from __future__ import annotations

import concurrent.futures
from collections import Counter

from gshp.deliberation import run_group_deliberation
from gshp.graph.caveman import CavemanTopology
from gshp.llm.logging_client import LoggingLLMClient
from gshp.llm.stub_client import parse_choice_json
from gshp.prompts import agent_system_prompt, final_user_prompt
from gshp.schedule import (
    ScheduleName,
    build_two_phase_schedule,
    expand_schedule_parallel_matchings,
)
from gshp.session import LLMClient, run_dyad_llm
from gshp.task.hiring import HiringTaskSpec, InformationCondition
from gshp.types import AgentDecision, ExperimentRun, RunManifest


def run_hidden_profile_hiring(
    topo: CavemanTopology,
    schedule_name: ScheduleName | str,
    task: HiringTaskSpec,
    client: LLMClient,
    *,
    condition: InformationCondition = InformationCondition.HIDDEN_PROFILE,
    tom_bridge: bool = False,
    dyad_turns: int = 6,
    model_label: str | None = None,
    seed: int | None = None,
    parallel_dyad_layers: bool = False,
    max_workers: int = 1,
    group_deliberation: bool = False,
) -> ExperimentRun:
    """
    Run all scheduled dyads then query each agent for a structured hire decision.

    Parameters
    ----------
    max_workers:
        Number of parallel threads for dyads within a matching layer.
        Requires ``parallel_dyad_layers=True`` to have any effect (layers must
        be computed before parallelism can be exploited).
        Set > 1 only when ``client`` is a ``LoggingLLMClient`` wrapping an
        ``OpenAICompatibleChat`` — the inner client's ``clone()`` method is used
        to give each thread its own connection state.
    group_deliberation:
        If True, run a group deliberation round after individual decisions:
        all agents see the full panel's choices and provide a final group
        recommendation. Stored in ``run.deliberation``.
    """
    if isinstance(schedule_name, str):
        schedule_name = ScheduleName(schedule_name)

    n = topo.l * topo.k
    schedule = build_two_phase_schedule(topo, schedule_name)
    if parallel_dyad_layers:
        schedule = expand_schedule_parallel_matchings(schedule)

    manifest = RunManifest(
        schedule=schedule_name.value,
        l=topo.l,
        k=topo.k,
        model=model_label,
        seed=seed,
        task_id=task.task_id,
        information_condition=condition.value,
        tom_bridge=tom_bridge,
        parallel_dyad_layers=parallel_dyad_layers,
    )
    run = ExperimentRun(manifest=manifest)

    systems = {
        i: agent_system_prompt(
            i, task, topo, condition=condition, tom_bridge=tom_bridge
        )
        for i in range(n)
    }
    agent_memory: dict[int, list[str]] = {i: [] for i in range(n)}

    use_parallel = max_workers > 1 and parallel_dyad_layers

    total_dyads = sum(len(rnd.edges) for rnd in schedule)
    dyad_num = 0
    print(f"  schedule={schedule_name.value}  condition={condition.value}  tom_bridge={tom_bridge}", flush=True)
    print(f"  {total_dyads} dyads | {n} agents | {dyad_turns} turns each", flush=True)

    for rnd in schedule:
        layer_tag = f" L{rnd.sub_index}" if rnd.sub_index else ""
        print(f"  Round {rnd.index} [{rnd.label}{layer_tag}] — {len(rnd.edges)} dyads", flush=True)
        if use_parallel and len(rnd.edges) > 1:
            results = _run_layer_parallel(rnd, systems, client, dyad_turns, max_workers)
        else:
            results = []
            for u, v in rnd.edges:
                dyad_num += 1
                print(f"    [{dyad_num}/{total_dyads}] Agent {u} <-> Agent {v} — running ...", flush=True)
                results.append(_run_one_dyad(u, v, rnd, systems, client, dyad_turns))
                trans, _ = results[-1]
                print(f"    [{dyad_num}/{total_dyads}] Agent {u} <-> Agent {v} — done ({len(trans.messages)} turns)", flush=True)

        for (u, v), (trans, dyad_calls) in zip(rnd.edges, results):
            if use_parallel and len(rnd.edges) > 1:
                dyad_num += 1
                print(f"    [{dyad_num}/{total_dyads}] Agent {u} <-> Agent {v} — done ({len(trans.messages)} turns)", flush=True)
            run.dyads.append(trans)
            if dyad_calls is not None:
                main_calls = getattr(client, "calls", None)
                if main_calls is not None:
                    for call in dyad_calls:
                        call["seq"] = len(main_calls)
                        main_calls.append(call)
            layer_tag = f" L{rnd.sub_index}" if rnd.sub_index else ""
            block = "\n".join(f"{m.role}: {m.content}" for m in trans.messages)
            agent_memory[u].append(
                f"--- [{rnd.label}{layer_tag}] conversation with Agent {v} ---\n{block}"
            )
            agent_memory[v].append(
                f"--- [{rnd.label}{layer_tag}] conversation with Agent {u} ---\n{block}"
            )

    # Individual decisions
    print(f"  Collecting individual decisions ...", flush=True)
    for aid in range(n):
        mem = "\n\n".join(agent_memory[aid])
        user = final_user_prompt(aid, mem)
        _meta = getattr(client, "set_call_meta", None)
        if callable(_meta):
            _meta(kind="final_decision", agent_id=aid)
        raw = client.complete(systems[aid], [{"role": "user", "content": user}])
        choice, justification = parse_choice_json(raw)
        run.final_decisions.append(
            AgentDecision(
                agent_id=aid,
                choice=choice,
                justification=justification,
                raw_response=raw[:8000],
            )
        )
        print(f"    Agent {aid}: {choice or '?'} — {justification[:80]}", flush=True)

    # Group deliberation round (optional)
    if group_deliberation:
        print(f"  Group deliberation ...", flush=True)
        mem_strings = {aid: "\n\n".join(agent_memory[aid]) for aid in range(n)}
        run.deliberation = run_group_deliberation(
            individual_decisions=run.final_decisions,
            agent_memories=mem_strings,
            systems=systems,
            client=client,
        )

    # Summary notes
    correct = task.correct_candidate
    votes = [d.choice for d in run.final_decisions if d.choice]
    run.notes["num_dyads"] = len(run.dyads)
    run.notes["parallel_dyad_layers"] = parallel_dyad_layers
    run.notes["max_workers"] = max_workers
    run.notes["group_deliberation"] = group_deliberation
    if parallel_dyad_layers:
        run.notes["schedule_layers_total"] = len(schedule)
    run.notes["connector_nodes"] = sorted(topo.connector_nodes)
    run.notes["accuracy_agent_level"] = (
        sum(1 for c in votes if c == correct) / len(votes) if votes else 0.0
    )
    run.notes["unanimous_correct"] = bool(votes) and all(c == correct for c in votes)
    run.notes["majority_vote"] = _majority(votes) if votes else None

    if run.deliberation:
        g_votes = [d.choice for d in run.deliberation.group_decisions if d.choice]
        run.notes["group_consensus"] = run.deliberation.group_consensus
        run.notes["group_accuracy"] = (
            sum(1 for c in g_votes if c == correct) / len(g_votes) if g_votes else 0.0
        )
        run.notes["group_unanimous"] = run.deliberation.unanimous

    return run


# ---------------------------------------------------------------------------
# Parallel / serial layer helpers
# ---------------------------------------------------------------------------


def _run_one_dyad(u, v, rnd, systems, client, dyad_turns):
    """Run a single dyad using the shared client. Returns (transcript, None)."""
    trans = run_dyad_llm(
        u, v,
        round_index=rnd.index,
        round_label=rnd.label,
        round_sub_index=rnd.sub_index,
        client=client,
        system_u=systems[u],
        system_v=systems[v],
        turns=dyad_turns,
    )
    return trans, None  # None = calls already in main client


def _make_dyad_client(main_client) -> tuple[object, bool]:
    """
    Create a per-dyad logging client for parallel execution.

    Returns (dyad_client, calls_need_merge).
    - If main_client is a LoggingLLMClient wrapping a cloneable inner client,
      each dyad gets its own clone → fresh last_completion → no races.
    - Otherwise fall back to sharing the inner client without capturing raw completions.
    """
    inner = getattr(main_client, "_inner", main_client)
    capture_raw = getattr(main_client, "capture_raw_completion", True)

    clone_fn = getattr(inner, "clone", None)
    if callable(clone_fn):
        dyad_inner = clone_fn()
    else:
        # Stateless clients (StubLLM) are safe to share; disable raw capture to avoid races
        dyad_inner = inner
        capture_raw = False

    return LoggingLLMClient(dyad_inner, capture_raw_completion=capture_raw), True


def _run_layer_parallel(rnd, systems, client, dyad_turns, max_workers):
    """
    Run all dyads in a matching layer concurrently.

    Each dyad gets its own LoggingLLMClient with a cloned inner client so that
    ``last_completion`` state is never shared between threads. After the pool
    completes, results are returned in deterministic edge order.
    """
    def run_one(edge):
        u, v = edge
        dyad_client, _ = _make_dyad_client(client)
        trans = run_dyad_llm(
            u, v,
            round_index=rnd.index,
            round_label=rnd.label,
            round_sub_index=rnd.sub_index,
            client=dyad_client,
            system_u=systems[u],
            system_v=systems[v],
            turns=dyad_turns,
        )
        return trans, dyad_client.calls

    n_workers = min(max_workers, len(rnd.edges))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        # Submit in edge order; .result() preserves submission order
        futures = [pool.submit(run_one, edge) for edge in rnd.edges]
        return [f.result() for f in futures]


def _majority(votes: list[str]) -> str | None:
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]
