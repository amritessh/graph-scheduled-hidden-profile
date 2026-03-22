"""End-to-end run: topology + schedule + hiring task + LLM dyads + final JSON votes."""

from __future__ import annotations

from gshp.graph.caveman import CavemanTopology
from gshp.llm.stub_client import parse_choice_json
from gshp.prompts import agent_system_prompt, final_user_prompt
from gshp.schedule import ScheduleName, build_two_phase_schedule
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
) -> ExperimentRun:
    """
    Run all scheduled dyads (each agent only knows facts from ``task`` under ``condition``),
    then query each agent for a structured hire decision.
    """
    if isinstance(schedule_name, str):
        schedule_name = ScheduleName(schedule_name)

    n = topo.l * topo.k
    schedule = build_two_phase_schedule(topo, schedule_name)

    manifest = RunManifest(
        schedule=schedule_name.value,
        l=topo.l,
        k=topo.k,
        model=model_label,
        seed=seed,
        task_id=task.task_id,
        information_condition=condition.value,
        tom_bridge=tom_bridge,
    )
    run = ExperimentRun(manifest=manifest)

    systems = {
        i: agent_system_prompt(
            i, task, topo, condition=condition, tom_bridge=tom_bridge
        )
        for i in range(n)
    }
    agent_memory: dict[int, list[str]] = {i: [] for i in range(n)}

    for rnd in schedule:
        for u, v in rnd.edges:
            trans = run_dyad_llm(
                u,
                v,
                round_index=rnd.index,
                round_label=rnd.label,
                client=client,
                system_u=systems[u],
                system_v=systems[v],
                turns=dyad_turns,
            )
            run.dyads.append(trans)
            block = "\n".join(f"{m.role}: {m.content}" for m in trans.messages)
            agent_memory[u].append(
                f"--- [{rnd.label}] conversation with Agent {v} ---\n{block}"
            )
            agent_memory[v].append(
                f"--- [{rnd.label}] conversation with Agent {u} ---\n{block}"
            )

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

    correct = task.correct_candidate
    votes = [d.choice for d in run.final_decisions if d.choice]
    run.notes["num_dyads"] = len(run.dyads)
    run.notes["connector_nodes"] = sorted(topo.connector_nodes)
    run.notes["accuracy_agent_level"] = (
        sum(1 for c in votes if c == correct) / len(votes) if votes else 0.0
    )
    run.notes["unanimous_correct"] = bool(votes) and all(c == correct for c in votes)
    run.notes["majority_vote"] = _majority(votes) if votes else None
    return run


def _majority(votes: list[str]) -> str | None:
    from collections import Counter

    if not votes:
        return None
    c = Counter(votes)
    return c.most_common(1)[0][0]
