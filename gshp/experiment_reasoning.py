"""End-to-end run for collective reasoning task (no information asymmetry)."""

from __future__ import annotations

import concurrent.futures
import json
import re
from collections import Counter

from gshp.graph.caveman import CavemanTopology
from gshp.llm.logging_client import LoggingLLMClient
from gshp.prompts_reasoning import (
    agent_system_prompt_reasoning,
    deliberation_user_prompt_reasoning,
    final_user_prompt_reasoning,
)
from gshp.schedule import (
    ScheduleName,
    build_two_phase_schedule,
    expand_schedule_parallel_matchings,
)
from gshp.session import LLMClient, run_dyad_llm
from gshp.task.reasoning import ReasoningProblem, ReasoningTaskSpec
from gshp.types import AgentDecision, ExperimentRun, GroupDecision, GroupDeliberationResult, RunManifest


def parse_answer_json(raw: str) -> tuple[str | None, str]:
    """Extract answer + justification from LLM response."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    match = re.search(r'\{[^{}]*"answer"\s*:[^{}]*\}', cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return str(obj.get("answer", "")).strip(), str(obj.get("justification", ""))
        except json.JSONDecodeError:
            pass
    # fallback: grab first quoted value after "answer"
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', cleaned)
    if m:
        return m.group(1).strip(), ""
    return None, ""


def run_reasoning_experiment(
    topo: CavemanTopology,
    schedule_name: ScheduleName | str,
    task: ReasoningTaskSpec,
    problem_id: str,
    client: LLMClient,
    *,
    dyad_turns: int = 6,
    model_label: str | None = None,
    seed: int | None = None,
    parallel_dyad_layers: bool = False,
    max_workers: int = 1,
    group_deliberation: bool = False,
    verbose: bool = False,
    dyad_context_chars: int = 5000,
    final_memory_chars: int = 6500,
) -> ExperimentRun:
    if isinstance(schedule_name, str):
        schedule_name = ScheduleName(schedule_name)

    problem = task.get_problem(problem_id)
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
        task_id=f"{task.task_id}/{problem_id}",
        information_condition="full_information",
        tom_bridge=False,
        parallel_dyad_layers=parallel_dyad_layers,
    )
    run = ExperimentRun(manifest=manifest)

    systems = {i: agent_system_prompt_reasoning(i, problem) for i in range(n)}
    agent_memory: dict[int, list[str]] = {i: [] for i in range(n)}

    total_dyads = sum(len(rnd.edges) for rnd in schedule)
    dyad_num = 0
    print(f"  schedule={schedule_name.value}  problem={problem_id}", flush=True)
    print(f"  {total_dyads} dyads | {n} agents | {dyad_turns} turns each", flush=True)

    for rnd in schedule:
        layer_tag = f" L{rnd.sub_index}" if rnd.sub_index else ""
        print(f"  Round {rnd.index} [{rnd.label}{layer_tag}] — {len(rnd.edges)} dyads", flush=True)

        if max_workers > 1 and parallel_dyad_layers and len(rnd.edges) > 1:
            results = _run_layer_parallel(rnd, systems, client, dyad_turns, max_workers, dyad_context_chars)
        else:
            results = []
            for u, v in rnd.edges:
                dyad_num += 1
                print(f"    [{dyad_num}/{total_dyads}] Agent {u} <-> Agent {v} ...", flush=True)
                trans = run_dyad_llm(
                    u, v,
                    round_index=rnd.index,
                    round_label=rnd.label,
                    round_sub_index=rnd.sub_index,
                    client=client,
                    system_u=systems[u],
                    system_v=systems[v],
                    turns=dyad_turns,
                    verbose=verbose,
                    max_context_chars=dyad_context_chars,
                )
                results.append((trans, None))
                first = trans.messages[0].content[:60].replace("\n", " ") if trans.messages else ""
                print(f"    [{dyad_num}/{total_dyads}] Agent {u} <-> Agent {v} done  \"{first}...\"", flush=True)

        for (u, v), (trans, dyad_calls) in zip(rnd.edges, results):
            if max_workers > 1 and parallel_dyad_layers and len(rnd.edges) > 1:
                dyad_num += 1
                print(f"    [{dyad_num}/{total_dyads}] Agent {u} <-> Agent {v} done ({len(trans.messages)} turns)", flush=True)
            run.dyads.append(trans)
            if dyad_calls is not None:
                main_calls = getattr(client, "calls", None)
                if main_calls is not None:
                    for call in dyad_calls:
                        call["seq"] = len(main_calls)
                        main_calls.append(call)
            layer_tag = f" L{rnd.sub_index}" if rnd.sub_index else ""
            block = "\n".join(f"{m.role}: {m.content}" for m in trans.messages)
            agent_memory[u].append(f"--- [{rnd.label}{layer_tag}] conversation with Agent {v} ---\n{block}")
            agent_memory[v].append(f"--- [{rnd.label}{layer_tag}] conversation with Agent {u} ---\n{block}")

    # Individual decisions
    print("  Collecting individual decisions ...", flush=True)
    for aid in range(n):
        mem = _compress_memory(agent_memory[aid], max_chars=final_memory_chars)
        user = final_user_prompt_reasoning(aid, mem, problem)
        _meta = getattr(client, "set_call_meta", None)
        if callable(_meta):
            _meta(kind="final_decision", agent_id=aid)
        raw = client.complete(systems[aid], [{"role": "user", "content": user}])
        answer, justification = parse_answer_json(raw)
        correct = task.is_correct(problem_id, answer) if answer else False
        run.final_decisions.append(AgentDecision(
            agent_id=aid,
            choice=answer,
            justification=justification,
            raw_response=raw[:8000],
        ))
        print(f"    Agent {aid}: {answer or '?'} (correct={correct}) — {justification[:80]}", flush=True)

    # Group deliberation
    if group_deliberation:
        print("  Group deliberation ...", flush=True)
        mem_strings = {
            aid: _compress_memory(agent_memory[aid], max_chars=final_memory_chars)
            for aid in range(n)
        }
        group_decisions = []
        for decision in run.final_decisions:
            aid = decision.agent_id
            user_msg = deliberation_user_prompt_reasoning(
                aid, mem_strings[aid], run.final_decisions, problem
            )
            _meta = getattr(client, "set_call_meta", None)
            if callable(_meta):
                _meta(kind="deliberation", agent_id=aid)
            raw = client.complete(systems[aid], [{"role": "user", "content": user_msg}])
            answer, justification = parse_answer_json(raw)
            group_decisions.append(GroupDecision(
                agent_id=aid,
                choice=answer,
                justification=justification,
                raw_response=raw[:8000],
            ))
        consensus = _majority([d.choice for d in group_decisions if d.choice])
        unanimous = bool(group_decisions) and len({d.choice for d in group_decisions if d.choice}) == 1
        run.deliberation = GroupDeliberationResult(
            group_decisions=group_decisions,
            group_consensus=consensus,
            unanimous=unanimous,
        )

    # Summary metrics
    answers = [d.choice for d in run.final_decisions if d.choice]
    correct_answers = [a for a in answers if task.is_correct(problem_id, a)]
    run.notes["problem_id"] = problem_id
    run.notes["correct_answer"] = problem.correct_answer
    run.notes["num_dyads"] = len(run.dyads)
    run.notes["group_deliberation"] = group_deliberation
    run.notes["accuracy_agent_level"] = len(correct_answers) / len(answers) if answers else 0.0
    run.notes["majority_vote"] = _majority(answers)
    run.notes["majority_correct"] = task.is_correct(problem_id, _majority(answers) or "")

    if run.deliberation:
        g_answers = [d.choice for d in run.deliberation.group_decisions if d.choice]
        g_correct = [a for a in g_answers if task.is_correct(problem_id, a)]
        run.notes["group_consensus"] = run.deliberation.group_consensus
        run.notes["group_accuracy"] = len(g_correct) / len(g_answers) if g_answers else 0.0
        run.notes["group_consensus_correct"] = task.is_correct(
            problem_id, run.deliberation.group_consensus or ""
        )
        run.notes["group_unanimous"] = run.deliberation.unanimous

    return run


def _majority(votes: list[str]) -> str | None:
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]


def _compress_memory(blocks: list[str], *, max_chars: int) -> str:
    if not blocks:
        return ""
    text = "\n\n".join(blocks)
    if len(text) <= max_chars or max_chars <= 0:
        return text
    kept: list[str] = []
    used = 0
    for block in reversed(blocks):
        add = len(block) + (2 if kept else 0)
        if used + add > max_chars:
            break
        kept.append(block)
        used += add
    kept = list(reversed(kept))
    if not kept:
        return "[Earlier conversations omitted for length]\n" + text[-max_chars:]
    return "[Some earlier conversations omitted for length]\n\n" + "\n\n".join(kept)


def _run_layer_parallel(rnd, systems, client, dyad_turns, max_workers, dyad_context_chars):
    def run_one(edge):
        u, v = edge
        inner = getattr(client, "_inner", client)
        clone_fn = getattr(inner, "clone", None)
        if clone_fn:
            dyad_client = LoggingLLMClient(clone_fn(), capture_raw_completion=True)
        else:
            dyad_client = client
        trans = run_dyad_llm(
            u, v,
            round_index=rnd.index,
            round_label=rnd.label,
            round_sub_index=rnd.sub_index,
            client=dyad_client,
            system_u=systems[u],
            system_v=systems[v],
            turns=dyad_turns,
            max_context_chars=dyad_context_chars,
        )
        return trans, getattr(dyad_client, "calls", None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(rnd.edges))) as pool:
        futures = [pool.submit(run_one, edge) for edge in rnd.edges]
        return [f.result() for f in futures]
