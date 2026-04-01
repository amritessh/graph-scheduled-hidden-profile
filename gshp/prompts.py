"""System prompts for hiring task agents (optional ToM on designated bridge agents)."""

from __future__ import annotations

# Version string written to run ``config.json`` (``audit.prompt_template_id``).
from gshp.audit import PROMPT_TEMPLATE_ID  # noqa: F401

from gshp.graph.caveman import CavemanTopology
from gshp.task.hiring import HiringTaskSpec, InformationCondition, cluster_index_for_agent

# One designated "linking" bridge per cluster in the default 3×3 ring (see task.bridge_agent_fact_ids).
DEFAULT_TOM_AGENT_IDS: frozenset[int] = frozenset({2, 5, 8})

PERSONAS: list[str] = [
    "HR and recruiting specialist",
    "Technical skills evaluator",
    "Culture and mission fit assessor",
    "HR and recruiting specialist",
    "Technical skills evaluator",
    "Culture and mission fit assessor",
    "HR and recruiting specialist",
    "Technical skills evaluator",
    "Culture and mission fit assessor",
]


def agent_system_prompt(
    agent_id: int,
    task: HiringTaskSpec,
    topo: CavemanTopology,
    *,
    condition: InformationCondition,
    tom_bridge: bool,
    tom_agent_ids: frozenset[int] | None = None,
) -> str:
    persona = PERSONAS[agent_id % len(PERSONAS)]
    ci = cluster_index_for_agent(agent_id, topo.k)
    facts_block = task.fact_lines_for_agent(
        agent_id, cluster_index=ci, condition=condition
    )

    tom_ids = tom_agent_ids if tom_agent_ids is not None else DEFAULT_TOM_AGENT_IDS
    use_tom = tom_bridge and agent_id in tom_ids

    tom_block = ""
    if use_tom:
        # Exact wording from study design spec (Riedl 2024)
        tom_block = (
            "\n\nYou connect two different groups of colleagues. Before each conversation, "
            "consider: *What does this person likely know about the candidates? "
            "What important information might they be missing that I have? "
            "What do they probably think is true, and does my information change that?* "
            "Use this reasoning to decide what to share and how to frame it.\n"
        )

    return (
        f"You are Agent {agent_id}, a {persona} on a hiring panel. "
        f"You must eventually recommend exactly one candidate: {task.candidates[0]}, "
        f"{task.candidates[1]}, or {task.candidates[2]}.\n\n"
        f"Information available to you (do not assume others know what is not listed here):\n"
        f"{facts_block}\n"
        f"{tom_block}"
        "When you discuss with a colleague, you may share any subset of what you know; "
        "do not invent facts. Keep responses concise."
    )


def final_user_prompt(agent_id: int, memory: str) -> str:
    return (
        f"You are Agent {agent_id}. Here is a log of conversations you took part in "
        f"(what was said, in order):\n\n"
        f"{memory if memory.strip() else '(You had no conversations yet.)'}\n\n"
        "After reviewing your own information and these exchanges, you must choose "
        'exactly one candidate: X, Y, or Z.\n'
        "Reply with **only** a single JSON object, no markdown fences, in this exact shape:\n"
        '{"choice":"X","justification":"one short sentence"}\n'
        'Use choice exactly "X", "Y", or "Z".'
    )
