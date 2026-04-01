"""
Group deliberation round: after individual decisions are made, all agents see
the full panel's choices and provide a final group recommendation.

This implements the study-design step:
  "Then one final 'group deliberation' round — all 9 agents share their
  recommendation — and a final group consensus is recorded."
"""

from __future__ import annotations

from collections import Counter

from gshp.llm.stub_client import parse_choice_json
from gshp.session import LLMClient
from gshp.types import AgentDecision, GroupDecision, GroupDeliberationResult


def deliberation_system_prompt(agent_id: int, base_system: str) -> str:
    """Reuse the agent's hiring-panel system prompt (already contains their facts)."""
    return base_system


def deliberation_user_prompt(
    agent_id: int,
    memory: str,
    individual_decisions: list[AgentDecision],
) -> str:
    """
    Show the agent every colleague's individual recommendation, then ask for a
    final group recommendation.  The full conversation log is included so the
    agent can contextualise how people reached their choices.
    """
    decisions_block = "\n".join(
        f"  Agent {d.agent_id}: {d.choice or '?'} — {d.justification[:120]}"
        for d in individual_decisions
    )
    mem_section = memory.strip() if memory.strip() else "(No conversations recorded.)"
    return (
        f"You are Agent {agent_id}. Here is a log of your conversations:\n\n"
        f"{mem_section}\n\n"
        "After all discussions concluded, each panel member shared their individual "
        "recommendation:\n"
        f"{decisions_block}\n\n"
        "Considering everything you know, your conversations, and your colleagues' "
        "views, what is your final recommendation for the group?\n"
        "Reply with **only** a single JSON object, no markdown fences:\n"
        '{"choice":"X","justification":"one short sentence"}\n'
        'Use choice exactly "X", "Y", or "Z".'
    )


def run_group_deliberation(
    individual_decisions: list[AgentDecision],
    agent_memories: dict[int, str],
    systems: dict[int, str],
    client: LLMClient,
) -> GroupDeliberationResult:
    """
    Broadcast individual decisions to all agents and collect final group recommendations.

    Parameters
    ----------
    individual_decisions:
        Each agent's initial independent choice (from the main experiment run).
    agent_memories:
        Full conversation memory per agent (concatenated transcript blocks).
    systems:
        System prompt per agent_id (same as used during dyads).
    client:
        LLM client (logging wrapper or bare client).

    Returns
    -------
    GroupDeliberationResult with one GroupDecision per agent and a group_consensus field.
    """
    group_decisions: list[GroupDecision] = []

    for decision in individual_decisions:
        aid = decision.agent_id
        mem = agent_memories.get(aid, "")
        user_msg = deliberation_user_prompt(aid, mem, individual_decisions)
        system = systems[aid]

        _meta = getattr(client, "set_call_meta", None)
        if callable(_meta):
            _meta(kind="deliberation", agent_id=aid)

        raw = client.complete(system, [{"role": "user", "content": user_msg}])
        choice, justification = parse_choice_json(raw)
        group_decisions.append(
            GroupDecision(
                agent_id=aid,
                choice=choice,
                justification=justification,
                raw_response=raw[:8000],
            )
        )

    consensus = _majority([d.choice for d in group_decisions if d.choice])
    unanimous = bool(group_decisions) and len({d.choice for d in group_decisions if d.choice}) == 1

    return GroupDeliberationResult(
        group_decisions=group_decisions,
        group_consensus=consensus,
        unanimous=unanimous,
    )


def _majority(votes: list[str]) -> str | None:
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]
