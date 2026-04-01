"""
DV4 — Bridge Node Communication Quality.

For each message a bridge agent sends in a cross-cluster conversation, classify it as:

  relay     — bridge repeats what it heard without adapting to the recipient's knowledge state
  filter    — bridge selectively shares some items but doesn't frame for the recipient
  translate — bridge explicitly models the recipient's knowledge gaps and frames accordingly

The "translate" mode is the theoretically decisive behaviour: it's what converts a bridge
agent's structural position into actual epistemic value for the recipient.

This is a post-hoc LLM judge: it reads the conversation and classifies each bridge turn.
It requires a (typically cheap/fast) LLM client.

Usage::

    codings = code_bridge_conversations(run, task, topo_k, judge_client)
    summary = bridge_coding_summary(codings)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from gshp.session import LLMClient
from gshp.task.hiring import HiringTaskSpec, cluster_index_for_agent
from gshp.types import DyadTranscript, ExperimentRun


BRIDGE_MODES = frozenset({"relay", "filter", "translate"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BridgeTurnCoding:
    """Classification of one bridge agent turn in a cross-cluster dyad."""

    dyad_u: int
    dyad_v: int
    round_label: str
    bridge_agent: int  # which of u / v is the bridge agent
    other_agent: int   # the non-bridge participant
    turn_index: int    # 0-based turn within the dyad
    message: str       # the bridge agent's message text
    mode: str          # "relay" | "filter" | "translate"
    reasoning: str     # judge's one-sentence explanation
    parse_ok: bool = True  # False if the judge response couldn't be parsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "dyad": [self.dyad_u, self.dyad_v],
            "round_label": self.round_label,
            "bridge_agent": self.bridge_agent,
            "other_agent": self.other_agent,
            "turn_index": self.turn_index,
            "mode": self.mode,
            "reasoning": self.reasoning,
            "parse_ok": self.parse_ok,
            "message_preview": self.message[:200],
        }


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------


_JUDGE_SYSTEM = (
    "You are a research assistant coding how AI agents communicate information "
    "across groups. Reply with valid JSON only — no prose, no markdown fences."
)


def _judge_user_prompt(
    bridge_agent: int,
    other_agent: int,
    bridge_cluster: int,
    other_cluster: int,
    bridge_unique_facts: list[str],
    conversation_so_far: str,
    message_to_classify: str,
) -> str:
    facts_block = (
        "\n".join(f"  - {f}" for f in bridge_unique_facts)
        if bridge_unique_facts
        else "  (none beyond the shared panel information)"
    )
    context_block = (
        conversation_so_far.strip()
        if conversation_so_far.strip()
        else "(this is the opening message)"
    )
    return (
        f"Agent {bridge_agent} (Cluster {bridge_cluster}) is speaking with "
        f"Agent {other_agent} (Cluster {other_cluster}). "
        f"Agent {bridge_agent} acts as a bridge between the two clusters.\n\n"
        f"Information that Agent {bridge_agent} has but Agent {other_agent} likely does NOT know:\n"
        f"{facts_block}\n\n"
        f"Conversation so far:\n{context_block}\n\n"
        f"Message from Agent {bridge_agent} to classify:\n"
        f'"{message_to_classify}"\n\n'
        "Classify the communication strategy of this message as exactly one of:\n"
        '  "relay"     — passes along information without considering the recipient\'s '
        "knowledge state\n"
        '  "filter"    — selectively shares some items but doesn\'t frame for the '
        "recipient's specific knowledge gaps\n"
        '  "translate" — explicitly models the recipient\'s knowledge gaps and frames '
        "the information accordingly\n\n"
        'Reply with JSON only: {"mode": "relay" or "filter" or "translate", '
        '"reasoning": "one short sentence"}'
    )


def _parse_judge_response(raw: str) -> tuple[str, str, bool]:
    """Return (mode, reasoning, parse_ok)."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        mode = str(data.get("mode", "")).lower().strip()
        reasoning = str(data.get("reasoning", "")).strip()
        if mode in BRIDGE_MODES:
            return mode, reasoning, True
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: scan for the first mode keyword
    for mode in ("translate", "filter", "relay"):
        if mode in text.lower():
            return mode, text[:200], False
    return "relay", text[:200], False


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def _bridge_unique_facts(
    bridge_agent: int,
    task: HiringTaskSpec,
    topo_k: int,
    condition: str,
) -> list[str]:
    """Facts held by the bridge agent but NOT by all agents (cluster + bridge facts)."""
    if condition == "shared_only":
        return []
    ci = cluster_index_for_agent(bridge_agent, topo_k)
    unique: list[str] = []
    for fid in task.cluster_fact_ids.get(ci, []):
        if fid in task.facts:
            unique.append(task.facts[fid])
    bid = task.bridge_agent_fact_ids.get(bridge_agent)
    if bid and bid in task.facts:
        unique.append(task.facts[bid])
    return unique


def code_bridge_conversations(
    run: ExperimentRun,
    task: HiringTaskSpec,
    topo_k: int,
    judge_client: LLMClient,
    *,
    bridge_agent_ids: frozenset[int] | None = None,
) -> list[BridgeTurnCoding]:
    """
    Classify every bridge-agent turn in every inter-cluster dyad.

    Parameters
    ----------
    run:
        Completed experiment run.
    task:
        The hiring task spec (provides facts).
    topo_k:
        Clique size — used to derive cluster membership.
    judge_client:
        LLM client for the judge (e.g. a cheap/fast model).
    bridge_agent_ids:
        Which agents are bridge nodes. Defaults to agents in connector positions
        (inferred from ``run.notes["connector_nodes"]`` if present, else {2, 5, 8}).
    """
    if bridge_agent_ids is None:
        connector_nodes = run.notes.get("connector_nodes")
        if connector_nodes:
            bridge_agent_ids = frozenset(int(n) for n in connector_nodes)
        else:
            bridge_agent_ids = frozenset({2, 5, 8})

    condition = run.manifest.information_condition
    codings: list[BridgeTurnCoding] = []

    for dyad in run.dyads:
        if dyad.round_label != "inter_community":
            continue  # only classify bridge conversations

        u, v = dyad.u, dyad.v
        # Determine which of u, v is the bridge agent (may be both)
        bridges_in_dyad = [a for a in (u, v) if a in bridge_agent_ids]
        if not bridges_in_dyad:
            continue

        lines: list[str] = []
        for t, msg in enumerate(dyad.messages):
            speaker = int(msg.role.split("_")[-1])
            line = f"Agent {speaker}: {msg.content}"
            lines.append(line)

            if speaker not in bridge_agent_ids:
                continue  # only classify bridge agent turns

            other = v if speaker == u else u
            bridge_ci = cluster_index_for_agent(speaker, topo_k)
            other_ci = cluster_index_for_agent(other, topo_k)
            unique_facts = _bridge_unique_facts(speaker, task, topo_k, condition)
            so_far = "\n".join(lines[:-1])  # conversation before this turn

            _meta = getattr(judge_client, "set_call_meta", None)
            if callable(_meta):
                _meta(kind="bridge_judge", dyad_u=u, dyad_v=v, turn=t, bridge_agent=speaker)

            raw = judge_client.complete(
                _JUDGE_SYSTEM,
                [
                    {
                        "role": "user",
                        "content": _judge_user_prompt(
                            speaker, other,
                            bridge_ci, other_ci,
                            unique_facts,
                            so_far,
                            msg.content,
                        ),
                    }
                ],
            )
            mode, reasoning, parse_ok = _parse_judge_response(raw)
            codings.append(
                BridgeTurnCoding(
                    dyad_u=u,
                    dyad_v=v,
                    round_label=dyad.round_label,
                    bridge_agent=speaker,
                    other_agent=other,
                    turn_index=t,
                    message=msg.content,
                    mode=mode,
                    reasoning=reasoning,
                    parse_ok=parse_ok,
                )
            )

    return codings


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def bridge_coding_summary(codings: list[BridgeTurnCoding]) -> dict[str, Any]:
    """
    Aggregate mode counts per bridge agent and overall.
    The translate_rate is the key DV4 metric.
    """
    if not codings:
        return {"n_turns": 0}

    from collections import Counter

    per_agent: dict[int, list[str]] = {}
    for c in codings:
        per_agent.setdefault(c.bridge_agent, []).append(c.mode)

    agents_out: dict[str, Any] = {}
    for aid, modes in per_agent.items():
        cnt = Counter(modes)
        n = len(modes)
        agents_out[str(aid)] = {
            "n_turns": n,
            "relay": cnt.get("relay", 0),
            "filter": cnt.get("filter", 0),
            "translate": cnt.get("translate", 0),
            "translate_rate": cnt.get("translate", 0) / n,
        }

    all_modes = [c.mode for c in codings]
    cnt_all = Counter(all_modes)
    n_all = len(all_modes)
    parse_failures = sum(1 for c in codings if not c.parse_ok)

    return {
        "n_turns": n_all,
        "relay": cnt_all.get("relay", 0),
        "filter": cnt_all.get("filter", 0),
        "translate": cnt_all.get("translate", 0),
        "translate_rate": cnt_all.get("translate", 0) / n_all,
        "parse_failures": parse_failures,
        "per_bridge_agent": agents_out,
        "all_codings": [c.to_dict() for c in codings],
    }
