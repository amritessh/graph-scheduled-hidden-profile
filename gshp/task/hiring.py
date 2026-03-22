"""
Hidden-profile hiring task (3 candidates X, Y, Z).

Shared facts favor X; cluster + bridge facts support Y. Matches the study-design intent
for l=3, k=9 agents with cluster indices 0,1,2 and bridge-only facts on agents 2, 5, 8.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class InformationCondition(str, Enum):
    HIDDEN_PROFILE = "hidden_profile"
    SHARED_ONLY = "shared_only"


class HiringTaskSpec(BaseModel):
    """Ground-truth task: fact IDs, texts, who sees what, correct hire."""

    task_id: str = "hiring_v1"
    candidates: tuple[str, str, str] = ("X", "Y", "Z")
    correct_candidate: Literal["X", "Y", "Z"] = "Y"
    attractor_candidate: Literal["X", "Y", "Z"] = "X"

    facts: dict[str, str] = Field(
        description="fact_id -> verbatim text shown to agents who receive that fact"
    )
    shared_fact_ids: list[str]
    cluster_fact_ids: dict[int, list[str]] = Field(
        description="cluster index 0..l-1 -> fact ids known to everyone in that cluster"
    )
    bridge_agent_fact_ids: dict[int, str] = Field(
        default_factory=dict,
        description="agent_id -> single bridge fact id (cross-cluster linking)",
    )

    def fact_lines_for_agent(
        self,
        agent_id: int,
        *,
        cluster_index: int,
        condition: InformationCondition,
    ) -> str:
        """Bullet list of everything this agent is allowed to know."""
        if condition == InformationCondition.SHARED_ONLY:
            ids = list(self.shared_fact_ids)
        else:
            ids = list(self.shared_fact_ids)
            ids.extend(self.cluster_fact_ids.get(cluster_index, []))
            bid = self.bridge_agent_fact_ids.get(agent_id)
            if bid:
                ids.append(bid)
        lines = [f"- ({fid}) {self.facts[fid]}" for fid in ids if fid in self.facts]
        return "\n".join(lines) if lines else "(No facts.)"


def build_default_hiring_task() -> HiringTaskSpec:
    """
    Default instance for 9 agents / 3 clusters.

    Bridge agents 2, 5, 8 hold linking facts (one each); others in cluster share 2 facts.
    """
    facts: dict[str, str] = {
        "s_x1": "Candidate X has a high GPA and graduated with honors.",
        "s_x2": "Candidate X received strong reference letters.",
        "s_x3": "Candidate X had an excellent interview with the panel.",
        "s_x4": "Candidate X has several years of relevant experience on paper.",
        "s_y1": "Candidate Y has adequate experience for the role.",
        "s_z1": "Candidate Z holds a relevant professional certification.",
        "c0_y1": "[Cluster A] Y previously solved exactly the type of challenge this role faces.",
        "c0_y2": "[Cluster A] Y's references specifically address the hardest part of this job.",
        "c1_y1": "[Cluster B] Y's technical scores in the domain that matters most are exceptional.",
        "c1_y2": "[Cluster B] The shared packet under-represents Y's strongest projects.",
        "c2_y1": "[Cluster C] Z has a potential conflict of interest that did not appear in the shared packet.",
        "c2_y2": "[Cluster C] Y's stated values align closely with the team's mission.",
        "b2_link": "[Bridge] The problem Y solved (Cluster A) is the same technical area Cluster B flagged as critical.",
        "b5_link": "[Bridge] The under-reported strengths of Y (Cluster B) bear directly on the mission fit concerns raised in Cluster C.",
        "b8_link": "[Bridge] The conflict concern about Z (Cluster C) is unrelated to the reference depth that favors Y over X in early clusters.",
    }
    return HiringTaskSpec(
        facts=facts,
        shared_fact_ids=["s_x1", "s_x2", "s_x3", "s_x4", "s_y1", "s_z1"],
        cluster_fact_ids={
            0: ["c0_y1", "c0_y2"],
            1: ["c1_y1", "c1_y2"],
            2: ["c2_y1", "c2_y2"],
        },
        bridge_agent_fact_ids={
            2: "b2_link",
            5: "b5_link",
            8: "b8_link",
        },
    )


def cluster_index_for_agent(agent_id: int, k: int) -> int:
    return agent_id // k
