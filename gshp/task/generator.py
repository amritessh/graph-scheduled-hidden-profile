"""
Hidden profile story generator.

Generates arbitrary hidden profile task instances for any domain using an LLM.
The structural constraint is always the same:
  - shared facts → support the WRONG answer (the attractor)
  - cluster-unique facts → support the CORRECT answer
  - bridge-unique linking facts → connect cluster information across clusters

Generated stories are returned as HiddenProfileTaskSpec objects that plug directly
into the experiment runner.

Usage (CLI)::

    python -m gshp.cli generate-task --domain "medical diagnosis" --model gpt-4o-mini

Usage (API)::

    from gshp.task.generator import generate_hidden_profile_task
    task = generate_hidden_profile_task("policy decision", client, n_clusters=3)
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any

from pydantic import BaseModel, Field

from gshp.session import LLMClient


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class HiddenProfileTaskSpec(BaseModel):
    """
    A generated hidden profile task. Drop-in replacement for HiringTaskSpec
    in the experiment runner — same field names and fact_lines_for_agent interface.
    """

    task_id: str
    domain: str
    scenario: str  # 2–3 sentence framing shown to all agents
    options: list[str] = Field(min_length=3, max_length=3)  # exactly 3 options
    correct_option: str   # the option uniquely supported by cluster+bridge facts
    attractor_option: str  # the option shared info makes look best

    # mirrors HiringTaskSpec fact structure
    facts: dict[str, str]
    shared_fact_ids: list[str]
    cluster_fact_ids: dict[int, list[str]]   # cluster index → fact ids
    bridge_agent_fact_ids: dict[int, str]    # agent id → fact id

    # Optional: why each option is right/wrong (for ground-truth analysis)
    option_rationale: dict[str, str] = Field(default_factory=dict)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    fact_eliminates: dict[str, list[str]] = Field(
        default_factory=dict,
        description="fact_id -> options ruled out by this fact (hard elimination analysis)",
    )
    interaction_eliminates: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "combination-only elimination rules. Key format: 'fid_a+fid_b[+fid_c]' "
            "with fact ids sorted lexicographically."
        ),
    )

    def fact_lines_for_agent(
        self,
        agent_id: int,
        *,
        cluster_index: int,
        condition: Any,  # InformationCondition
    ) -> str:
        """Same interface as HiringTaskSpec — works with existing prompts."""
        from gshp.task.hiring import InformationCondition

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

    # Compatibility with HiringTaskSpec callers
    @property
    def candidates(self) -> tuple[str, str, str]:
        return (self.options[0], self.options[1], self.options[2])

    @property
    def correct_candidate(self) -> str:
        return self.correct_option

    @property
    def attractor_candidate(self) -> str:
        return self.attractor_option


# ---------------------------------------------------------------------------
# Generator prompt
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = textwrap.dedent("""
    You are a research assistant designing hidden-profile experiments for multi-agent AI systems.

    A hidden profile task has exactly this structure:
      - Three options (A, B, C). One is CORRECT, one is the ATTRACTOR.
      - SHARED facts: known to all agents — these make the ATTRACTOR look best.
      - CLUSTER facts: each cluster of agents gets 2 unique facts — these support the CORRECT option.
      - BRIDGE facts: each bridge agent gets 1 linking fact — these connect two clusters' information.

    The design principle: an agent who only sees shared facts will rationally prefer the ATTRACTOR.
    Only by pooling unique information across clusters can agents discover that the CORRECT option
    is better. The bridge agents' linking facts explain HOW the cluster-unique facts connect.

    You must output valid JSON only — no prose, no markdown fences.
""").strip()


def _build_user_prompt(
    domain: str,
    n_clusters: int,
    bridge_agent_ids: list[int],
    options: list[str] | None,
) -> str:
    opts = options or ["A", "B", "C"]
    opt_str = ", ".join(opts)
    bridge_str = ", ".join(f"agent {i}" for i in bridge_agent_ids)

    return textwrap.dedent(f"""
        Create a hidden profile task for the domain: **{domain}**

        Parameters:
        - Options: {opt_str}  (use these exact labels)
        - Correct option (revealed only through unique info): {opts[1]}
        - Attractor option (shared info makes it look best): {opts[0]}
        - Clusters: {n_clusters}  (Cluster 0, Cluster 1, Cluster 2)
        - Bridge agents: {bridge_str}

        Required output (JSON):
        {{
          "scenario": "2-3 sentence setup. Describe the group's task and why they must make a collective decision. Name the options clearly.",
          "options": {json.dumps(opts)},
          "correct_option": "{opts[1]}",
          "attractor_option": "{opts[0]}",
          "option_rationale": {{
            "{opts[0]}": "why shared info makes this look best (1 sentence)",
            "{opts[1]}": "why this is actually best when all info is pooled (1 sentence)",
            "{opts[2]}": "why this is clearly weaker than both (1 sentence)"
          }},
          "facts": {{
            "s_a1": "SHARED — strong positive fact about {opts[0]}",
            "s_a2": "SHARED — strong positive fact about {opts[0]}",
            "s_a3": "SHARED — strong positive fact about {opts[0]}",
            "s_a4": "SHARED — strong positive fact about {opts[0]}",
            "s_b1": "SHARED — one weak positive fact about {opts[1]} (not enough to prefer it)",
            "s_c1": "SHARED — one weak positive fact about {opts[2]} (not enough to prefer it)",
            "c0_b1": "CLUSTER 0 ONLY — strong fact supporting {opts[1]}. Be specific.",
            "c0_b2": "CLUSTER 0 ONLY — second strong fact supporting {opts[1]}. Be specific.",
            "c1_b1": "CLUSTER 1 ONLY — strong fact supporting {opts[1]}. Be specific, different angle from Cluster 0.",
            "c1_b2": "CLUSTER 1 ONLY — second strong fact supporting {opts[1]}. Be specific.",
            "c2_b1": "CLUSTER 2 ONLY — fact exposing a problem with {opts[2]} OR strongly supporting {opts[1]}.",
            "c2_b2": "CLUSTER 2 ONLY — second fact distinguishing {opts[1]} from {opts[2]}.",
            "bridge_{bridge_agent_ids[0]}": "BRIDGE agent {bridge_agent_ids[0]} ONLY — linking fact connecting Cluster 0's unique info to Cluster 1's. Explain HOW they are related.",
            "bridge_{bridge_agent_ids[1]}": "BRIDGE agent {bridge_agent_ids[1]} ONLY — linking fact connecting Cluster 1's unique info to Cluster 2's.",
            "bridge_{bridge_agent_ids[2]}": "BRIDGE agent {bridge_agent_ids[2]} ONLY — linking fact connecting Cluster 2's unique info back to Cluster 0's."
          }},
          "shared_fact_ids": ["s_a1", "s_a2", "s_a3", "s_a4", "s_b1", "s_c1"],
          "cluster_fact_ids": {{
            "0": ["c0_b1", "c0_b2"],
            "1": ["c1_b1", "c1_b2"],
            "2": ["c2_b1", "c2_b2"]
          }},
          "bridge_agent_fact_ids": {{
            "{bridge_agent_ids[0]}": "bridge_{bridge_agent_ids[0]}",
            "{bridge_agent_ids[1]}": "bridge_{bridge_agent_ids[1]}",
            "{bridge_agent_ids[2]}": "bridge_{bridge_agent_ids[2]}"
          }},
          "fact_eliminates": {{
            "s_a1": [],
            "s_a2": [],
            "s_a3": [],
            "s_a4": [],
            "s_b1": [],
            "s_c1": [],
            "c0_b1": ["{opts[0]}"],
            "c0_b2": ["{opts[0]}"],
            "c1_b1": ["{opts[0]}"],
            "c1_b2": ["{opts[0]}"],
            "c2_b1": ["{opts[2]}"],
            "c2_b2": ["{opts[0]}"],
            "bridge_{bridge_agent_ids[0]}": [],
            "bridge_{bridge_agent_ids[1]}": [],
            "bridge_{bridge_agent_ids[2]}": []
          }},
          "interaction_eliminates": {{
            "bridge_{bridge_agent_ids[0]}+c0_b1": ["{opts[0]}"],
            "bridge_{bridge_agent_ids[0]}+c1_b1": ["{opts[0]}"],
            "bridge_{bridge_agent_ids[1]}+c1_b2": ["{opts[0]}"],
            "bridge_{bridge_agent_ids[1]}+c2_b2": ["{opts[0]}"],
            "bridge_{bridge_agent_ids[2]}+c2_b1": ["{opts[2]}"]
          }}
        }}

        Rules:
        1. Every fact must be a complete, concrete sentence — no placeholders.
        2. Shared facts must collectively make {opts[0]} look clearly best to an agent with ONLY those facts.
        3. Cluster facts must be specific enough that an agent reading them would update toward {opts[1]}.
        4. Bridge linking facts must explicitly connect what one cluster knows to what another cluster knows.
        5. The scenario must make the information distribution feel natural (agents have different specialties, access, or roles).
        6. Keep facts under 40 words each.
        7. Output valid JSON only.
    """).strip()


# ---------------------------------------------------------------------------
# Parser / validator
# ---------------------------------------------------------------------------


def _parse_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    # Strip optional markdown fence
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if fence:
        text = fence.group(1).strip()
    # Try to extract JSON object if there's surrounding text
    brace = re.search(r'\{[\s\S]*\}', text)
    if brace:
        text = brace.group(0)
    return json.loads(text)


def _validate_and_coerce(data: dict[str, Any], bridge_agent_ids: list[int]) -> dict[str, Any]:
    """
    Ensure cluster_fact_ids has integer keys (LLM sometimes emits strings).
    Ensure bridge_agent_fact_ids has integer keys.
    """
    # cluster_fact_ids: coerce string keys to int
    raw_cluster = data.get("cluster_fact_ids", {})
    data["cluster_fact_ids"] = {int(k): v for k, v in raw_cluster.items()}

    # bridge_agent_fact_ids: coerce string keys to int
    raw_bridge = data.get("bridge_agent_fact_ids", {})
    data["bridge_agent_fact_ids"] = {int(k): v for k, v in raw_bridge.items()}

    # Verify all referenced fact IDs exist in facts dict
    facts = data.get("facts", {})
    missing = []
    for fid in data.get("shared_fact_ids", []):
        if fid not in facts:
            missing.append(fid)
    for fids in data["cluster_fact_ids"].values():
        for fid in fids:
            if fid not in facts:
                missing.append(fid)
    for fid in data["bridge_agent_fact_ids"].values():
        if fid not in facts:
            missing.append(fid)
    if missing:
        raise ValueError(f"Generated task references fact IDs not in facts dict: {missing}")

    # Ensure elimination maps exist and reference valid options/facts.
    options = list(data.get("options", []))
    valid_opts = set(options)
    fe = data.get("fact_eliminates") or {}
    cleaned_fe: dict[str, list[str]] = {}
    for fid, opts in fe.items():
        if fid not in facts:
            continue
        keep = [o for o in (opts or []) if o in valid_opts]
        cleaned_fe[fid] = keep
    for fid in facts:
        cleaned_fe.setdefault(fid, [])
    data["fact_eliminates"] = cleaned_fe

    ie = data.get("interaction_eliminates") or {}
    cleaned_ie: dict[str, list[str]] = {}
    for key, opts in ie.items():
        fids = [x.strip() for x in str(key).split("+") if x.strip()]
        if not fids or any(fid not in facts for fid in fids):
            continue
        norm_key = "+".join(sorted(fids))
        keep = [o for o in (opts or []) if o in valid_opts]
        cleaned_ie[norm_key] = keep
    data["interaction_eliminates"] = cleaned_ie

    return data


def _quality_report(data: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministic quality checks for generated tasks.

    A task should satisfy:
    - shared facts do NOT strongly eliminate options (attractor remains plausible)
    - unique+bridge facts are sufficient to eliminate to one option
    - correct option survives all elimination rules
    """
    facts = data.get("facts", {}) or {}
    options = list(data.get("options", []) or [])
    correct = str(data.get("correct_option", "")).strip()
    shared = list(data.get("shared_fact_ids", []) or [])
    cluster_map = data.get("cluster_fact_ids", {}) or {}
    bridge_map = data.get("bridge_agent_fact_ids", {}) or {}
    fact_elim = data.get("fact_eliminates", {}) or {}
    interaction_elim = data.get("interaction_eliminates", {}) or {}

    issues: list[str] = []
    score = 1.0

    if correct not in options:
        issues.append("correct_option is not in options")
        score -= 0.5

    # Shared facts should not collapse the answer set.
    shared_elims = set()
    for fid in shared:
        shared_elims.update(x for x in fact_elim.get(fid, []) if x in options)
    remaining_after_shared = [o for o in options if o not in shared_elims]
    if len(remaining_after_shared) <= 1:
        issues.append("shared facts over-determine outcome (attractor too strong/decisive)")
        score -= 0.4

    # Full known facts should isolate to one answer and preserve correctness.
    all_facts = set(facts.keys())
    remaining_full = _apply_elimination_rules(
        options,
        all_facts,
        fact_elim,
        interaction_elim,
    )
    if len(remaining_full) != 1:
        issues.append("full information does not reduce to a unique answer")
        score -= 0.4
    elif remaining_full[0] != correct:
        issues.append("full information resolves to non-correct option")
        score -= 0.4

    if correct not in remaining_after_shared:
        issues.append("shared facts eliminate the correct option outright")
        score -= 0.3

    # Unique facts should add elimination power beyond shared baseline.
    unique_fact_ids = set()
    for fids in cluster_map.values():
        unique_fact_ids.update(fids or [])
    unique_fact_ids.update(bridge_map.values())
    remaining_shared_unique = _apply_elimination_rules(
        options,
        set(shared) | unique_fact_ids,
        fact_elim,
        interaction_elim,
    )
    if len(remaining_shared_unique) >= len(remaining_after_shared):
        issues.append("unique facts provide little/no additional elimination power")
        score -= 0.2

    return {
        "score": max(0.0, score),
        "passed": score >= 0.7 and not issues,
        "issues": issues,
        "remaining_after_shared": remaining_after_shared,
        "remaining_with_all_facts": remaining_full,
        "n_unique_facts": len(unique_fact_ids),
    }


def _apply_elimination_rules(
    options: list[str],
    seen_facts: set[str],
    fact_eliminates: dict[str, list[str]],
    interaction_eliminates: dict[str, list[str]],
) -> list[str]:
    feasible = set(options)
    for fid in seen_facts:
        feasible -= set(x for x in (fact_eliminates.get(fid) or []) if x in feasible)
    for key, elim in interaction_eliminates.items():
        needed = {x.strip() for x in str(key).split("+") if x.strip()}
        if needed and needed.issubset(seen_facts):
            feasible -= set(x for x in (elim or []) if x in feasible)
    return sorted(feasible)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_hidden_profile_task(
    domain: str,
    client: LLMClient,
    *,
    n_clusters: int = 3,
    bridge_agent_ids: list[int] | None = None,
    options: list[str] | None = None,
    task_id: str | None = None,
    max_retries: int = 3,
    min_quality_score: float = 0.7,
) -> HiddenProfileTaskSpec:
    """
    Generate an arbitrary hidden profile task for ``domain`` using ``client``.

    Parameters
    ----------
    domain:
        The decision domain, e.g. "hiring", "medical diagnosis", "policy decision",
        "military strategy", "investment committee", "product launch".
    client:
        Any LLMClient (real or stub). A capable model (GPT-4o-mini or better) produces
        cleaner tasks; smaller models may need retries.
    n_clusters:
        Number of clusters (must match the experiment topology). Default 3.
    bridge_agent_ids:
        Agent IDs that are bridge nodes and receive linking facts.
        Defaults to [k-1, 2k-1, 3k-1] for k=3 → [2, 5, 8].
    options:
        Labels for the three options. Default ["A", "B", "C"].
    task_id:
        Identifier written to artifacts. Auto-generated from domain if not provided.
    max_retries:
        Number of generation attempts before raising.
    """
    if bridge_agent_ids is None:
        # Default: last node of each cluster for 3×3 topology
        k = 3  # default clique size
        bridge_agent_ids = [(i + 1) * k - 1 for i in range(n_clusters)]

    if options is None:
        options = ["A", "B", "C"]

    if task_id is None:
        slug = re.sub(r"[^a-z0-9]+", "_", domain.lower().strip()).strip("_")
        task_id = f"hp_{slug}"

    user_prompt = _build_user_prompt(domain, n_clusters, bridge_agent_ids, options)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = client.complete(_SYSTEM_PROMPT, [{"role": "user", "content": user_prompt}])
            data = _parse_response(raw)
            data = _validate_and_coerce(data, bridge_agent_ids)
            quality = _quality_report(data)
            if quality["score"] < min_quality_score:
                raise ValueError(
                    f"Generated task failed quality threshold ({quality['score']:.3f} < {min_quality_score:.3f}): "
                    + "; ".join(quality["issues"])
                )
            data["task_id"] = task_id
            data["domain"] = domain
            data["quality_report"] = quality
            return HiddenProfileTaskSpec(**data)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                continue

    raise RuntimeError(
        f"Failed to generate a valid hidden profile task for domain '{domain}' "
        f"after {max_retries} attempts. Last error: {last_err}"
    ) from last_err


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def save_task(task: HiddenProfileTaskSpec, path: str) -> None:
    """Save a generated task to JSON for reuse across experiment runs."""
    import pathlib
    pathlib.Path(path).write_text(task.model_dump_json(indent=2), encoding="utf-8")


def load_task(path: str) -> HiddenProfileTaskSpec:
    """Load a previously generated task from JSON."""
    import pathlib
    return HiddenProfileTaskSpec.model_validate_json(
        pathlib.Path(path).read_text(encoding="utf-8")
    )
