"""
LLM-based semantic fact classifier for DV2.

Replaces keyword matching with semantic detection — agents paraphrase facts,
so exact string matching misses most disclosures. This module classifies each
dyad message against all task facts using an LLM judge.

Usage::

    from gshp.fact_classifier import classify_run
    records = classify_run(run_dir, task, topo_k, client, condition)
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from gshp.task.hiring import HiringTaskSpec, InformationCondition, cluster_index_for_agent


_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a precise fact-detection assistant. "
    "Given a message from a hiring panel discussion, "
    "identify which facts from a provided list are conveyed — even if paraphrased."
)


def _classify_message(message: str, facts: dict[str, str], client: Any) -> dict[str, bool]:
    """
    Ask the LLM which facts are conveyed in ``message``.

    Returns {fact_id: True/False} for all facts.
    """
    clean = _strip_think(message)
    if len(clean) < 15:
        return {fid: False for fid in facts}

    facts_block = "\n".join(f'{fid}: "{text}"' for fid, text in facts.items())

    prompt = (
        f'Message: "{clean}"\n\n'
        f"Facts to check:\n{facts_block}\n\n"
        "For each fact id, output 1 if the message conveys that information, 0 if not.\n"
        "Be LIBERAL in matching — mark 1 if:\n"
        "- The message says the same thing in different words\n"
        "- The message draws a conclusion that directly follows from the fact\n"
        "- The message references the cross-cluster connection described by the fact\n"
        "- The message mentions 'bridge', 'bridge note', 'bridge link' in context of the fact's content\n"
        "Examples: 'Cluster A aligns with Cluster B needs' conveys a fact linking Cluster A and Cluster B. "
        "'validated by bridge note' conveys a bridge linking fact if the surrounding context matches.\n"
        'Output ONLY a JSON object like {"s_x1": 0, "c0_y1": 1, ...}. No explanation.'
    )

    raw = client.complete(_SYSTEM, [{"role": "user", "content": prompt}])
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if not match:
        return {fid: False for fid in facts}
    try:
        parsed = json.loads(match.group())
        return {fid: bool(parsed.get(fid, 0)) for fid in facts}
    except json.JSONDecodeError:
        return {fid: False for fid in facts}


# ---------------------------------------------------------------------------
# Run-level classification
# ---------------------------------------------------------------------------

def classify_run(
    run_dir: Path | str,
    task: HiringTaskSpec,
    topo_k: int,
    client: Any,
    condition: InformationCondition,
    *,
    workers: int = 5,
) -> dict[str, Any]:
    """
    Classify all messages in a run directory.

    Reads all dyad_*.json files, classifies each message, then computes
    disclosure / transmission / integration rates — same structure as
    fact_transmission.json but semantically detected.

    Returns the summary dict (also writes fact_classification.json).
    """
    run_dir = Path(run_dir)

    # load task facts relevant to this condition
    if condition == InformationCondition.SHARED_ONLY:
        active_fids = set(task.shared_fact_ids)
    else:
        active_fids = set(task.shared_fact_ids)
        for fids in task.cluster_fact_ids.values():
            active_fids.update(fids)
        for fid in task.bridge_agent_fact_ids.values():
            active_fids.add(fid)

    active_facts = {fid: task.facts[fid] for fid in active_fids if fid in task.facts}

    # collect all messages to classify
    tasks: list[tuple[str, str, int, int, bool, str]] = []  # (dyad_path, msg_idx, u, v, is_inter, role)

    dyad_files = sorted(run_dir.glob("dyad_*.json"))
    dyad_data: dict[str, Any] = {}

    for dyad_file in dyad_files:
        dyad = json.loads(dyad_file.read_text())
        dyad_data[dyad_file.name] = dyad
        u, v = dyad["u"], dyad["v"]
        cu = cluster_index_for_agent(u, topo_k)
        cv = cluster_index_for_agent(v, topo_k)
        is_inter = cu != cv
        for i, msg in enumerate(dyad["messages"]):
            tasks.append((dyad_file.name, i, u, v, is_inter, msg["content"]))

    # classify in parallel
    results: dict[tuple[str, int], dict[str, bool]] = {}

    def _classify(task_item):
        fname, idx, u, v, is_inter, content = task_item
        detected = _classify_message(content, active_facts, client)
        hits = [fid for fid, v in detected.items() if v]
        preview = content[:60].replace("\n", " ")
        print(f"      msg {fname}[{idx}] u={u} v={v} inter={is_inter} hits={hits} | \"{preview}\"", flush=True)
        return (fname, idx), detected

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_classify, t): t for t in tasks}
        for fut in as_completed(futures):
            key, detected = fut.result()
            results[key] = detected

    # --- compute DV2 rates ---
    # per fact: disclosed, transmitted, integrated
    fact_records: dict[str, dict[str, Any]] = {
        fid: {"disclosed": False, "transmitted": False, "integrated_by": []}
        for fid in active_facts
    }

    for fname, idx, u, v, is_inter, _ in tasks:
        detected = results.get((fname, idx), {})
        for fid, hit in detected.items():
            if not hit:
                continue
            rec = fact_records[fid]
            rec["disclosed"] = True
            if is_inter:
                rec["transmitted"] = True

    # integration: check final decision justifications
    run_json_path = run_dir / "run.json"
    if run_json_path.exists():
        run_data = json.loads(run_json_path.read_text())
        for dec in run_data.get("final_decisions", []):
            justification = dec.get("justification", "")
            if not justification.strip():
                continue
            detected = _classify_message(justification, active_facts, client)
            for fid, hit in detected.items():
                if hit:
                    aid = dec.get("agent_id", -1)
                    if aid not in fact_records[fid]["integrated_by"]:
                        fact_records[fid]["integrated_by"].append(aid)

    # --- aggregate by category ---
    def _cat(fid: str) -> str:
        if fid in task.shared_fact_ids:
            return "shared"
        for fids in task.cluster_fact_ids.values():
            if fid in fids:
                return "cluster"
        if fid in task.bridge_agent_fact_ids.values():
            return "bridge"
        return "other"

    cats = ["shared", "cluster", "bridge"]
    summary: dict[str, Any] = {"per_fact": {}}

    for fid, rec in fact_records.items():
        summary["per_fact"][fid] = {
            "category": _cat(fid),
            "disclosed": rec["disclosed"],
            "transmitted": rec["transmitted"],
            "integrated_by": rec["integrated_by"],
        }

    for cat in cats:
        subset = [rec for fid, rec in fact_records.items() if _cat(fid) == cat]
        n = len(subset)
        if n == 0:
            summary[cat] = {"n": 0}
            continue
        summary[cat] = {
            "n": n,
            "disclosed": sum(r["disclosed"] for r in subset),
            "disclosed_rate": sum(r["disclosed"] for r in subset) / n,
            "transmitted": sum(r["transmitted"] for r in subset),
            "transmitted_rate": sum(r["transmitted"] for r in subset) / n,
            "integrated": sum(bool(r["integrated_by"]) for r in subset),
            "integrated_rate": sum(bool(r["integrated_by"]) for r in subset) / n,
        }

    overall = list(fact_records.values())
    n_all = len(overall)
    summary["overall"] = {
        "n": n_all,
        "disclosed_rate": sum(r["disclosed"] for r in overall) / n_all if n_all else 0,
        "transmitted_rate": sum(r["transmitted"] for r in overall) / n_all if n_all else 0,
        "integrated_rate": sum(bool(r["integrated_by"]) for r in overall) / n_all if n_all else 0,
    }

    (run_dir / "fact_classification.json").write_text(json.dumps(summary, indent=2))
    return summary
