"""Batch runner for the collective reasoning task (2×2×N factorial)."""

from __future__ import annotations

import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

from gshp.experiment_reasoning import run_reasoning_experiment
from gshp.graph.caveman import CavemanTopology
from gshp.llm.logging_client import LoggingLLMClient
from gshp.llm.openai_local import make_llm_client
from gshp.task.reasoning import ReasoningTaskSpec, build_default_reasoning_task


DEFAULT_CONFIG: dict[str, Any] = {
    "model": "",
    "grid": {
        "problem": ["bayes_factory", "conditional_balls", "expected_value_dice"],
        "schedule": ["within_first", "cross_first"],
        "deliberation": [True, False],
    },
    "runs_per_cell": 25,
    "concurrent_runs": 4,
    "dyad_turns": 6,
    "temperature": 0.0,
}


def _cell_id(problem: str, schedule: str, deliberation: bool) -> str:
    d = "deliberation" if deliberation else "no_deliberation"
    return f"problem={problem}__schedule={schedule}__{d}"


def run_reasoning_batch(
    config_path: str | Path | None = None,
    *,
    resume: bool = False,
    overrides: dict[str, Any] | None = None,
) -> Path:
    cfg = dict(DEFAULT_CONFIG)
    if config_path:
        cfg.update(json.loads(Path(config_path).read_text()))
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})

    model = cfg["model"]
    if not model:
        raise ValueError("Provide 'model' in config or via --model")

    grid = cfg["grid"]
    runs_per_cell: int = cfg["runs_per_cell"]
    concurrent: int = cfg["concurrent_runs"]
    dyad_turns: int = cfg.get("dyad_turns", 6)
    temperature: float = cfg.get("temperature", 0.0)

    import re
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model.replace("://", "_").replace("/", "_").replace(":", "_")).strip("_")[:80] or "model"

    base_dir = None
    if resume:
        candidates = sorted(Path("results").glob(f"reasoning_{slug}_*"), reverse=True)
        base_dir = next((c for c in candidates if (c / "progress.json").exists()), None)
    if base_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path("results") / f"reasoning_{slug}_{ts}"
    base_dir.mkdir(parents=True, exist_ok=True)

    topo = CavemanTopology.build(l=3, k=3)
    task = build_default_reasoning_task()

    cells = list(product(
        grid.get("problem", ["bayes_factory"]),
        grid.get("schedule", ["within_first", "cross_first"]),
        grid.get("deliberation", [False, True]),
    ))

    index_rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    index_path = base_dir / "index.csv"
    progress_path = base_dir / "progress.json"

    completed_keys: set[str] = set()
    if resume and progress_path.exists():
        prog = json.loads(progress_path.read_text())
        completed_keys = set(prog.get("completed", []))

    def _run_one(problem: str, schedule: str, deliberation: bool, run_idx: int) -> dict[str, Any]:
        cell = _cell_id(problem, schedule, deliberation)
        run_key = f"{cell}/run_{run_idx:03d}"
        run_dir = base_dir / cell / f"run_{run_idx:03d}"

        if resume and run_key in completed_keys:
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                s = json.loads(summary_path.read_text())
                return {"status": "skipped", **s}

        run_dir.mkdir(parents=True, exist_ok=True)
        base_client = make_llm_client(model, temperature=temperature)
        client = LoggingLLMClient(base_client, capture_raw_completion=True)

        try:
            run = run_reasoning_experiment(
                topo,
                schedule,
                task,
                problem,
                client,
                dyad_turns=dyad_turns,
                model_label=model,
                seed=run_idx,
                group_deliberation=deliberation,
            )
            (run_dir / "run.json").write_text(json.dumps(run.model_dump(mode="json"), indent=2))
            (run_dir / "llm_calls.json").write_text(json.dumps(client.calls, indent=2))
            summary = dict(run.notes)
            summary.update({"model": model, "schedule": schedule, "problem": problem,
                            "deliberation": deliberation, "run_index": run_idx, "status": "ok"})
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

            with lock:
                completed_keys.add(run_key)
                progress_path.write_text(json.dumps({"completed": sorted(completed_keys)}, indent=2))

            return summary
        except Exception as exc:
            return {"status": "error", "error": str(exc), "problem": problem,
                    "schedule": schedule, "deliberation": deliberation, "run_index": run_idx}

    all_jobs = [
        (problem, schedule, deliberation, idx)
        for problem, schedule, deliberation in cells
        for idx in range(runs_per_cell)
    ]
    total = len(all_jobs)
    done = 0

    print(f"Reasoning batch: {len(cells)} cells × {runs_per_cell} runs = {total} total", flush=True)
    print(f"Model: {model}  concurrent: {concurrent}", flush=True)
    print(f"Output: {base_dir}", flush=True)

    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        futures = {pool.submit(_run_one, *job): job for job in all_jobs}
        for fut in as_completed(futures):
            done += 1
            row = fut.result()
            with lock:
                index_rows.append(row)
            problem = row.get("problem", "?")
            schedule = row.get("schedule", "?")
            delib = row.get("deliberation", "?")
            acc = row.get("accuracy_agent_level", "?")
            mc = row.get("majority_correct", "?")
            gc = row.get("group_consensus_correct", "?") if row.get("deliberation") else "-"
            status = row.get("status", "?")
            print(
                f"  [{done}/{total}] {problem} {schedule} delib={delib} "
                f"acc={acc} majority_correct={mc} group_correct={gc} [{status}]",
                flush=True,
            )
            # write index.csv incrementally — union of all keys seen so far, not just
            # the first row's (deliberation=True rows have an extra field, which
            # otherwise breaks DictWriter partway through the batch)
            with lock:
                all_keys: set[str] = set()
                for r in index_rows:
                    all_keys.update(r.keys())
                with open(index_path, "w", newline="") as f:
                    if index_rows:
                        writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                        writer.writeheader()
                        writer.writerows(index_rows)

    print(f"\nDone. Results: {base_dir}", flush=True)
    return base_dir
