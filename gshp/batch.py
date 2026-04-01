"""
Batch / factorial runs: JSON config, ``progress.json`` + ``index.csv``, full per-run artifacts.

Set ``"concurrent_runs": N`` in the batch config to run N experiments in parallel.
Each concurrent run gets its own LLM client instance — vLLM handles the concurrent
HTTP connections on the server side. Progress is written after every completed run
so partial batches are recoverable with ``--resume``.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import re
import threading
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

from gshp.batch_errors import classify_error
from gshp.artifacts import write_run_bundle
from gshp.experiment import run_hidden_profile_hiring
from gshp.graph.caveman import CavemanTopology
from gshp.llm.logging_client import LoggingLLMClient
from gshp.llm.openai_local import make_llm_client
from gshp.llm.stub_client import StubLLM
from gshp.schedule import ScheduleName
from gshp.task.hiring import InformationCondition, build_default_hiring_task
from gshp.task.generator import load_task


def _slug(s: str) -> str:
    s = s.replace("/", "_")
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")[:120] or "x"


def cell_slug(params: dict[str, Any]) -> str:
    parts = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, bool):
            v = "1" if v else "0"
        parts.append(f"{_slug(str(k))}={_slug(str(v))}")
    return "__".join(parts)


def load_batch_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "base_dir" not in data:
        raise ValueError("batch config requires string field: base_dir")
    if "grid" not in data or not isinstance(data["grid"], dict):
        raise ValueError("batch config requires object field: grid (keys -> list of values)")
    return data


# ---------------------------------------------------------------------------
# Single-run worker (called from thread pool or directly)
# ---------------------------------------------------------------------------


def _execute_one_run(
    *,
    run_dir: Path,
    cell_params: dict[str, Any],
    cell_id: str,
    seed: int,
    cfg: dict[str, Any],
    task_template: Any,
    topo_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute one complete experiment run and write its artifact bundle.
    Returns a status row dict. Thread-safe: uses only local state + its own run_dir.
    """
    l = int(topo_cfg.get("l", 3))
    k = int(topo_cfg.get("k", 3))
    kind = str(topo_cfg.get("kind", "full_clique_ring"))
    dyad_turns = int(cfg.get("dyad_turns", 6))
    parallel_dyad_layers_default = bool(cfg.get("parallel_dyad_layers", False))
    stub = bool(cfg.get("stub", False))
    stub_final = str(cfg.get("stub_final", "X")).upper()
    model = str(cfg.get("model", "") or "")
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens: int | None = cfg.get("max_tokens")
    if max_tokens is not None:
        max_tokens = int(max_tokens)
    timeout = float(cfg.get("timeout", 120.0))
    max_retries_client = int(cfg.get("max_retries", 3))

    row: dict[str, Any] = {
        "cell_id": cell_id,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "status": "",
    }
    for pk, pv in cell_params.items():
        row[f"grid_{pk}"] = pv

    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        topo = CavemanTopology.build(l=l, k=k, kind=kind)  # type: ignore[arg-type]

        sched_raw = cell_params.get("schedule", ScheduleName.WITHIN_FIRST.value)
        schedule_name = ScheduleName(sched_raw) if isinstance(sched_raw, str) else sched_raw

        cond_raw = cell_params.get("condition", InformationCondition.HIDDEN_PROFILE.value)
        condition = InformationCondition(cond_raw) if isinstance(cond_raw, str) else cond_raw

        tom_bridge = bool(cell_params.get("tom_bridge", False))
        parallel_dyad_layers = bool(
            cell_params.get("parallel_dyad_layers", parallel_dyad_layers_default)
        )

        # Each run gets its own client — vLLM handles concurrent HTTP on the server side
        if stub:
            base_client: Any = StubLLM(final_choice=stub_final)
            model_label = f"stub:{stub_final}"
        else:
            if not model:
                raise ValueError('Set "model" in batch config or use "stub": true')
            base_client = make_llm_client(
                model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries_client,
            )
            model_label = model

        client = LoggingLLMClient(base_client, capture_raw_completion=True)

        exp_run = run_hidden_profile_hiring(
            topo,
            schedule_name,
            task_template,
            client,
            condition=condition,
            tom_bridge=tom_bridge,
            dyad_turns=dyad_turns,
            model_label=model_label,
            seed=seed,
            parallel_dyad_layers=parallel_dyad_layers,
            max_workers=int(cfg.get("max_workers", 1)),
            group_deliberation=bool(cfg.get("group_deliberation", False)),
        )

        write_run_bundle(
            run_dir,
            run=exp_run,
            task=task_template,
            topo=topo,
            llm_calls=client.calls,
            dyad_turns=dyad_turns,
            tom_bridge=tom_bridge,
            extra_config={
                "batch_cell": cell_params,
                "batch_seed": seed,
                "capture_raw_completion": True,
            },
        )

        row["status"] = "ok"
        row["accuracy"] = exp_run.notes.get("accuracy_agent_level")
        row["majority"] = exp_run.notes.get("majority_vote")
        row["unanimous_correct"] = exp_run.notes.get("unanimous_correct")
        if exp_run.notes.get("group_deliberation"):
            row["group_consensus"] = exp_run.notes.get("group_consensus")
            row["group_accuracy"] = exp_run.notes.get("group_accuracy")

    except Exception as e:
        etype = classify_error(e)
        row["status"] = "failed"
        row["error_type"] = etype
        row["error"] = str(e)[:2000]
        try:
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass

    return row


# ---------------------------------------------------------------------------
# Main batch orchestrator
# ---------------------------------------------------------------------------


def run_batch_from_config(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """
    Execute all Cartesian products of ``grid`` lists; ``runs_per_cell`` replicates per cell.

    Set ``"concurrent_runs": N`` in the config to run N experiments simultaneously.
    Each concurrent run has its own LLM client — safe to parallelize against vLLM.

    Returns the resolved ``base_dir``.
    """
    p = Path(config_path)
    cfg = load_batch_config(p)
    if overrides:
        cfg.update(overrides)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(cfg["base_dir"]) / ts
    base_dir.mkdir(parents=True, exist_ok=True)

    resolved = dict(cfg)
    resolved["_loaded_from"] = str(p.resolve())
    resolved["_started_at"] = datetime.now().isoformat(timespec="milliseconds")
    (base_dir / "batch_config.json").write_text(
        json.dumps(resolved, indent=2, default=str), encoding="utf-8"
    )

    grid: dict[str, list[Any]] = cfg["grid"]
    keys = sorted(grid.keys())
    value_lists = [grid[k] for k in keys]
    cells = [dict(zip(keys, combo, strict=True)) for combo in product(*value_lists)]

    runs_per_cell = int(cfg.get("runs_per_cell", 1))
    seed_base = int(cfg.get("seed_base", 0))
    increment_seed = bool(cfg.get("increment_seed_per_run", True))
    concurrent_runs = int(cfg.get("concurrent_runs", 1))

    topo_cfg = cfg.get("topology", {})

    # Load task: task_file in config overrides default hiring task
    task_file = cfg.get("task_file", "")
    if task_file:
        task_template = load_task(task_file)
        print(f"Loaded task from {task_file}: domain='{getattr(task_template, 'domain', 'unknown')}'")
    else:
        task_template = build_default_hiring_task()

    # Build the full work list (all runs across all cells)
    work_items: list[dict[str, Any]] = []
    run_global = 0
    for cell_params in cells:
        cid = cell_slug(cell_params)
        cell_dir = base_dir / cid
        for rep in range(runs_per_cell):
            run_idx = rep + 1
            seed = seed_base + (run_global if increment_seed else rep)
            run_dir = cell_dir / f"run_{run_idx:03d}"
            work_items.append({
                "cell_id": cid,
                "cell_dir": cell_dir,
                "run_dir": run_dir,
                "run_index": run_idx,
                "seed": seed,
                "cell_params": cell_params,
                "run_global": run_global,
            })
            run_global += 1

    total = len(work_items)
    progress: dict[str, Any] = {
        "started_at": resolved["_started_at"],
        "config_path": str(p.resolve()),
        "base_dir": str(base_dir.resolve()),
        "total_planned": total,
        "concurrent_runs": concurrent_runs,
        "completed": [],
        "failed": [],
        "skipped": [],
    }
    progress_path = base_dir / "progress.json"
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    # Dry run: just record the plan
    if dry_run:
        index_rows = []
        for w in work_items:
            row = {
                "cell_id": w["cell_id"],
                "run_index": w["run_index"],
                "seed": w["seed"],
                "status": "dry_run",
                "run_dir": str(w["run_dir"].resolve()),
            }
            for pk, pv in w["cell_params"].items():
                row[f"grid_{pk}"] = pv
            index_rows.append(row)
        _write_index_csv(base_dir, index_rows)
        progress["finished_at"] = datetime.now().isoformat(timespec="milliseconds")
        progress_path.write_text(json.dumps(progress, indent=2, default=str), encoding="utf-8")
        return base_dir

    # Shared state for concurrent progress tracking
    lock = threading.Lock()
    index_rows: list[dict[str, Any]] = []

    def _handle_work_item(w: dict[str, Any]) -> dict[str, Any]:
        """Called once per run — either skipped, or executed."""
        run_dir: Path = w["run_dir"]
        cell_id: str = w["cell_id"]
        seed: int = w["seed"]
        run_idx: int = w["run_index"]

        if resume and (run_dir / "summary.json").exists():
            row = {
                "cell_id": cell_id,
                "run_index": run_idx,
                "seed": seed,
                "status": "skipped_resume",
                "run_dir": str(run_dir.resolve()),
            }
            for pk, pv in w["cell_params"].items():
                row[f"grid_{pk}"] = pv
            with lock:
                progress["skipped"].append({
                    "cell_id": cell_id,
                    "run_dir": str(run_dir.resolve()),
                    "seed": seed,
                    "reason": "summary.json exists",
                })
                _flush_progress(progress, progress_path)
            return row

        row = _execute_one_run(
            run_dir=run_dir,
            cell_params=w["cell_params"],
            cell_id=cell_id,
            seed=seed,
            cfg=cfg,
            task_template=task_template,
            topo_cfg=topo_cfg,
        )
        row["run_index"] = run_idx

        with lock:
            if row["status"] == "ok":
                progress["completed"].append({
                    "cell_id": cell_id,
                    "run_dir": str(run_dir.resolve()),
                    "seed": seed,
                    "status": "ok",
                    "accuracy": row.get("accuracy"),
                    "majority": row.get("majority"),
                })
            else:
                progress["failed"].append({
                    "cell_id": cell_id,
                    "run_dir": str(run_dir.resolve()),
                    "seed": seed,
                    "error_type": row.get("error_type"),
                    "error": row.get("error"),
                })
            n_done = len(progress["completed"]) + len(progress["failed"]) + len(progress["skipped"])
            print(f"[{n_done}/{total}] {cell_id}/run_{run_idx:03d} → {row['status']}"
                  + (f" accuracy={row['accuracy']:.2f}" if row.get("accuracy") is not None else ""))
            _flush_progress(progress, progress_path)

        return row

    if concurrent_runs > 1:
        print(f"Running {total} experiments with {concurrent_runs} concurrent workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_runs) as pool:
            futures = [pool.submit(_handle_work_item, w) for w in work_items]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    row = fut.result()
                    with lock:
                        index_rows.append(row)
                except Exception as e:
                    print(f"Unexpected worker error: {e}")
    else:
        for w in work_items:
            row = _handle_work_item(w)
            index_rows.append(row)

    _write_index_csv(base_dir, index_rows)
    progress["finished_at"] = datetime.now().isoformat(timespec="milliseconds")
    _flush_progress(progress, progress_path)

    n_ok = len(progress["completed"])
    n_fail = len(progress["failed"])
    n_skip = len(progress["skipped"])
    print(f"\nDone. {n_ok} completed, {n_fail} failed, {n_skip} skipped → {base_dir}")
    return base_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flush_progress(progress: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(progress, indent=2, default=str), encoding="utf-8")


def _write_index_csv(base_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r})
    with (base_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
