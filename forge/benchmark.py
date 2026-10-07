"""FORGE benchmark runner.

Runs all 12 tasks through the agentic loop (shared taxonomy memory across
tasks, in fixed order), plus a one-shot 'human expert' baseline per task:
a fixed, competent hand-designed model with no critic loop.

Headline measurements:
- success rate, mean iterations-to-success
- first-attempt success: tasks 1-6 (cold taxonomy) vs 7-12 (learned taxonomy)
  -> the novelty claim: the planner gets better as the critic's memory grows
- FORGE final metric vs human baseline per task
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .taxonomy import FailureTaxonomy
from .planner import Planner, Plan
from .coder import Coder
from .executor import Executor
from .orchestrator import Orchestrator
from .tasks import TASKS, Task

# Fixed "competent human" configs: what a practitioner would write in one shot.
EXPERT_CFG = {
    "cnn_small": {"filters": [32, 64, 128], "dense": 256, "dropout": 0.3},
    "mlp": {"hidden": [128, 64], "dropout": 0.3},
    "embedding_rnn": {"embedding_dim": 128, "rnn_units": 128, "dropout": 0.3,
                      "vocab_size": 2000},
}


def human_baseline(task: Task, coder: Coder, executor: Executor,
                   available_backends: list[str]) -> float | None:
    planner = Planner(FailureTaxonomy(), available_backends)  # no memory
    ncls = task.n_classes if task.n_classes else 1
    plan = planner.plan(task.name, task.description, ncls, task.input_shape)
    if task.n_classes == 0:
        plan.loss, plan.output_activation, plan.metric = "mse", "linear", "mae"
    plan.model_config.update(EXPERT_CFG.get(plan.model_family, {}))
    plan.max_epochs = 12
    script = coder.generate(plan, task.data_code)
    result = executor.run(script)
    return result.metric


def run_benchmark(workdir: str | Path, python_bin: str,
                  available_backends: list[str],
                  timeout_s: int = 600, max_iterations: int = 6,
                  task_names: list[str] | None = None) -> dict:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    taxonomy = FailureTaxonomy(workdir / "taxonomy.json")
    orch = Orchestrator(taxonomy, available_backends, python_bin,
                        timeout_s, max_iterations, workdir)
    coder = Coder()
    executor = Executor(python_bin, timeout_s, workdir)

    tasks = [t for t in TASKS if task_names is None or t.name in task_names]
    rows = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] FORGE loop: {task.name}", flush=True)
        outcome = orch.run_task(task)
        print(f"[{i}/{len(tasks)}] human baseline: {task.name}", flush=True)
        base = human_baseline(task, coder, executor, available_backends)
        rows.append({
            "task": task.name, "family": outcome.attempts and task.description[:40],
            "forge_success": outcome.success,
            "forge_iterations": outcome.iterations,
            "forge_first_attempt_ok": outcome.first_attempt_ok,
            "forge_metric": outcome.final_metric,
            "baseline_metric": base,
            "target": task.target,
            "higher_is_better": task.higher_is_better,
            "seconds": outcome.total_seconds,
            "attempts": [asdict(a) for a in outcome.attempts],
        })
        (workdir / "results_partial.json").write_text(json.dumps(rows, indent=2))

    half = len(rows) // 2
    first_half = [r for r in rows[:half] if r["forge_first_attempt_ok"] is not None]
    early_rate = sum(r["forge_first_attempt_ok"] for r in rows[:half]) / max(half, 1)
    late_rate = sum(r["forge_first_attempt_ok"] for r in rows[half:]) / max(len(rows) - half, 1)
    summary = {
        "n_tasks": len(rows),
        "success_rate": sum(r["forge_success"] for r in rows) / len(rows),
        "mean_iterations": sum(r["forge_iterations"] for r in rows) / len(rows),
        "first_attempt_success_early": round(early_rate, 3),
        "first_attempt_success_late": round(late_rate, 3),
        "planner_improvement_pp": round((late_rate - early_rate) * 100, 1),
        "forge_beats_or_ties_baseline": sum(
            1 for r in rows
            if r["forge_metric"] is not None and r["baseline_metric"] is not None
            and (r["forge_metric"] >= r["baseline_metric"]
                 if r["higher_is_better"] else
                 r["forge_metric"] <= r["baseline_metric"])),
        "taxonomy_fix_rates": taxonomy.fix_rate_summary(),
        "total_seconds": sum(r["seconds"] for r in rows),
    }
    report = {"summary": summary, "tasks": rows,
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (workdir / "results.json").write_text(json.dumps(report, indent=2))
    return report


def markdown_report(report: dict) -> str:
    s = report["summary"]
    lines = ["# FORGE benchmark results", "",
             f"Tasks: {s['n_tasks']} | Success rate: {s['success_rate']:.0%} | "
             f"Mean iterations: {s['mean_iterations']:.1f}", "",
             f"First-attempt success — early half: {s['first_attempt_success_early']:.0%}, "
             f"late half: {s['first_attempt_success_late']:.0%} "
             f"(Δ {s['planner_improvement_pp']:+.1f} pp — confounded by task difficulty; "
             f"see PAPER.md §5)",
             "",
             f"FORGE ≥ human baseline on {s['forge_beats_or_ties_baseline']}/{s['n_tasks']} tasks",
             "", "| task | forge iters | 1st-try | forge metric | baseline | target |",
             "|---|---|---|---|---|---|"]
    for r in report["tasks"]:
        fm = f"{r['forge_metric']:.4f}" if r['forge_metric'] is not None else "—"
        bm = f"{r['baseline_metric']:.4f}" if r['baseline_metric'] is not None else "—"
        lines.append(f"| {r['task']} | {r['forge_iterations']} | "
                     f"{'yes' if r['forge_first_attempt_ok'] else 'no'} | {fm} | {bm} | {r['target']} |")
    lines += ["", "## Failure taxonomy fix rates",
              json.dumps(s["taxonomy_fix_rates"], indent=2)]
    return "\n".join(lines)
