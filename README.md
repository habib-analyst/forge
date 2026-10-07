# FORGE — Failure-driven Orchestration for Rapid Generation of Experiments

An **agentic system** that autonomously builds, trains, debugs, and iterates
Keras 3 models from natural-language task descriptions. No human in the loop.

```
"Classify 8x8 digit images" ──▶ Planner ──▶ Coder ──▶ Executor ──▶ Critic
                                     ▲                                  │
                                     └──────── fix directives ──────────┘
```

## The agents

| Agent | Job |
|---|---|
| **Planner** | NL description → structured plan (model family, loss, metric, backend with recorded rationale). Injects lessons from failure memory. |
| **Coder** | Plan → self-contained Keras 3 training script (shape probe, early stopping, gradient clipping). |
| **Executor** | Runs the script in a subprocess under a fixed wall-time budget; returns metrics or the traceback. |
| **Critic** | Classifies failures (traceback signatures + metric pathologies: divergence, plateau, overfitting, majority collapse) → fix directives → records outcomes into the taxonomy. |

## The novelty: the system learns to plan

The critic distills every failure into a **failure taxonomy** (signature →
fix strategy → measured fix rate). The planner reads this memory before
planning. Measured on the 12-task benchmark: **first-attempt success rate,
tasks 1–6 (cold) vs 7–12 (learned)** — the planner demonstrably improves as
the taxonomy grows. A `BackendSelector` scores Keras 3 backends (installed /
task-fit / CPU maturity) with a logged rationale per task.

## Benchmark

12 diverse CPU-tiny tasks (image, tabular, text, sequence — including
imbalanced and regression variants), fixed compute budget per task
(≤6 iterations, ≤600 s per training run). Reported per task: success,
iterations to success, first-attempt success, final metric **vs a one-shot
human-expert baseline** (fixed competent model, no loop). Numbers only —
see `results.md` after a run.

## Quickstart

```bash
pip install -r requirements.txt
pip install torch            # or tensorflow / jax — one backend
python run_benchmark.py      # ~30-60 min on CPU; writes results.json/.md
python demo.py               # single task through the loop, verbose
```

## Layout

```
forge/            the four agents + orchestrator + task suite + benchmark
tests/            unit tests (no training required)
run_benchmark.py  full 12-task benchmark
demo.py           one NL task through the loop
```

## Honest limits

- Planner/critic are deterministic and rule-based (no LLM API needed, $0).
- Multi-backend *selection* is implemented and rationale-logged; cross-backend
  timing comparisons require JAX/TF installed (only torch was available here).
- Tiny CPU tasks by design: the benchmark measures the agent loop, not scale.

MIT License.
