[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/forge.svg)]

# FORGE — Failure-driven Orchestration for Rapid Generation of Experiments

An agentic system that autonomously builds, trains, debugs, and iterates Keras 3 models from natural-language task descriptions. No human in the loop.

## Quickstart

```bash
git clone https://github.com/habib-analyst/forge.git && cd forge
pip install -r requirements.txt && pip install torch   # or tensorflow / jax — one backend
python demo.py two_moons        # single NL task through the loop, verbose
python run_benchmark.py          # full 12-task benchmark (~30–60 min on CPU)
```

Example output (`python demo.py two_moons` — values from the measured run):

```
Task: Classify 2D two-moons points into 2 classes.
Success: True | iterations: 1 | first-attempt ok: True | final metric: 0.8188 (target 0.78)
Backend rationale: <chosen Keras 3 backend + logged reason for the pick>
```

## Measured results

12 diverse CPU-tiny tasks (image, tabular, text, sequence — including imbalanced and regression variants), ≤6 iterations per task. Reported against a **one-shot human-expert baseline** (fixed competent model, no loop):

| metric | value |
|---|---|
| tasks solved | **10/12 (83%)** |
| mean iterations to success | 1.5 |
| FORGE ≥ human baseline | 5/12 |
| unit tests | 14/14 |

Honest failures: `parity_sequence` exhausted all 6 iterations without reaching target. First-attempt success was 100% on the cold first half vs 67% on the later half — confounded by task difficulty (see `PAPER.md` §5), not clear evidence the failure memory helps. Multi-backend *selection* is implemented with rationale logging; cross-backend timing comparisons need JAX/TF installed (only torch was available here). Full per-task rows: `results.md`.

## How it works

```
"Classify 8x8 digit images" ──▶ Planner ──▶ Coder ──▶ Executor ──▶ Critic
                                     ▲                                  │
                                     └──────── fix directives ──────────┘
```

| agent | job |
|---|---|
| **Planner** | NL description → structured plan (model family, loss, metric, backend) with logged rationale; injects lessons from failure memory |
| **Coder** | plan → self-contained Keras 3 training script (shape probe, early stopping, gradient clipping) |
| **Executor** | runs the script in a subprocess under a fixed wall-time budget; returns metrics or the traceback |
| **Critic** | classifies failures (traceback signatures + metric pathologies: divergence, plateau, overfitting, majority collapse) → fix directives → records outcomes into the failure taxonomy |

The novelty: the critic distills every failure into a **failure taxonomy** (signature → fix strategy → measured fix rate) that the planner reads before planning. The planner and critic are deterministic and rule-based — no LLM API needed, $0. Tiny CPU tasks by design: the benchmark measures the agent loop, not scale.

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
