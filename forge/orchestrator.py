"""Orchestrator: the closed agentic loop.

Planner -> Coder -> Executor -> Critic -> (fix directives) -> Coder -> ...
No human in the loop. Bounded by max_iterations (the per-task compute
budget, together with the executor's wall-time limit).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .taxonomy import FailureTaxonomy
from .planner import Planner
from .coder import Coder
from .executor import Executor, RunResult
from .critic import Critic, Critique
from .tasks import Task


@dataclass
class AttemptRecord:
    iteration: int
    ok: bool
    metric: float | None
    seconds: float
    failure_name: str = ""
    directives: list = field(default_factory=list)


@dataclass
class TaskOutcome:
    task_name: str
    success: bool
    iterations: int
    final_metric: float | None
    first_attempt_ok: bool
    total_seconds: float
    attempts: list = field(default_factory=list)
    backend_rationale: str = ""


class Orchestrator:
    def __init__(self, taxonomy: FailureTaxonomy,
                 available_backends: list[str] | None = None,
                 python_bin: str | None = None,
                 timeout_s: int = 180, max_iterations: int = 6,
                 workdir=None):
        self.taxonomy = taxonomy
        self.planner = Planner(taxonomy, available_backends)
        self.coder = Coder()
        self.executor = Executor(python_bin=python_bin or __import__("sys").executable,
                                 timeout_s=timeout_s, workdir=workdir)
        self.critic = Critic(taxonomy)
        self.max_iterations = max_iterations

    def _succeeded(self, metric: float | None, task: Task) -> bool:
        if metric is None:
            return False
        return metric >= task.target if task.higher_is_better else metric <= task.target

    def run_task(self, task: Task) -> TaskOutcome:
        attempts: list[AttemptRecord] = []
        directives: list[str] = []
        first_ok = False
        total_s = 0.0
        backend_rationale = ""
        final_metric = None

        for it in range(1, self.max_iterations + 1):
            plan = self.planner.plan(task.name, task.description,
                                     max(task.n_classes, 1) if task.n_classes else 1,
                                     task.input_shape, extra_directives=directives)
            # regression: force regression head
            if task.n_classes == 0:
                plan.loss, plan.output_activation, plan.metric = "mse", "linear", "mae"
            backend_rationale = plan.backend_rationale
            script = self.coder.generate(plan, task.data_code)
            t0 = time.time()
            result: RunResult = self.executor.run(script)
            secs = time.time() - t0
            total_s += secs

            ncls = task.n_classes if task.n_classes else 1
            if self._succeeded(result.metric, task):
                if it == 1:
                    first_ok = True
                attempts.append(AttemptRecord(it, True, result.metric, secs))
                final_metric = result.metric
                break

            critique = self.critic.critique(result, ncls)
            if critique is None:
                # Ran fine but missed the target with no diagnosed pathology:
                # still a learning event for the taxonomy (it teaches the
                # planner that this task family needs more capacity).
                critique = Critique(
                    failure_name="below_target",
                    fix_strategy="Missed target without a diagnosed pathology; "
                                 "widen the model once, then stop escalating blindly.",
                    directives=["widen_model"] if it < 2 else [])
            fail_name = critique.failure_name
            dirs = critique.directives
            attempts.append(AttemptRecord(it, result.ok, result.metric, secs,
                                          fail_name, dirs))
            # learn() records the failure; fix-credit comes if a later attempt succeeds
            self.critic.learn(critique, fixed=False)
            if not dirs:
                # no actionable fix: one blind retry (fresh randomness) at most,
                # then stop escalating
                if it >= 2:
                    break
                dirs = ["widen_model"]
            directives = dirs
            # mark previous critique fixed if we eventually succeed (after loop)
        success = final_metric is not None and self._succeeded(final_metric, task)
        if success and len(attempts) > 1:
            # the last critique's fix worked — credit it without double-counting
            last_fail = attempts[-2].failure_name
            if last_fail:
                self.taxonomy.mark_fixed(last_fail)
        return TaskOutcome(task_name=task.name, success=success,
                           iterations=len(attempts), final_metric=final_metric,
                           first_attempt_ok=first_ok, total_seconds=round(total_s, 1),
                           attempts=attempts, backend_rationale=backend_rationale)
