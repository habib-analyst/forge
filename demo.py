"""One NL task through the FORGE loop, verbose. Usage: python demo.py [task_name]"""

import sys
from pathlib import Path

from forge.taxonomy import FailureTaxonomy
from forge.orchestrator import Orchestrator
from forge.tasks import get_task, TASKS


def detect_backends():
    avail = []
    for name, mod in [("jax", "jax"), ("tensorflow", "tensorflow"), ("torch", "torch")]:
        try:
            __import__(mod)
            avail.append(name)
        except ImportError:
            pass
    return avail or ["numpy"]


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "two_moons"
    task = get_task(name)
    workdir = Path(__file__).parent / "demo_runs"
    workdir.mkdir(exist_ok=True)
    taxonomy = FailureTaxonomy(workdir / "taxonomy.json")
    orch = Orchestrator(taxonomy, detect_backends(), sys.executable,
                        timeout_s=180, max_iterations=6, workdir=workdir)
    print(f"Task: {task.description}")
    outcome = orch.run_task(task)
    print(f"\nSuccess: {outcome.success} | iterations: {outcome.iterations} | "
          f"first-attempt ok: {outcome.first_attempt_ok} | "
          f"final metric: {outcome.final_metric} (target {task.target})")
    for a in outcome.attempts:
        print(f"  iter {a.iteration}: ok={a.ok} metric={a.metric} "
              f"failure={a.failure_name or '-'} directives={a.directives} "
              f"({a.seconds:.0f}s)")
    print(f"Backend rationale: {outcome.backend_rationale}")
