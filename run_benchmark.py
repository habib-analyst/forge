"""Run the full 12-task FORGE benchmark. Writes results.json + results.md."""

import sys
from pathlib import Path

from forge.benchmark import run_benchmark, markdown_report
from forge.tasks import TASKS


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
    backends = detect_backends()
    print(f"Available Keras backends: {backends}")
    workdir = Path(__file__).parent / "benchmark_runs"
    report = run_benchmark(workdir, sys.executable, backends,
                           timeout_s=600, max_iterations=6)
    (workdir / "results.md").write_text(markdown_report(report))
    print(markdown_report(report))
    print(f"\nWrote {workdir / 'results.json'} and results.md")
