"""Measure the one-shot human-expert baseline per task -> calibrate targets.

Writes calibration.json {task: baseline_metric}. Targets are then set to
baseline + margin (documented, honest: the benchmark measures the AGENT LOOP
vs a one-shot expert, not absolute SOTA).
"""

import json
import sys
from pathlib import Path

from forge.benchmark import human_baseline, EXPERT_CFG
from forge.coder import Coder
from forge.executor import Executor
from forge.tasks import TASKS


def main():
    out = Path(__file__).parent / "calibration.json"
    calib_dir = Path(__file__).parent / "calib_runs"
    calib_dir.mkdir(parents=True, exist_ok=True)
    coder = Coder()
    executor = Executor(sys.executable, timeout_s=240, workdir=calib_dir)
    results = {}
    for i, t in enumerate(TASKS, 1):
        print(f"[{i}/{len(TASKS)}] baseline: {t.name}", flush=True)
        m = human_baseline(t, coder, executor, ["torch"])
        results[t.name] = {"metric": m, "higher_is_better": t.higher_is_better,
                           "old_target": t.target}
        print(f"   -> {m}", flush=True)
        out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
