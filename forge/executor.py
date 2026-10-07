"""Executor agent: runs generated training scripts under a fixed compute budget.

Each run executes in a subprocess with a wall-time limit (the per-task
compute budget). Returns a RunResult: parsed metric JSON on success, or the
captured traceback / timeout signal on failure for the Critic.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunResult:
    ok: bool
    metric: float | None = None
    train_metric: float | None = None
    val_metric: float | None = None
    epochs_run: int = 0
    traceback: str = ""
    timed_out: bool = False
    seconds: float = 0.0
    raw_stdout: str = ""


class Executor:
    def __init__(self, python_bin: str = sys.executable, timeout_s: int = 180,
                 workdir: str | Path | None = None):
        self.python_bin = python_bin
        self.timeout_s = timeout_s
        self.workdir = Path(workdir) if workdir else Path(tempfile.gettempdir())

    def run(self, script: str) -> RunResult:
        import time
        path = self.workdir / "forge_run.py"
        path.write_text(script)
        start = time.time()
        try:
            proc = subprocess.run(
                [self.python_bin, str(path)], capture_output=True, text=True,
                timeout=self.timeout_s, cwd=str(self.workdir))
            secs = time.time() - start
            stdout, stderr = proc.stdout or "", proc.stderr or ""
            # last JSON-looking line is the result
            for line in reversed(stdout.strip().splitlines()):
                line = line.strip()
                if line.startswith("{") and '"status"' in line:
                    try:
                        d = json.loads(line)
                        if d.get("status") == "ok":
                            return RunResult(ok=True, metric=d.get("metric"),
                                             train_metric=d.get("train_metric"),
                                             val_metric=d.get("val_metric"),
                                             epochs_run=d.get("epochs_run", 0),
                                             seconds=secs, raw_stdout=stdout)
                    except json.JSONDecodeError:
                        continue
            return RunResult(ok=False, traceback=(stderr or stdout)[-4000:],
                             seconds=secs, raw_stdout=stdout)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return RunResult(ok=False, timed_out=True, seconds=self.timeout_s,
                             traceback=f"TIMEOUT after {self.timeout_s}s\n" + out[-2000:])
