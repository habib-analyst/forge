"""FORGE: Failure-driven Orchestration for Rapid Generation of Experiments.

An agentic system that autonomously builds, trains, debugs, and iterates
Keras 3 models from natural-language task descriptions — planner, coder,
executor, and critic in a closed loop, with a growing failure-taxonomy
memory that makes the planner better over time.
"""

from .taxonomy import FailureTaxonomy, FailureEntry
from .planner import Planner, Plan, BackendSelector
from .coder import Coder
from .executor import Executor, RunResult
from .critic import Critic, Critique
from .orchestrator import Orchestrator, TaskOutcome
from .tasks import TASKS, Task, get_task
from .benchmark import run_benchmark, markdown_report, human_baseline

__version__ = "0.1.0"
__all__ = ["FailureTaxonomy", "FailureEntry", "Planner", "Plan", "BackendSelector",
           "Coder", "Executor", "RunResult", "Critic", "Critique",
           "Orchestrator", "TaskOutcome", "TASKS", "Task", "get_task",
           "run_benchmark", "markdown_report", "human_baseline"]
