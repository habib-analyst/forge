"""Unit tests for FORGE — no training, no network required."""

import json
import tempfile
from pathlib import Path

import pytest

from forge.taxonomy import FailureTaxonomy, FailureEntry
from forge.planner import Planner, BackendSelector, HEAD_PRESETS
from forge.coder import Coder
from forge.critic import Critic, diagnose_metrics
from forge.executor import RunResult
from forge.tasks import TASKS, get_task
from forge.orchestrator import Orchestrator


def test_taxonomy_match_and_learn(tmp_path):
    tax = FailureTaxonomy(tmp_path / "t.json")
    e = tax.match_traceback("ValueError: Shapes (None, 10) and (None, 8) are incompatible")
    assert e is not None and e.name == "shape_mismatch"
    tax.record("shape_mismatch", fixed=True)
    assert tax.fix_rate_summary()["shape_mismatch"] == 1.0
    # persistence round-trip
    tax2 = FailureTaxonomy(tmp_path / "t.json")
    assert tax2.fix_rate_summary()["shape_mismatch"] == 1.0


def test_taxonomy_grows_on_unknown(tmp_path):
    tax = FailureTaxonomy(tmp_path / "t.json")
    n0 = len(tax.entries)
    tax.record("unknown:weird new error", fixed=False)
    assert len(tax.entries) == n0 + 1


def test_planner_families():
    tax = FailureTaxonomy()
    p = Planner(tax, ["torch"])
    assert p.detect_family("Classify 8x8 digit images") == "image_classification"
    assert p.detect_family("predict house price regression") == "tabular_regression"
    assert p.detect_family("sentiment of tweets") == "text_classification"
    plan = p.plan("t", "Classify 8x8 digit images", 10, (8, 8, 1))
    assert plan.model_family == "cnn_small"
    assert (plan.output_activation, plan.loss) == ("softmax", "sparse_categorical_crossentropy")
    assert plan.backend == "torch"
    assert "scores=" in plan.backend_rationale


def test_backend_selector_logs_rationale():
    sel = BackendSelector()
    b, r = sel.select("image_classification", ["torch"])
    assert b == "torch" and "scores=" in r and "available=" in r


def test_coder_generates_runnable_python():
    tax = FailureTaxonomy()
    plan = Planner(tax, ["torch"]).plan("t", "Classify iris flowers", 3, (4,))
    code = Coder().generate(plan, "X_train = 1")
    compile(code, "<gen>", "exec")  # must be valid Python
    assert "KERAS_BACKEND" in code and "sparse_categorical_crossentropy" in code


def test_coder_directives():
    tax = FailureTaxonomy()
    planner = Planner(tax, ["torch"])
    plan = planner.plan("t", "Classify iris flowers", 3, (4,), extra_directives=["lower_lr", "widen_model"])
    code = Coder().generate(plan, "X_train = 1")
    assert "learning_rate=0.0001" in code
    assert "Dense(128" in code  # widened from 64


def test_shrink_model_directive():
    tax = FailureTaxonomy()
    plan = Planner(tax, ["torch"]).plan("t", "Classify 8x8 digit images", 10, (8, 8, 1),
                                        extra_directives=["shrink_model"])
    code = Coder().generate(plan, "X_train = 1")
    compile(code, "<gen>", "exec")
    assert "Conv2D(16" in code  # filters halved from [32, 64] -> [16, 32][:2]


def test_critic_traceback_match():
    tax = FailureTaxonomy()
    crit = Critic(tax)
    r = RunResult(ok=False, traceback="ResourceExhaustedError: OOM when allocating tensor")
    c = crit.critique(r, 3)
    assert c is not None and c.failure_name == "oom_or_resource"
    assert "smaller_batch" in c.directives


def test_critic_metric_pathologies():
    tax = FailureTaxonomy()
    crit = Critic(tax)
    # divergence
    r = RunResult(ok=True, metric=float("nan"), train_metric=0.1, val_metric=0.1, epochs_run=6)
    assert crit.critique(r, 3).failure_name == "nan_loss_divergence"
    # majority collapse on imbalanced task
    r = RunResult(ok=True, metric=0.5, train_metric=0.9, val_metric=0.51, epochs_run=6)
    assert crit.critique(r, 2).failure_name == "class_imbalance_ignored"
    # healthy run -> no critique
    r = RunResult(ok=True, metric=0.95, train_metric=0.97, val_metric=0.95, epochs_run=8)
    assert crit.critique(r, 10) is None


def test_orchestrator_success_path():
    """Orchestrator with a stub executor: succeeds first try."""
    tax = FailureTaxonomy()
    orch = Orchestrator(tax, ["torch"], max_iterations=3)

    class StubExec:
        def run(self, script):
            return RunResult(ok=True, metric=0.96, train_metric=0.97,
                             val_metric=0.96, epochs_run=5, seconds=1.0)
    orch.executor = StubExec()
    outcome = orch.run_task(get_task("two_moons"))
    assert outcome.success and outcome.iterations == 1 and outcome.first_attempt_ok


def test_orchestrator_retry_then_success():
    """Fails once (shape mismatch), critic directs retry, succeeds."""
    tax = FailureTaxonomy()
    orch = Orchestrator(tax, ["torch"], max_iterations=3)
    calls = {"n": 0}

    class StubExec:
        def run(self, script):
            calls["n"] += 1
            if calls["n"] == 1:
                return RunResult(ok=False, traceback="Shapes (None,10) incompatible", seconds=1.0)
            return RunResult(ok=True, metric=0.96, train_metric=0.97,
                             val_metric=0.96, epochs_run=5, seconds=1.0)
    orch.executor = StubExec()
    outcome = orch.run_task(get_task("two_moons"))
    assert outcome.success and outcome.iterations == 2
    assert tax.fix_rate_summary().get("shape_mismatch") == 1.0


def test_below_target_teaches_taxonomy(tmp_path):
    """A below-target run with no pathology still records a taxonomy lesson."""
    tax = FailureTaxonomy(tmp_path / "t.json")
    orch = Orchestrator(tax, ["torch"], max_iterations=3)

    class StubExec:
        def run(self, script):
            # always "ok" but below the 0.95 target, healthy dynamics
            return RunResult(ok=True, metric=0.70, train_metric=0.72,
                             val_metric=0.70, epochs_run=8, seconds=1.0)
    orch.executor = StubExec()
    outcome = orch.run_task(get_task("two_moons"))
    assert not outcome.success
    # 0.70 with 8 epochs is diagnosed as metric_plateau (val < 0.75, stalled)
    assert tax.fix_rate_summary().get("metric_plateau") == 0.0  # seen, never fixed
    assert any(a.failure_name == "metric_plateau" for a in outcome.attempts)


def test_below_target_without_pathology_teaches_taxonomy(tmp_path):
    """A below-target run the critic can't diagnose still records a lesson."""
    tax = FailureTaxonomy(tmp_path / "t.json")
    orch = Orchestrator(tax, ["torch"], max_iterations=3)

    class StubExec:
        def run(self, script):
            # healthy-looking but below the 0.78 target: val 0.77 avoids the
            # plateau rule (>= 0.75), small gap avoids overfitting rule
            return RunResult(ok=True, metric=0.76, train_metric=0.78,
                             val_metric=0.77, epochs_run=8, seconds=1.0)
    orch.executor = StubExec()
    outcome = orch.run_task(get_task("two_moons"))
    assert not outcome.success
    assert tax.fix_rate_summary().get("below_target") == 0.0  # seen, never fixed
    assert any(a.failure_name == "below_target" for a in outcome.attempts)


def test_tasks_all_valid():
    assert len(TASKS) == 12
    names = [t.name for t in TASKS]
    assert len(set(names)) == 12
    for t in TASKS:
        compile(t.data_code, f"<{t.name}>", "exec")
        assert isinstance(t.input_shape, tuple)
