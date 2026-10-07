"""Critic agent: reads failures/metrics -> classifies -> proposes fixes.

Two failure channels:
1. Traceback -> matched against the failure taxonomy signature -> fix strategy.
2. Metric pathology (success exit but bad learning dynamics) -> pattern rules:
   divergence / plateau / overfitting / majority_collapse.

Every classification is recorded back into the taxonomy (times_seen/fixed),
so the taxonomy — and therefore future plans — improves over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from .taxonomy import FailureTaxonomy
from .executor import RunResult


@dataclass
class Critique:
    failure_name: str
    fix_strategy: str
    directives: list[str] = field(default_factory=list)  # coder directives
    learned: bool = False


# directive mapping: failure -> concrete code transformations for the coder
DIRECTIVES = {
    "shape_mismatch": [],
    "channel_mismatch": [],
    "oom_or_resource": ["smaller_batch"],
    "nan_loss_divergence": ["lower_lr"],
    "label_loss_mismatch": [],
    "metric_plateau": ["widen_model", "more_epochs"],
    "severe_overfitting": ["more_regularization"],
    "class_imbalance_ignored": ["add_class_weight"],
}


def diagnose_metrics(result: RunResult, n_classes: int) -> str | None:
    """Classify metric pathology; None = healthy or not a metric failure."""
    if result.metric is None:
        return None
    import math
    if math.isnan(result.metric) or math.isinf(result.metric):
        return "nan_loss_divergence"
    tm, vm = result.train_metric or 0.0, result.val_metric or 0.0
    # majority collapse: no better than chance on classification
    chance = 1.0 / n_classes if n_classes > 1 else 0.0
    if n_classes >= 2 and vm <= chance + 0.03 and tm > chance + 0.15:
        return "class_imbalance_ignored"
    # severe overfitting: big train/val gap
    if n_classes >= 2 and tm - vm > 0.20 and tm > 0.85:
        return "severe_overfitting"
    # divergence: train metric itself is at/below chance late in training
    if n_classes >= 2 and tm <= chance + 0.05:
        return "nan_loss_divergence"
    # plateau: learned something but stalled far from good
    if n_classes >= 2 and vm < 0.75 and result.epochs_run >= 5:
        return "metric_plateau"
    return None


class Critic:
    def __init__(self, taxonomy: FailureTaxonomy):
        self.taxonomy = taxonomy

    def critique(self, result: RunResult, n_classes: int) -> Critique | None:
        """None = success, no critique needed."""
        if result.ok and result.metric is not None:
            # still check for pathology worth learning from
            pattern = diagnose_metrics(result, n_classes)
            if pattern is None:
                return None
            entry = self.taxonomy.match_metric(pattern)
            name = entry.name if entry else pattern
            # metric pathologies on an otherwise-"ok" run still count as failures
            return Critique(failure_name=name,
                            fix_strategy=entry.fix_strategy if entry else "",
                            directives=DIRECTIVES.get(name, []))
        if result.timed_out:
            self.taxonomy.record("timeout", fixed=False)
            return Critique(failure_name="timeout",
                            fix_strategy="Over budget: shrink the model and cap epochs — "
                                         "do less work per step, not more steps.",
                            directives=["shrink_model"])
        entry = self.taxonomy.match_traceback(result.traceback)
        if entry:
            return Critique(failure_name=entry.name, fix_strategy=entry.fix_strategy,
                            directives=DIRECTIVES.get(entry.name, []))
        # unknown traceback -> grow the taxonomy honestly (error is the LAST line)
        lines = [ln for ln in result.traceback.strip().splitlines() if ln.strip()]
        name = "unknown:" + lines[-1][:80] if lines else "unknown"
        self.taxonomy.record(name, fixed=False)
        return Critique(failure_name=name,
                        fix_strategy="Unseen error signature — recorded; retry once with "
                                     "lower LR and smaller batch.",
                        directives=["lower_lr", "smaller_batch"])

    def learn(self, critique: Critique, fixed: bool) -> None:
        """Record whether the fix worked; the taxonomy (and future plans) improve."""
        self.taxonomy.record(critique.failure_name, fixed=fixed)
        critique.learned = True
