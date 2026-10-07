"""Failure taxonomy memory — the learning component of FORGE.

The Critic distills every failure (traceback or metric pathology) into a
taxonomy entry: a *signature* (regex over tracebacks / metric pattern),
a *fix strategy*, and running *success statistics*. The Planner queries this
memory before writing a plan, so plans get better as the system sees more
tasks. This is the measured novelty: planner first-attempt success rate
early vs late in a benchmark sequence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class FailureEntry:
    name: str
    # regex matched against the traceback text (empty = metric-pathology entry)
    signature: str
    # e.g. "metric_plateau", "divergence", "shape_mismatch" for metric entries
    metric_pattern: str = ""
    fix_strategy: str = ""
    # human-readable lesson injected into future plans
    lesson: str = ""
    times_seen: int = 0
    times_fixed: int = 0

    @property
    def fix_rate(self) -> float:
        return self.times_fixed / self.times_seen if self.times_seen else 0.0


# Seed taxonomy: known failure modes of small-model training, distilled from
# practitioner knowledge. FORGE grows this at runtime (see Critic.learn).
SEED_ENTRIES = [
    FailureEntry(
        name="shape_mismatch",
        signature=r"(shapes? .* (incompatible|mismatch)|Input .* incompatible|expected .* got |incompatible shapes)",
        fix_strategy="Align tensor shapes: check model input_shape vs data shape; "
                     "add Reshape/Flatten or fix the final Dense units to n_classes.",
        lesson="Always verify data batch shape against model.input_shape before training; "
               "add an explicit shape-assertion probe run.",
    ),
    FailureEntry(
        name="channel_mismatch",
        signature=r"(channels|channel dimension|depth of input)",
        fix_strategy="Fix Conv input channels (e.g. expand grayscale to 1 channel) "
                     "and set data_format consistently.",
        lesson="For image tasks, normalize inputs to (H, W, C) channels-last and set "
               "Conv2D input_shape accordingly.",
    ),
    FailureEntry(
        name="oom_or_resource",
        signature=r"(ResourceExhausted|out of memory|OOM|allocation failed)",
        fix_strategy="Halve batch size; shrink model width/depth; enable mixed precision "
                     "only if backend supports it.",
        lesson="Start with batch_size<=64 and small widths on CPU; scale up only if stable.",
    ),
    FailureEntry(
        name="nan_loss_divergence",
        signature=r"(NaN|nan|inf |InvalidArgumentError.*loss)",
        metric_pattern="divergence",
        fix_strategy="Lower learning rate 10x, add gradient clipping (clipnorm=1.0), "
                     "check label encoding matches loss (sparse vs one-hot).",
        lesson="Default to Adam lr=1e-3 with clipnorm=1.0; verify labels are integers "
               "in [0, n_classes) for sparse_categorical_crossentropy.",
    ),
    FailureEntry(
        name="label_loss_mismatch",
        signature=r"(logits and labels|Received a label value|SparseCategorical|expected binary)",
        fix_strategy="Match loss to label format: integer labels -> sparse_categorical_crossentropy; "
                     "one-hot -> categorical_crossentropy; binary 0/1 -> binary_crossentropy "
                     "with sigmoid output.",
        lesson="Choose (activation, loss) as a pair: (softmax, sparse_categorical_crossentropy) "
               "multiclass; (sigmoid, binary_crossentropy) binary; (linear, mse) regression.",
    ),
    FailureEntry(
        name="metric_plateau",
        signature="",
        metric_pattern="plateau",
        fix_strategy="Increase model capacity (wider/deeper), train longer, or add augmentation; "
                     "check the task isn't under-specified (e.g. too few features).",
        lesson="If val metric stalls within 15% of baseline after full budget, the model is "
               "under-capacity for the task — widen before deepening.",
    ),
    FailureEntry(
        name="severe_overfitting",
        signature="",
        metric_pattern="overfitting",
        fix_strategy="Add Dropout(0.3-0.5), L2 regularization, early stopping on val_loss, "
                     "or reduce capacity; check for label leakage in features.",
        lesson="Train/val gap > 20pp with tiny data means regularize first, not more epochs.",
    ),
    FailureEntry(
        name="class_imbalance_ignored",
        signature="",
        metric_pattern="majority_collapse",
        fix_strategy="Use class_weight='balanced' or focal-style weighting; monitor per-class "
                     "recall, not just accuracy.",
        lesson="On imbalanced tasks, accuracy lies — optimize macro-F1 and pass class weights.",
    ),
]


class FailureTaxonomy:
    """Persistent, growing memory of failure modes and their fixes."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.entries: list[FailureEntry] = [FailureEntry(**asdict(e)) for e in SEED_ENTRIES]
        if self.path and self.path.exists():
            self.load()

    # -- matching ------------------------------------------------------
    def match_traceback(self, tb_text: str) -> FailureEntry | None:
        for e in self.entries:
            if e.signature and re.search(e.signature, tb_text, re.IGNORECASE | re.DOTALL):
                return e
        return None

    def match_metric(self, pattern: str) -> FailureEntry | None:
        for e in self.entries:
            if e.metric_pattern and e.metric_pattern == pattern:
                return e
        return None

    # -- learning ------------------------------------------------------
    def record(self, entry_name: str, fixed: bool) -> None:
        for e in self.entries:
            if e.name == entry_name:
                e.times_seen += 1
                e.times_fixed += int(fixed)
                self.save()
                return
        # genuinely new failure mode -> grow the taxonomy
        self.entries.append(FailureEntry(
            name=entry_name, signature="", fix_strategy="escalate: new pattern",
            times_seen=1, times_fixed=int(fixed)))
        self.save()

    def mark_fixed(self, entry_name: str) -> None:
        """Credit a fix without double-counting times_seen."""
        for e in self.entries:
            if e.name == entry_name:
                e.times_fixed += 1
                self.save()
                return

    def lessons_for(self, task_family: str) -> list[str]:
        """Lessons injected into the planner prompt, ranked by fix rate."""
        ranked = sorted(self.entries, key=lambda e: (e.fix_rate, e.times_seen), reverse=True)
        return [e.lesson for e in ranked if e.lesson][:6]

    def fix_rate_summary(self) -> dict[str, float]:
        return {e.name: round(e.fix_rate, 3) for e in self.entries if e.times_seen}

    # -- persistence ---------------------------------------------------
    def save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps([asdict(e) for e in self.entries], indent=2))

    def load(self) -> None:
        data = json.loads(self.path.read_text())
        self.entries = [FailureEntry(**d) for d in data]
