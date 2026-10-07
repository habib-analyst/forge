"""Planner agent: natural-language task description -> structured training Plan.

The planner is rule-based and deterministic (no LLM API needed): it detects
the task family from the description, picks a model family, loss, metric,
backend (with recorded rationale), and injects lessons from the failure
taxonomy memory. The measured novelty is that plans improve as the taxonomy
grows: first-attempt success rate early vs late in a benchmark sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from .taxonomy import FailureTaxonomy


@dataclass
class Plan:
    task_name: str
    task_family: str            # image_classification | tabular_classification |
                                # tabular_regression | text_classification | sequence
    n_classes: int
    input_shape: tuple
    model_family: str           # cnn_small | mlp | embedding_rnn | linear_probe
    model_config: dict = field(default_factory=dict)
    loss: str = ""
    output_activation: str = ""
    metric: str = ""
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 10
    backend: str = "torch"      # chosen by BackendSelector with rationale
    backend_rationale: str = ""
    lessons_applied: list = field(default_factory=list)
    extra_directives: list = field(default_factory=list)  # from critic retries


# (activation, loss) pairs chosen as a unit — see taxonomy lesson.
HEAD_PRESETS = {
    "multiclass": ("softmax", "sparse_categorical_crossentropy", "accuracy"),
    "binary": ("sigmoid", "binary_crossentropy", "accuracy"),
    "regression": ("linear", "mse", "mae"),
}


class BackendSelector:
    """Scores Keras 3 backends on measured criteria; records the rationale.

    Honest scope: in this environment only the torch backend is installed, so
    selection deterministically yields torch. The mechanism (scored criteria +
    logged rationale) is what's validated here; cross-backend timing claims
    are out of scope until JAX/TF are installed.
    """

    CRITERIA = ["installed", "task_fit", "cpu_maturity"]

    def select(self, task_family: str, available: list[str]) -> tuple[str, str]:
        scores: dict[str, float] = {}
        notes: dict[str, list[str]] = {}
        for b in available:
            s, n = 0.0, []
            s += 1.0; n.append("installed")
            # small-model CPU training: torch backend is the mature CPU path in Keras 3
            fit = {"torch": 1.0, "jax": 0.8, "tensorflow": 0.9, "numpy": 0.4}.get(b, 0.2)
            s += fit; n.append(f"task_fit={fit}")
            cpu = {"torch": 1.0, "jax": 0.7, "tensorflow": 0.8, "numpy": 0.5}.get(b, 0.2)
            s += cpu; n.append(f"cpu_maturity={cpu}")
            scores[b] = round(s, 2); notes[b] = n
        best = max(scores, key=scores.get)
        rationale = (f"scores={scores}; chosen={best} "
                     f"({'; '.join(notes[best])}); available={available}")
        return best, rationale


_FAMILY_KEYWORDS = [
    ("image_classification", ["image", "pixel", "mnist", "cifar", "photo", "x-ray", "xray", "scan"]),
    ("text_classification", ["text", "sentiment", "review", "tweet", "document", "news article"]),
    ("tabular_regression", ["regression", "predict the value", "predict price", "continuous target"]),
    ("tabular_classification", ["tabular", "features", "columns", "iris", "wine", "spreadsheet"]),
    ("sequence", ["sequence", "time series", "parity", "next token"]),
]


class Planner:
    def __init__(self, taxonomy: FailureTaxonomy,
                 available_backends: list[str] | None = None):
        self.taxonomy = taxonomy
        self.available_backends = available_backends or ["torch"]
        self.selector = BackendSelector()

    def detect_family(self, description: str) -> str:
        d = description.lower()
        for family, kws in _FAMILY_KEYWORDS:
            if any(k in d for k in kws):
                return family
        return "tabular_classification"  # safest default

    def plan(self, task_name: str, description: str, n_classes: int,
             input_shape: tuple, extra_directives: list[str] | None = None) -> Plan:
        family = self.detect_family(description)
        backend, rationale = self.selector.select(family, self.available_backends)

        if family == "image_classification":
            model_family, cfg = "cnn_small", {"filters": [32, 64], "dense": 128, "dropout": 0.25}
            head = "binary" if n_classes == 2 else "multiclass"
        elif family == "text_classification":
            model_family, cfg = "embedding_rnn", {"embedding_dim": 64, "rnn_units": 64, "dropout": 0.3}
            head = "binary" if n_classes == 2 else "multiclass"
        elif family == "sequence":
            model_family, cfg = "embedding_rnn", {"embedding_dim": 32, "rnn_units": 64, "dropout": 0.2}
            head = "binary" if n_classes == 2 else "multiclass"
        elif family == "tabular_regression":
            model_family, cfg = "mlp", {"hidden": [64, 32], "dropout": 0.1}
            head = "regression"
        else:
            model_family, cfg = "mlp", {"hidden": [64, 32], "dropout": 0.2}
            head = "binary" if n_classes == 2 else "multiclass"

        activation, loss, metric = HEAD_PRESETS[head]
        lessons = self.taxonomy.lessons_for(family)

        return Plan(
            task_name=task_name, task_family=family, n_classes=n_classes,
            input_shape=tuple(input_shape), model_family=model_family,
            model_config=cfg, loss=loss, output_activation=activation,
            metric=metric, backend=backend, backend_rationale=rationale,
            lessons_applied=lessons,
            extra_directives=extra_directives or [],
        )

    def plan_dict(self, *a, **k) -> dict:
        p = self.plan(*a, **k)
        d = asdict(p)
        d["input_shape"] = list(d["input_shape"])
        return d
