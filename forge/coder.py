"""Coder agent: Plan -> executable Keras 3 training script.

Generates a self-contained Python script (model + training + metric report)
from a Plan. The script prints a final JSON line: {"metric": ..., "history": ...}
or, on failure, the traceback goes to stderr. Critic directives (from a
previous failed iteration) are applied as code transformations.
"""

from __future__ import annotations

import textwrap
from .planner import Plan

TRAIN_SCRIPT = textwrap.dedent('''\
    import json, os, sys, traceback
    os.environ.setdefault("KERAS_BACKEND", "{backend}")
    # be a good citizen on shared/contended CPUs: cap thread pools
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import numpy as np
    try:
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(2)
    except ImportError:
        pass

    # ---- data (injected) ----
    {data_code}

    # ---- model ----
    import keras
    from keras import layers

    def build_model():
    {model_code}

    try:
        model = build_model()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate={lr}, clipnorm=1.0),
                      loss="{loss}", metrics=["{metric}"])
        # shape probe: fail fast on mismatch before burning budget
        _x_probe = X_train[:min(8, len(X_train))]
        _ = model(_x_probe)
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                   restore_best_weights=True)]
        {class_weight_code}
        # adaptive budget: tiny datasets get smaller batches and more epochs
        # so every task gets a comparable number of gradient steps
        _batch = min({batch}, max(8, len(X_train) // 16))
        _steps = max(1, len(X_train) // _batch)
        _epochs = max({epochs}, min(25, 300 // _steps))
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=_epochs, batch_size=_batch,
                         callbacks=callbacks, verbose=0, **_fit_kwargs)
        loss, metric_val = model.evaluate(X_test, y_test, verbose=0)
        # metric pathology signals for the critic
        train_acc = hist.history["{metric}"][-1]
        val_acc = hist.history.get("val_{metric}", [0])[-1]
        print(json.dumps({{"status": "ok", "metric": float(metric_val),
                           "train_metric": float(train_acc), "val_metric": float(val_acc),
                           "epochs_run": len(hist.history["loss"])}}))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    ''')


def _model_code(plan: Plan) -> str:
    ind = "        "
    if plan.model_family == "cnn_small":
        filters = plan.model_config.get("filters", [32, 64])
        dense = plan.model_config.get("dense", 128)
        dropout = plan.model_config.get("dropout", 0.25)
        lines = [f"inputs = keras.Input(shape={tuple(plan.input_shape)})", "x = inputs"]
        for f in filters:
            lines.append(f"x = layers.Conv2D({f}, 3, activation='relu', padding='same')(x)")
            lines.append("x = layers.MaxPooling2D()(x)")
        lines += ["x = layers.Flatten()(x)",
                  f"x = layers.Dense({dense}, activation='relu')(x)",
                  f"x = layers.Dropout({dropout})(x)",
                  f"outputs = layers.Dense({plan.n_classes if plan.n_classes > 2 else 1}, "
                  f"activation='{plan.output_activation}')(x)",
                  "return keras.Model(inputs, outputs)"]
        return "\n".join(ind + ln for ln in lines)
    if plan.model_family == "embedding_rnn":
        vocab = plan.model_config.get("vocab_size", 2000)
        edim = plan.model_config.get("embedding_dim", 64)
        units = plan.model_config.get("rnn_units", 64)
        dropout = plan.model_config.get("dropout", 0.3)
        seqlen = plan.input_shape[0] if plan.input_shape else 50
        lines = [f"inputs = keras.Input(shape=({seqlen},))",
                 f"x = layers.Embedding({vocab}, {edim})(inputs)",
                 f"x = layers.GRU({units}, dropout={dropout})(x)",
                 f"outputs = layers.Dense({plan.n_classes if plan.n_classes > 2 else 1}, "
                 f"activation='{plan.output_activation}')(x)",
                 "return keras.Model(inputs, outputs)"]
        return "\n".join(ind + ln for ln in lines)
    # mlp (tabular)
    hidden = plan.model_config.get("hidden", [64, 32])
    dropout = plan.model_config.get("dropout", 0.2)
    dim = plan.input_shape[0] if plan.input_shape else 10
    lines = [f"inputs = keras.Input(shape=({dim},))", "x = inputs"]
    for h in hidden:
        lines.append(f"x = layers.Dense({h}, activation='relu')(x)")
        lines.append(f"x = layers.Dropout({dropout})(x)")
    lines.append(f"outputs = layers.Dense({plan.n_classes if plan.n_classes > 2 else 1}, "
                 f"activation='{plan.output_activation}')(x)")
    lines.append("return keras.Model(inputs, outputs)")
    return "\n".join(ind + ln for ln in lines)


def _apply_directives(plan: Plan) -> Plan:
    """Apply critic directives from a previous iteration (bounded, predictable)."""
    import copy
    plan = copy.deepcopy(plan)
    for d in plan.extra_directives:
        if d == "widen_model":
            cfg = plan.model_config
            if "filters" in cfg:
                cfg["filters"] = [f * 2 for f in cfg["filters"]]
            if "hidden" in cfg:
                cfg["hidden"] = [h * 2 for h in cfg["hidden"]]
            if "dense" in cfg:
                cfg["dense"] *= 2
            if "embedding_dim" in cfg:
                cfg["embedding_dim"] *= 2
            if "rnn_units" in cfg:
                cfg["rnn_units"] *= 2
        elif d == "lower_lr":
            plan.learning_rate /= 10.0
        elif d == "more_epochs":
            plan.max_epochs = min(plan.max_epochs + 10, 30)
        elif d == "add_class_weight":
            plan.model_config["class_weight_balanced"] = True
        elif d == "more_regularization":
            plan.model_config["dropout"] = min(plan.model_config.get("dropout", 0.2) + 0.2, 0.6)
        elif d == "smaller_batch":
            plan.batch_size = max(plan.batch_size // 2, 8)
        elif d == "shrink_model":
            # timed out: do less work per step, not more steps
            cfg = plan.model_config
            if "filters" in cfg:
                cfg["filters"] = [max(8, f // 2) for f in cfg["filters"][:2]]
            if "hidden" in cfg:
                cfg["hidden"] = [max(8, h // 2) for h in cfg["hidden"][:1]]
            if "dense" in cfg:
                cfg["dense"] = max(32, cfg["dense"] // 2)
            plan.max_epochs = min(plan.max_epochs, 8)
    plan.extra_directives = []
    return plan


class Coder:
    def generate(self, plan: Plan, data_code: str) -> str:
        plan = _apply_directives(plan)
        if plan.model_config.get("class_weight_balanced"):
            cw_code = '''\
from sklearn.utils.class_weight import compute_class_weight
_classes = __import__("numpy").unique(y_train)
_cw = compute_class_weight("balanced", classes=_classes, y=y_train)
_fit_kwargs = {"class_weight": dict(zip(_classes, _cw))}'''
        else:
            cw_code = "_fit_kwargs = {}"
        # NOTE: TRAIN_SCRIPT is dedented by 4: {data_code} lands at column 0
        # (module level) and {class_weight_code} at 4 spaces (inside try:).
        # Payloads therefore carry no extra indent of their own.
        return TRAIN_SCRIPT.format(
            backend=plan.backend,
            class_weight_code=cw_code.strip(),
            data_code=data_code.strip(),
            model_code=_model_code(plan),
            lr=plan.learning_rate,
            loss=plan.loss,
            metric=plan.metric,
            epochs=plan.max_epochs,
            batch=plan.batch_size,
        )
