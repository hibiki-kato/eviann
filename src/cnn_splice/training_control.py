"""Vendored early-stopping/validation-metric control utilities.

Verbatim (trimmed to the ``val_loss`` metric path actually used by
``dev/IntronModel/run/.train_v43.sh``'s ``VALIDATION_METRIC="val_loss"``)
from ``dev/IntronModel/src/util/training_control.py``. Function bodies are
copied unchanged from that file; only the metric-fallback tables were
narrowed to the metrics this driver actually computes (loss, acc@0.5).

Source line ranges:
- ``resolve_early_stopping_params``: training_control.py:93-135
- ``resolve_validation_metric``: training_control.py:188-213
- ``select_validation_score``: training_control.py:216-252
- ``get_metric_value``: training_control.py:255-283
"""

from __future__ import annotations

from typing import Mapping

VALIDATION_METRIC_CHOICES: tuple[str, ...] = ("acc@0.5", "val_loss")
_VALIDATION_METRIC_ALIASES: dict[str, str] = {"loss": "val_loss"}
_VALIDATION_METRIC_FALLBACKS: dict[str, tuple[str, ...]] = {
    "acc@0.5": ("acc@0.5",),
    # Lower loss is better; select_validation_score() negates it so the
    # shared "higher score is better" comparison used for checkpointing and
    # early stopping still applies without special-casing direction.
    "val_loss": ("val_loss",),
}
# select_validation_score() reads this raw metrics key for "val_loss" instead
# of a same-named key, since evaluation reports validation loss as "loss".
_VALIDATION_METRIC_SOURCE_KEYS: dict[str, str] = {"val_loss": "loss"}


def resolve_early_stopping_params(
    patience_arg: object,
    min_delta_arg: object,
    min_delta_rel_arg: object = 0.0,
) -> tuple[int, float, float]:
    """Validate and normalize early-stopping parameters.

    Verbatim from ``training_control.py:93-135``.
    """
    patience = int(patience_arg)
    if patience < 0:
        raise ValueError("--early_stop_patience must be >= 0.")

    min_delta = float(min_delta_arg)
    if min_delta < 0.0:
        raise ValueError("--early_stop_min_delta must be >= 0.")

    min_delta_rel = float(min_delta_rel_arg)
    if min_delta_rel < 0.0:
        raise ValueError("--early_stop_min_delta_rel must be >= 0.")

    return patience, min_delta, min_delta_rel


def resolve_validation_metric(metric_arg: object) -> str:
    """Validate and normalize one validation metric name.

    Verbatim from ``training_control.py:188-213``.
    """
    metric = str(metric_arg).strip().lower()
    if metric == "":
        raise ValueError("--validation_metric must not be empty.")
    metric = _VALIDATION_METRIC_ALIASES.get(metric, metric)
    if metric not in VALIDATION_METRIC_CHOICES:
        joined = "|".join(VALIDATION_METRIC_CHOICES)
        raise ValueError(f"--validation_metric must be one of: {joined}.")
    return metric


def select_validation_score(
    metrics: Mapping[str, object],
    validation_metric: object,
) -> tuple[float, str]:
    """Select one validation score for checkpointing and early stopping.

    Verbatim from ``training_control.py:216-252``.
    """
    normalized_metric = resolve_validation_metric(validation_metric)
    fallback_order = _VALIDATION_METRIC_FALLBACKS[normalized_metric]
    for metric_name in fallback_order:
        source_key = _VALIDATION_METRIC_SOURCE_KEYS.get(metric_name, metric_name)
        value = metrics.get(source_key)
        if value is None:
            continue
        score = -float(value) if metric_name == "val_loss" else float(value)
        return score, metric_name
    joined = ", ".join(fallback_order)
    raise ValueError(
        "Validation metrics are missing all compatible scoring keys for "
        f"--validation_metric={normalized_metric}: {joined}."
    )


def get_metric_value(
    metrics: Mapping[str, object],
    metric_name: object,
) -> float | None:
    """Return one named metric value as a float when it is available.

    Verbatim from ``training_control.py:255-283``.
    """
    key = str(metric_name).strip()
    if key == "":
        return None
    value = metrics.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
