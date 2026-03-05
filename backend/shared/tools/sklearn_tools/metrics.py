"""Metric wrappers for sklearn tools."""

from __future__ import annotations

from typing import Any, Callable

from sklearn.metrics import (
    accuracy_score as _accuracy_score,
    auc as _auc,
    f1_score as _f1_score,
    mean_absolute_error as _mean_absolute_error,
    mean_squared_error as _mean_squared_error,
    precision_recall_curve as _precision_recall_curve,
    precision_score as _precision_score,
    r2_score as _r2_score,
    recall_score as _recall_score,
    roc_curve as _roc_curve,
)


def _require_named_kwargs(kwargs: dict[str, Any], function_name: str, *required: str) -> tuple[Any, ...]:
    values = tuple(kwargs.get(name) for name in required)
    if any(value is None for value in values):
        raise ValueError(f"{function_name} requires {', '.join(required)}.")
    return values


def _make_scalar_metric(
    function_name: str,
    metric_fn: Callable[..., Any],
    *,
    result_key: str,
    required: tuple[str, ...],
    option_defaults: dict[str, Any] | None = None,
) -> Callable[..., dict[str, Any]]:
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        values = _require_named_kwargs(kwargs, function_name, *required)
        defaults = option_defaults or {}
        options = {name: kwargs.get(name, default) for name, default in defaults.items()}
        return {result_key: float(metric_fn(*values, **options))}

    wrapper.__name__ = function_name
    wrapper.__doc__ = f"Wrapper for sklearn `{function_name}`."
    return wrapper


def _make_curve_metric(
    function_name: str,
    metric_fn: Callable[..., Any],
    *,
    required: tuple[str, ...],
    result_keys: tuple[str, ...],
) -> Callable[..., dict[str, Any]]:
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        values = _require_named_kwargs(kwargs, function_name, *required)
        outputs = metric_fn(*values)
        return {key: value.tolist() for key, value in zip(result_keys, outputs)}

    wrapper.__name__ = function_name
    wrapper.__doc__ = f"Wrapper for sklearn `{function_name}`."
    return wrapper


r2_score = _make_scalar_metric("r2_score", _r2_score, result_key="r2_score", required=("y_true", "y_pred"))
mean_squared_error = _make_scalar_metric(
    "mean_squared_error",
    _mean_squared_error,
    result_key="mean_squared_error",
    required=("y_true", "y_pred"),
)
mean_absolute_error = _make_scalar_metric(
    "mean_absolute_error",
    _mean_absolute_error,
    result_key="mean_absolute_error",
    required=("y_true", "y_pred"),
)
accuracy_score = _make_scalar_metric(
    "accuracy_score",
    _accuracy_score,
    result_key="accuracy_score",
    required=("y_true", "y_pred"),
)
f1_score = _make_scalar_metric(
    "f1_score",
    _f1_score,
    result_key="f1_score",
    required=("y_true", "y_pred"),
    option_defaults={"average": "binary"},
)
precision_score = _make_scalar_metric(
    "precision_score",
    _precision_score,
    result_key="precision_score",
    required=("y_true", "y_pred"),
    option_defaults={"average": "binary"},
)
recall_score = _make_scalar_metric(
    "recall_score",
    _recall_score,
    result_key="recall_score",
    required=("y_true", "y_pred"),
    option_defaults={"average": "binary"},
)
auc_score = _make_scalar_metric("auc_score", _auc, result_key="auc", required=("x", "y"))
roc_curve = _make_curve_metric(
    "roc_curve",
    _roc_curve,
    required=("y_true", "y_score"),
    result_keys=("fpr", "tpr", "thresholds"),
)
precision_recall_curve = _make_curve_metric(
    "precision_recall_curve",
    _precision_recall_curve,
    required=("y_true", "y_score"),
    result_keys=("precision", "recall", "thresholds"),
)
