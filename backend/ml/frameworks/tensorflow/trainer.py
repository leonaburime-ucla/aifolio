from __future__ import annotations

"""TensorFlow runtime trainer implementation."""

from pathlib import Path
from typing import Any

import numpy as np

try:
    from ...file_util import coerce_value, load_tabular_file, split_features_target
    from ...ml_util import TaskType, expand_date_columns, infer_task
except ImportError:  # pragma: no cover
    from file_util import coerce_value, load_tabular_file, split_features_target  # type: ignore
    from ml_util import TaskType, expand_date_columns, infer_task  # type: ignore

from ...core.preprocessing import impute_non_finite_with_reference_medians
from ...core.types import Metrics, ModelBundle, TrainingConfig
from .data import prepare_arrays
from .distill import distill_model_from_file
from .models import build_model, set_seed
from .serialization import load_bundle as _load_bundle, save_bundle


def _build_fit_targets(
    training_mode: str,
    x_train: Any,
    x_test: Any,
    y_train: Any,
    y_test: Any,
) -> tuple[Any, Any]:
    """Build fit/eval targets for special multi-output training modes."""
    if training_mode == "multi_task_learning":
        return (
            {"main_output": y_train, "aux_output": y_train},
            {"main_output": y_test, "aux_output": y_test},
        )
    if training_mode == "autoencoder_head":
        return (
            {"main_output": y_train, "reconstruction_output": x_train},
            {"main_output": y_test, "reconstruction_output": x_test},
        )
    return y_train, y_test


def _build_class_weight_map(y_train: np.ndarray, output_dim: int) -> dict[int, float]:
    """Build class weights for imbalance-aware classification."""
    counts = np.bincount(y_train.astype(np.int64), minlength=output_dim).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    total = float(np.sum(counts))
    return {idx: float(total / (float(output_dim) * float(counts[idx]))) for idx in range(output_dim)}


def _main_output(preds: Any) -> np.ndarray:
    """Normalize Keras predict outputs to the primary prediction head."""
    if isinstance(preds, dict):
        if "main_output" in preds:
            return np.asarray(preds["main_output"])
        return np.asarray(next(iter(preds.values())))
    if isinstance(preds, list):
        return np.asarray(preds[0])
    return np.asarray(preds)


def _regression_metric(
    preds: np.ndarray,
    y_true: np.ndarray,
    target_scaler: Any | None,
    training_mode: str,
) -> tuple[str, float]:
    """Compute regression metric output for TensorFlow trainer modes."""
    if target_scaler is not None:
        preds_original = target_scaler.inverse_transform(preds)
        y_true_original = target_scaler.inverse_transform(y_true)
    else:
        preds_original = preds
        y_true_original = y_true
    rmse = float(np.sqrt(np.mean((preds_original - y_true_original) ** 2)))
    if training_mode == "quantile_regression":
        error = y_true_original - preds_original
        quantile = 0.8
        metric_name = "pinball_p80"
        metric_value = float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))
        return metric_name, metric_value
    return "rmse", rmse


def _regression_predictions_to_list(preds: np.ndarray, target_scaler: Any | None) -> list[Any]:
    """Convert regression outputs into response payload values."""
    transformed = target_scaler.inverse_transform(preds) if target_scaler is not None else preds
    return transformed.squeeze(1).tolist()


def train_model_from_file(
    data_path: str | Path,
    cfg: TrainingConfig,
    sheet_name: str | None = None,
    exclude_columns: list[str] | None = None,
    date_columns: list[str] | None = None,
    device: str | None = None,
) -> tuple[ModelBundle, Metrics]:
    set_seed(cfg.random_seed)
    _ = device

    rows = load_tabular_file(data_path, sheet_name=sheet_name)
    rows = expand_date_columns(rows, date_columns=date_columns)
    excluded = {column for column in (exclude_columns or []) if column}
    if cfg.target_column in excluded:
        raise ValueError("target_column cannot be excluded")
    if excluded:
        rows = [{k: v for k, v in row.items() if k not in excluded} for row in rows]
    x_rows, y_raw = split_features_target(rows, cfg.target_column)

    task: TaskType = infer_task(y_raw) if cfg.task == "auto" else cfg.task
    if cfg.training_mode in {"imbalance_aware", "calibrated_classifier"} and task != "classification":
        raise ValueError(
            f"training_mode '{cfg.training_mode}' is classification-only; use task='classification' or choose a regression mode."
        )
    if cfg.training_mode == "quantile_regression" and task != "regression":
        raise ValueError(
            "training_mode 'quantile_regression' is regression-only; use task='regression' or choose a classification mode."
        )

    (
        x_train,
        x_test,
        y_train,
        y_test,
        vectorizer,
        scaler,
        feature_medians,
        label_encoder,
        target_scaler,
        input_dim,
        output_dim,
        class_names,
    ) = prepare_arrays(x_rows, y_raw, task, cfg)

    model = build_model(
        input_dim=input_dim,
        output_dim=output_dim,
        task=task,
        training_mode=cfg.training_mode,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
    )
    y_train_fit, y_test_fit = _build_fit_targets(cfg.training_mode, x_train, x_test, y_train, y_test)

    fit_kwargs: dict[str, Any] = {}
    if task == "classification" and cfg.training_mode == "imbalance_aware":
        fit_kwargs["class_weight"] = _build_class_weight_map(y_train, output_dim)

    model.fit(x_train, y_train_fit, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0, **fit_kwargs)

    train_eval = model.evaluate(x_train, y_train_fit, verbose=0)
    test_eval = model.evaluate(x_test, y_test_fit, verbose=0)
    train_loss = float(train_eval[0] if isinstance(train_eval, (list, tuple)) else train_eval)
    test_loss = float(test_eval[0] if isinstance(test_eval, (list, tuple)) else test_eval)

    if task == "classification":
        probs = _main_output(model.predict(x_test, verbose=0))
        predicted_classes = np.argmax(probs, axis=1)
        test_accuracy = float(np.mean(predicted_classes == y_test))
        metrics = Metrics(
            task=task,
            train_loss=train_loss,
            test_loss=test_loss,
            test_metric_name="accuracy",
            test_metric_value=test_accuracy,
        )
    else:
        preds = _main_output(model.predict(x_test, verbose=0))
        metric_name, metric_value = _regression_metric(preds, y_test, target_scaler, cfg.training_mode)
        metrics = Metrics(
            task=task,
            train_loss=train_loss,
            test_loss=test_loss,
            test_metric_name=metric_name,
            test_metric_value=metric_value,
        )

    return (
        ModelBundle(
            model=model,
            task=task,
            vectorizer=vectorizer,
            scaler=scaler,
            feature_medians=feature_medians,
            label_encoder=label_encoder,
            target_scaler=target_scaler,
            target_column=cfg.target_column,
            input_dim=input_dim,
            output_dim=output_dim,
            class_names=class_names,
            model_config={
                "training_mode": cfg.training_mode,
                "hidden_dim": cfg.hidden_dim,
                "num_hidden_layers": cfg.num_hidden_layers,
                "dropout": cfg.dropout,
            },
        ),
        metrics,
    )


def predict_rows(bundle: ModelBundle, rows: list[dict[str, Any]], device: str | None = None) -> list[Any]:
    _ = device
    if not rows:
        return []

    x_rows = [{k: coerce_value(v) for k, v in row.items()} for row in rows]
    x_np = bundle.vectorizer.transform(x_rows).astype(np.float32)

    if bundle.feature_medians is not None and bundle.feature_medians.shape[0] == x_np.shape[1]:
        medians = bundle.feature_medians
    else:
        medians = np.zeros(x_np.shape[1], dtype=np.float32)

    x_np = impute_non_finite_with_reference_medians(x_np, medians)
    x_np = bundle.scaler.transform(x_np).astype(np.float32)
    outputs = bundle.model.predict(x_np, verbose=0)

    if bundle.task == "classification":
        indices = outputs.argmax(axis=1)
        if bundle.label_encoder is not None:
            return bundle.label_encoder.inverse_transform(indices).tolist()
        return indices.tolist()

    preds = outputs
    return _regression_predictions_to_list(preds, bundle.target_scaler)


__all__ = [
    "Metrics",
    "ModelBundle",
    "TrainingConfig",
    "distill_model_from_file",
    "load_bundle",
    "predict_rows",
    "save_bundle",
    "train_model_from_file",
]


def load_bundle(path: str | Path, map_location: str = "cpu") -> ModelBundle:
    return _load_bundle(path=path, map_location=map_location)
