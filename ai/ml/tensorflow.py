from __future__ import annotations

import argparse
import json
import pickle
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import tensorflow as tf

try:
    from .distill import (
        InMemoryBundleRegistry,
        distill_model_from_file as shared_distill_model_from_file,
        handle_distill_request as shared_handle_distill_request,
    )
    from .file_util import coerce_value, load_tabular_file, split_features_target
    from .ml_util import TaskType, expand_date_columns, infer_task
except ImportError:  # pragma: no cover
    from distill import (
        InMemoryBundleRegistry,
        distill_model_from_file as shared_distill_model_from_file,
        handle_distill_request as shared_handle_distill_request,
    )
    from file_util import coerce_value, load_tabular_file, split_features_target
    from ml_util import TaskType, expand_date_columns, infer_task

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TrainingMode = Literal[
    "mlp_dense",
    "linear_glm_baseline",
    "wide_and_deep",
    "imbalance_aware",
    "quantile_regression",
    "calibrated_classifier",
]
_BUNDLE_REGISTRY: InMemoryBundleRegistry[ModelBundle] = InMemoryBundleRegistry(
    ttl_seconds=900, max_items=128
)


@dataclass
class TrainingConfig:
    target_column: str
    training_mode: TrainingMode = "mlp_dense"
    task: TaskType | Literal["auto"] = "auto"
    test_size: float = 0.2
    random_seed: int = 42
    epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.1


@dataclass
class Metrics:
    task: TaskType
    train_loss: float
    test_loss: float
    test_metric_name: str
    test_metric_value: float


@dataclass
class ModelBundle:
    model: tf.keras.Model
    task: TaskType
    vectorizer: DictVectorizer
    scaler: StandardScaler
    feature_medians: np.ndarray | None
    label_encoder: LabelEncoder | None
    target_scaler: StandardScaler | None
    target_column: str
    input_dim: int
    output_dim: int
    class_names: list[str] | None
    model_config: dict[str, Any]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _prepare_arrays(
    x_rows: list[dict[str, Any]],
    y_raw: list[Any],
    task: TaskType,
    cfg: TrainingConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    DictVectorizer,
    StandardScaler,
    np.ndarray,
    LabelEncoder | None,
    StandardScaler | None,
    int,
    int,
    list[str] | None,
]:
    if task == "classification":
        y_labels = [str(v) for v in y_raw]
        unique_labels = set(y_labels)
        stratify: list[str] | None = y_labels if len(unique_labels) > 1 else None
    else:
        stratify = None

    x_train_rows, x_test_rows, y_train_raw, y_test_raw = train_test_split(
        x_rows,
        y_raw,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=stratify,
    )

    vectorizer = DictVectorizer(sparse=False)
    x_train_np = vectorizer.fit_transform(x_train_rows).astype(np.float32)
    x_test_np = vectorizer.transform(x_test_rows).astype(np.float32)

    col_medians = np.nanmedian(np.where(np.isfinite(x_train_np), x_train_np, np.nan), axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    for col_idx in range(x_train_np.shape[1]):
        train_mask = ~np.isfinite(x_train_np[:, col_idx])
        x_train_np[train_mask, col_idx] = col_medians[col_idx]
        test_mask = ~np.isfinite(x_test_np[:, col_idx])
        x_test_np[test_mask, col_idx] = col_medians[col_idx]

    scaler = StandardScaler()
    x_train_np = scaler.fit_transform(x_train_np).astype(np.float32)
    x_test_np = scaler.transform(x_test_np).astype(np.float32)

    if task == "classification":
        encoder = LabelEncoder()
        target_scaler = None
        y_train_np = encoder.fit_transform([str(v) for v in y_train_raw]).astype(np.int32)
        y_test_np = encoder.transform([str(v) for v in y_test_raw]).astype(np.int32)
        output_dim = int(len(encoder.classes_))
        class_names = [str(c) for c in encoder.classes_]
    else:
        encoder = None
        target_scaler = StandardScaler()
        y_train_np = np.array([float(v) for v in y_train_raw], dtype=np.float32).reshape(-1, 1)
        y_test_np = np.array([float(v) for v in y_test_raw], dtype=np.float32).reshape(-1, 1)
        y_train_np = target_scaler.fit_transform(y_train_np).astype(np.float32)
        y_test_np = target_scaler.transform(y_test_np).astype(np.float32)
        output_dim = 1
        class_names = None

    input_dim = int(x_train_np.shape[1])
    return (
        x_train_np,
        x_test_np,
        y_train_np,
        y_test_np,
        vectorizer,
        scaler,
        col_medians,
        encoder,
        target_scaler,
        input_dim,
        output_dim,
        class_names,
    )


def _build_model(
    input_dim: int,
    output_dim: int,
    task: TaskType,
    training_mode: TrainingMode,
    hidden_dim: int,
    num_hidden_layers: int,
    dropout: float,
    learning_rate: float,
) -> tf.keras.Model:
    quantile_tau = 0.8

    def pinball_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        err = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile_tau * err, (quantile_tau - 1.0) * err))

    inputs = tf.keras.layers.Input(shape=(input_dim,))

    if training_mode == "linear_glm_baseline":
        logits = tf.keras.layers.Dense(output_dim, activation=None, name="linear_head")(inputs)
    elif training_mode == "entity_embeddings":
        embed_dim = max(16, min(hidden_dim, 128))
        embedded = tf.keras.layers.Dense(embed_dim, activation=None, name="entity_embed_proj")(inputs)
        embedded = tf.keras.layers.BatchNormalization(name="entity_embed_bn")(embedded)
        embedded = tf.keras.layers.ReLU(name="entity_embed_relu")(embedded)
        deep = embedded
        for idx in range(max(1, num_hidden_layers)):
            deep = tf.keras.layers.Dense(hidden_dim, activation=None, name=f"entity_dense_{idx}")(deep)
            deep = tf.keras.layers.BatchNormalization(name=f"entity_bn_{idx}")(deep)
            deep = tf.keras.layers.ReLU(name=f"entity_relu_{idx}")(deep)
            if dropout > 0:
                deep = tf.keras.layers.Dropout(dropout, name=f"entity_dropout_{idx}")(deep)
        logits = tf.keras.layers.Dense(output_dim, activation=None, name="entity_head")(deep)
    elif training_mode == "wide_and_deep":
        wide_logits = tf.keras.layers.Dense(output_dim, activation=None, name="wide_head")(inputs)
        deep = inputs
        for idx in range(max(1, num_hidden_layers)):
            deep = tf.keras.layers.Dense(hidden_dim, activation=None, name=f"deep_dense_{idx}")(deep)
            deep = tf.keras.layers.BatchNormalization(name=f"deep_bn_{idx}")(deep)
            deep = tf.keras.layers.ReLU(name=f"deep_relu_{idx}")(deep)
            if dropout > 0:
                deep = tf.keras.layers.Dropout(dropout, name=f"deep_dropout_{idx}")(deep)
        deep_logits = tf.keras.layers.Dense(output_dim, activation=None, name="deep_head")(deep)
        logits = tf.keras.layers.Add(name="wide_deep_add")([wide_logits, deep_logits])
    elif training_mode == "time_aware_tabular":
        gate = tf.keras.layers.Dense(input_dim, activation="sigmoid", name="temporal_gate")(inputs)
        gated = tf.keras.layers.Multiply(name="temporal_multiply")([inputs, gate])
        merged = tf.keras.layers.Concatenate(name="temporal_concat")([inputs, gated])
        deep = merged
        for idx in range(max(1, num_hidden_layers)):
            deep = tf.keras.layers.Dense(hidden_dim, activation=None, name=f"time_dense_{idx}")(deep)
            deep = tf.keras.layers.BatchNormalization(name=f"time_bn_{idx}")(deep)
            deep = tf.keras.layers.ReLU(name=f"time_relu_{idx}")(deep)
            if dropout > 0:
                deep = tf.keras.layers.Dropout(dropout, name=f"time_dropout_{idx}")(deep)
        logits = tf.keras.layers.Dense(output_dim, activation=None, name="time_head")(deep)
    else:
        deep = inputs
        for idx in range(max(1, num_hidden_layers)):
            deep = tf.keras.layers.Dense(hidden_dim, activation=None, name=f"dense_{idx}")(deep)
            deep = tf.keras.layers.BatchNormalization(name=f"bn_{idx}")(deep)
            deep = tf.keras.layers.ReLU(name=f"relu_{idx}")(deep)
            if dropout > 0:
                deep = tf.keras.layers.Dropout(dropout, name=f"dropout_{idx}")(deep)
        if training_mode == "autoencoder_head":
            bottleneck_dim = max(8, hidden_dim // 2)
            bottleneck = tf.keras.layers.Dense(
                bottleneck_dim, activation="relu", name="autoencoder_bottleneck"
            )(deep)
            recon = tf.keras.layers.Dense(
                input_dim, activation="linear", name="reconstruction_output"
            )(bottleneck)
            pred_logits = tf.keras.layers.Dense(
                output_dim, activation=None, name="main_logits"
            )(bottleneck)
        elif training_mode == "multi_task_learning":
            shared = tf.keras.layers.Dense(hidden_dim, activation="relu", name="shared_trunk")(deep)
            main_logits = tf.keras.layers.Dense(
                output_dim, activation=None, name="main_logits"
            )(shared)
            aux_logits = tf.keras.layers.Dense(
                output_dim, activation=None, name="aux_logits"
            )(shared)
        else:
            logits = tf.keras.layers.Dense(output_dim, activation=None, name="mlp_head")(deep)

    if task == "classification":
        loss_fn: str | tf.keras.losses.Loss = "sparse_categorical_crossentropy"
        if training_mode == "calibrated_classifier":
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05)
        if training_mode == "autoencoder_head":
            main_probs = tf.keras.layers.Activation(
                "softmax", name="main_output"
            )(pred_logits)
            model = tf.keras.Model(
                inputs=inputs, outputs=[main_probs, recon], name=f"tf_{training_mode}"
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss={"main_output": loss_fn, "reconstruction_output": "mse"},
                loss_weights={"main_output": 1.0, "reconstruction_output": 0.2},
                metrics={"main_output": ["accuracy"]},
            )
        elif training_mode == "multi_task_learning":
            main_probs = tf.keras.layers.Activation(
                "softmax", name="main_output"
            )(main_logits)
            aux_probs = tf.keras.layers.Activation("softmax", name="aux_output")(aux_logits)
            model = tf.keras.Model(
                inputs=inputs, outputs=[main_probs, aux_probs], name=f"tf_{training_mode}"
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss={"main_output": loss_fn, "aux_output": loss_fn},
                loss_weights={"main_output": 1.0, "aux_output": 0.3},
                metrics={"main_output": ["accuracy"]},
            )
        else:
            outputs = tf.keras.layers.Activation("softmax", name="softmax")(logits)
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"tf_{training_mode}")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss_fn,
                metrics=["accuracy"],
            )
    else:
        loss_fn: str | Callable[..., tf.Tensor] = (
            pinball_loss if training_mode == "quantile_regression" else "mse"
        )
        if training_mode == "autoencoder_head":
            main_out = tf.keras.layers.Activation(
                "linear", name="main_output"
            )(pred_logits)
            model = tf.keras.Model(
                inputs=inputs, outputs=[main_out, recon], name=f"tf_{training_mode}"
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss={"main_output": loss_fn, "reconstruction_output": "mse"},
                loss_weights={"main_output": 1.0, "reconstruction_output": 0.2},
            )
        elif training_mode == "multi_task_learning":
            main_out = tf.keras.layers.Activation(
                "linear", name="main_output"
            )(main_logits)
            aux_out = tf.keras.layers.Activation("linear", name="aux_output")(aux_logits)
            model = tf.keras.Model(
                inputs=inputs, outputs=[main_out, aux_out], name=f"tf_{training_mode}"
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss={"main_output": loss_fn, "aux_output": "mse"},
                loss_weights={"main_output": 1.0, "aux_output": 0.3},
            )
        else:
            outputs = tf.keras.layers.Activation("linear", name="regression_output")(logits)
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"tf_{training_mode}")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss_fn,
            )

    return model


def train_model_from_file(
    data_path: str | Path,
    cfg: TrainingConfig,
    sheet_name: str | None = None,
    exclude_columns: list[str] | None = None,
    date_columns: list[str] | None = None,
    device: str | None = None,
) -> tuple[ModelBundle, Metrics]:
    _set_seed(cfg.random_seed)
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
    ) = _prepare_arrays(x_rows, y_raw, task, cfg)

    model = _build_model(
        input_dim=input_dim,
        output_dim=output_dim,
        task=task,
        training_mode=cfg.training_mode,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
    )

    is_multi_task = cfg.training_mode == "multi_task_learning"
    is_autoencoder_head = cfg.training_mode == "autoencoder_head"
    if is_multi_task:
        y_train_fit: Any = {"main_output": y_train, "aux_output": y_train}
        y_test_fit: Any = {"main_output": y_test, "aux_output": y_test}
    elif is_autoencoder_head:
        y_train_fit = {"main_output": y_train, "reconstruction_output": x_train}
        y_test_fit = {"main_output": y_test, "reconstruction_output": x_test}
    else:
        y_train_fit = y_train
        y_test_fit = y_test

    fit_kwargs: dict[str, Any] = {}
    if task == "classification" and cfg.training_mode == "imbalance_aware":
        counts = np.bincount(y_train.astype(np.int64), minlength=output_dim).astype(np.float32)
        counts = np.clip(counts, a_min=1.0, a_max=None)
        total = float(np.sum(counts))
        fit_kwargs["class_weight"] = {
            idx: float(total / (float(output_dim) * float(counts[idx])))
            for idx in range(output_dim)
        }

    model.fit(
        x_train,
        y_train_fit,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        **fit_kwargs,
    )

    train_eval = model.evaluate(x_train, y_train_fit, verbose=0)
    test_eval = model.evaluate(x_test, y_test_fit, verbose=0)
    train_loss = float(train_eval[0] if isinstance(train_eval, (list, tuple)) else train_eval)
    test_loss = float(test_eval[0] if isinstance(test_eval, (list, tuple)) else test_eval)

    def _main_output(preds: Any) -> np.ndarray:
        if isinstance(preds, dict):
            if "main_output" in preds:
                return np.asarray(preds["main_output"])
            return np.asarray(next(iter(preds.values())))
        if isinstance(preds, list):
            return np.asarray(preds[0])
        return np.asarray(preds)

    if task == "classification":
        preds_raw = model.predict(x_test, verbose=0)
        probs = _main_output(preds_raw)
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
        if target_scaler is not None:
            preds_original = target_scaler.inverse_transform(preds)
            y_test_original = target_scaler.inverse_transform(y_test)
        else:
            preds_original = preds
            y_test_original = y_test
        rmse = float(np.sqrt(np.mean((preds_original - y_test_original) ** 2)))
        if cfg.training_mode == "quantile_regression":
            error = y_test_original - preds_original
            quantile = 0.8
            metric_name = "pinball_p80"
            metric_value = float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))
        else:
            metric_name = "rmse"
            metric_value = rmse
        metrics = Metrics(
            task=task,
            train_loss=train_loss,
            test_loss=test_loss,
            test_metric_name=metric_name,
            test_metric_value=metric_value,
        )

    bundle = ModelBundle(
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
    )
    return bundle, metrics


def _model_hidden_dim(bundle: ModelBundle) -> int:
    if bundle.model_config.get("training_mode") == "linear_glm_baseline":
        return 0
    return int(bundle.model_config.get("hidden_dim", 128))


def _model_num_hidden_layers(bundle: ModelBundle) -> int:
    if bundle.model_config.get("training_mode") == "linear_glm_baseline":
        return 0
    return int(bundle.model_config.get("num_hidden_layers", 2))


def _model_dropout(bundle: ModelBundle) -> float:
    if bundle.model_config.get("training_mode") == "linear_glm_baseline":
        return 0.0
    return float(bundle.model_config.get("dropout", 0.0))


def _distill_model_from_file_impl(
    data_path: str | Path,
    cfg: TrainingConfig,
    teacher_path: str | Path | None = None,
    teacher_bundle: ModelBundle | None = None,
    sheet_name: str | None = None,
    exclude_columns: list[str] | None = None,
    date_columns: list[str] | None = None,
    device: str | None = None,
    temperature: float = 2.5,
    alpha: float = 0.5,
    student_hidden_dim: int | None = None,
    student_num_hidden_layers: int | None = None,
    student_dropout: float | None = None,
) -> tuple[ModelBundle, Metrics]:
    _set_seed(cfg.random_seed)
    _ = device

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    if teacher_bundle is not None:
        teacher = teacher_bundle
    elif teacher_path is not None:
        teacher = load_bundle(teacher_path)
    else:
        raise ValueError("teacher_path or teacher_bundle is required.")
    task: TaskType = teacher.task if cfg.task == "auto" else cfg.task
    if task != teacher.task:
        raise ValueError("Requested task does not match teacher task.")

    rows = load_tabular_file(data_path, sheet_name=sheet_name)
    rows = expand_date_columns(rows, date_columns=date_columns)
    excluded = {column for column in (exclude_columns or []) if column}
    if cfg.target_column in excluded:
        raise ValueError("target_column cannot be excluded")
    if excluded:
        rows = [{k: v for k, v in row.items() if k not in excluded} for row in rows]
    x_rows, y_raw = split_features_target(rows, cfg.target_column)

    if task == "classification":
        y_labels = [str(v) for v in y_raw]
        unique_labels = set(y_labels)
        stratify: list[str] | None = y_labels if len(unique_labels) > 1 else None
    else:
        stratify = None

    x_train_rows, x_test_rows, y_train_raw, y_test_raw = train_test_split(
        x_rows,
        y_raw,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=stratify,
    )

    x_train_np = teacher.vectorizer.transform(x_train_rows).astype(np.float32)
    x_test_np = teacher.vectorizer.transform(x_test_rows).astype(np.float32)

    col_medians = np.nanmedian(np.where(np.isfinite(x_train_np), x_train_np, np.nan), axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    for col_idx in range(x_train_np.shape[1]):
        train_mask = ~np.isfinite(x_train_np[:, col_idx])
        x_train_np[train_mask, col_idx] = col_medians[col_idx]
        test_mask = ~np.isfinite(x_test_np[:, col_idx])
        x_test_np[test_mask, col_idx] = col_medians[col_idx]

    x_train_np = teacher.scaler.transform(x_train_np).astype(np.float32)
    x_test_np = teacher.scaler.transform(x_test_np).astype(np.float32)

    if task == "classification":
        if teacher.label_encoder is None:
            raise ValueError("Teacher model is missing a label encoder.")
        y_train_np = teacher.label_encoder.transform([str(v) for v in y_train_raw]).astype(np.int32)
        y_test_np = teacher.label_encoder.transform([str(v) for v in y_test_raw]).astype(np.int32)
        y_train = y_train_np
        y_test = y_test_np
    else:
        y_train_np = np.array([float(v) for v in y_train_raw], dtype=np.float32).reshape(-1, 1)
        y_test_np = np.array([float(v) for v in y_test_raw], dtype=np.float32).reshape(-1, 1)
        if teacher.target_scaler is not None:
            y_train_np = teacher.target_scaler.transform(y_train_np).astype(np.float32)
            y_test_np = teacher.target_scaler.transform(y_test_np).astype(np.float32)
        y_train = y_train_np
        y_test = y_test_np

    teacher_hidden_dim = _model_hidden_dim(teacher)
    teacher_layers = _model_num_hidden_layers(teacher)
    teacher_dropout = _model_dropout(teacher)

    student_hd = student_hidden_dim or max(16, teacher_hidden_dim // 2)
    student_layers = student_num_hidden_layers or max(1, teacher_layers - 1)
    student_do = student_dropout if student_dropout is not None else min(0.5, teacher_dropout + 0.05)

    student = _build_model(
        input_dim=int(x_train_np.shape[1]),
        output_dim=int(teacher.output_dim),
        task=task,
        training_mode=cfg.training_mode,
        hidden_dim=int(student_hd),
        num_hidden_layers=int(student_layers),
        dropout=float(student_do),
        learning_rate=cfg.learning_rate,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

    if task == "classification":
        hard_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        soft_loss_fn = tf.keras.losses.KLDivergence()
        for _ in range(cfg.epochs):
            idx = np.random.permutation(len(x_train_np))
            for start in range(0, len(idx), cfg.batch_size):
                batch = idx[start : start + cfg.batch_size]
                if len(batch) < 2:
                    continue
                xb = x_train_np[batch]
                yb = y_train[batch]
                with tf.GradientTape() as tape:
                    student_probs = student(xb, training=True)
                    teacher_probs = teacher.model(xb, training=False)
                    hard_loss = hard_loss_fn(yb, student_probs)
                    teacher_temp = tf.nn.softmax(tf.math.log(tf.clip_by_value(teacher_probs, 1e-7, 1.0)) / temperature, axis=1)
                    student_temp = tf.nn.softmax(tf.math.log(tf.clip_by_value(student_probs, 1e-7, 1.0)) / temperature, axis=1)
                    soft_loss = soft_loss_fn(teacher_temp, student_temp) * (temperature**2)
                    loss = alpha * hard_loss + (1 - alpha) * soft_loss
                grads = tape.gradient(loss, student.trainable_variables)
                optimizer.apply_gradients(zip(grads, student.trainable_variables))
    else:
        hard_loss_fn = tf.keras.losses.MeanSquaredError()
        for _ in range(cfg.epochs):
            idx = np.random.permutation(len(x_train_np))
            for start in range(0, len(idx), cfg.batch_size):
                batch = idx[start : start + cfg.batch_size]
                if len(batch) < 2:
                    continue
                xb = x_train_np[batch]
                yb = y_train[batch]
                with tf.GradientTape() as tape:
                    student_out = student(xb, training=True)
                    teacher_out = teacher.model(xb, training=False)
                    hard_loss = hard_loss_fn(yb, student_out)
                    soft_loss = hard_loss_fn(teacher_out, student_out)
                    loss = alpha * hard_loss + (1 - alpha) * soft_loss
                grads = tape.gradient(loss, student.trainable_variables)
                optimizer.apply_gradients(zip(grads, student.trainable_variables))

    train_eval = student.evaluate(x_train_np, y_train, verbose=0)
    test_eval = student.evaluate(x_test_np, y_test, verbose=0)
    train_loss = float(train_eval[0] if isinstance(train_eval, (list, tuple)) else train_eval)
    test_loss = float(test_eval[0] if isinstance(test_eval, (list, tuple)) else test_eval)

    if task == "classification":
        test_accuracy = float(test_eval[1] if isinstance(test_eval, (list, tuple)) and len(test_eval) > 1 else 0.0)
        metrics = Metrics(
            task=task,
            train_loss=train_loss,
            test_loss=test_loss,
            test_metric_name="accuracy",
            test_metric_value=test_accuracy,
        )
    else:
        preds = student.predict(x_test_np, verbose=0)
        if teacher.target_scaler is not None:
            preds_original = teacher.target_scaler.inverse_transform(preds)
            y_test_original = teacher.target_scaler.inverse_transform(y_test)
        else:
            preds_original = preds
            y_test_original = y_test
        rmse = float(np.sqrt(np.mean((preds_original - y_test_original) ** 2)))
        metrics = Metrics(
            task=task,
            train_loss=train_loss,
            test_loss=test_loss,
            test_metric_name="rmse",
            test_metric_value=rmse,
        )

    bundle = ModelBundle(
        model=student,
        task=task,
        vectorizer=teacher.vectorizer,
        scaler=teacher.scaler,
        feature_medians=teacher.feature_medians if teacher.feature_medians is not None else col_medians,
        label_encoder=teacher.label_encoder,
        target_scaler=teacher.target_scaler,
        target_column=cfg.target_column,
        input_dim=int(x_train_np.shape[1]),
        output_dim=int(teacher.output_dim),
        class_names=teacher.class_names,
        model_config={
            "training_mode": cfg.training_mode,
            "hidden_dim": int(student_hd),
            "num_hidden_layers": int(student_layers),
            "dropout": float(student_do),
        },
    )

    return bundle, metrics


def save_bundle(bundle: ModelBundle, output_dir: str | Path, metrics: Metrics | None = None) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "model_bundle.keras"
    bundle.model.save(model_path)

    payload = {
        "task": bundle.task,
        "target_column": bundle.target_column,
        "input_dim": bundle.input_dim,
        "output_dim": bundle.output_dim,
        "class_names": bundle.class_names,
        "vectorizer": bundle.vectorizer,
        "scaler": bundle.scaler,
        "feature_medians": bundle.feature_medians,
        "label_encoder": bundle.label_encoder,
        "target_scaler": bundle.target_scaler,
        "model_config": bundle.model_config,
    }

    with (out / "bundle_meta.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    if metrics is not None:
        (out / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    return model_path


def load_bundle(path: str | Path) -> ModelBundle:
    model_path = Path(path)
    out = model_path.parent
    meta_path = out / "bundle_meta.pkl"
    if not meta_path.exists():
        raise ValueError("TensorFlow bundle metadata not found.")

    with meta_path.open("rb") as handle:
        payload = pickle.load(handle)

    model = tf.keras.models.load_model(model_path)

    return ModelBundle(
        model=model,
        task=payload["task"],
        vectorizer=payload["vectorizer"],
        scaler=payload["scaler"],
        feature_medians=payload.get("feature_medians"),
        label_encoder=payload.get("label_encoder"),
        target_scaler=payload.get("target_scaler"),
        target_column=payload["target_column"],
        input_dim=int(payload["input_dim"]),
        output_dim=int(payload["output_dim"]),
        class_names=payload.get("class_names"),
        model_config=payload.get("model_config", {}),
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

    for col_idx in range(x_np.shape[1]):
        mask = ~np.isfinite(x_np[:, col_idx])
        x_np[mask, col_idx] = medians[col_idx]

    x_np = bundle.scaler.transform(x_np).astype(np.float32)
    outputs = bundle.model.predict(x_np, verbose=0)

    if bundle.task == "classification":
        indices = outputs.argmax(axis=1)
        if bundle.label_encoder is not None:
            return bundle.label_encoder.inverse_transform(indices).tolist()
        return indices.tolist()

    preds = outputs
    if bundle.target_scaler is not None:
        preds = bundle.target_scaler.inverse_transform(preds)
    return preds.squeeze(1).tolist()


def _safe_file_size(path: str | None) -> int | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    return int(file_path.stat().st_size)


def _parameter_count(model: tf.keras.Model) -> int:
    return int(model.count_params())


def _serialized_model_size_bytes(model: tf.keras.Model) -> int | None:
    try:
        payload = []
        for weights in model.get_weights():
            payload.append(weights.tobytes())
        return int(sum(len(chunk) for chunk in payload))
    except Exception:
        return None


def _store_in_memory_bundle(bundle: ModelBundle) -> str:
    return _BUNDLE_REGISTRY.store(bundle)


def _load_in_memory_bundle(run_id: str) -> ModelBundle | None:
    return _BUNDLE_REGISTRY.load(run_id)


def _handle_distill_request_impl(
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
) -> tuple[int, dict[str, Any]]:
    save_model = bool(payload.get("save_model", False))
    data_path = payload.get("data_path")
    dataset_id = payload.get("dataset_id")
    target_column = payload.get("target_column")
    if not data_path and dataset_id:
        resolved_path = resolve_dataset_path(str(dataset_id))
        if resolved_path is None:
            return 404, {"status": "error", "error": "Dataset not found in ai/ml/data."}
        data_path = str(resolved_path)

    teacher_model_path = payload.get("teacher_model_path")
    teacher_model_id = payload.get("teacher_model_id")
    teacher_run_id = payload.get("teacher_run_id")
    if not teacher_model_path and teacher_model_id:
        teacher_model_path = str(artifacts_dir / str(teacher_model_id) / "model_bundle.keras")

    if not data_path or not target_column:
        return 400, {
            "status": "error",
            "error": "target_column and either data_path or dataset_id are required.",
        }
    if not teacher_model_path and not teacher_run_id:
        return 400, {
            "status": "error",
            "error": "teacher_run_id, teacher_model_path, or teacher_model_id is required.",
        }

    model_id = (payload.get("model_id") or str(uuid.uuid4())) if save_model else None
    model_dir = (artifacts_dir / model_id) if model_id else None
    sheet_name = payload.get("sheet_name")

    raw_exclude_columns = payload.get("exclude_columns", [])
    if isinstance(raw_exclude_columns, str):
        exclude_columns = [part.strip() for part in raw_exclude_columns.split(",") if part.strip()]
    elif isinstance(raw_exclude_columns, list):
        exclude_columns = [str(item).strip() for item in raw_exclude_columns if str(item).strip()]
    else:
        return 400, {
            "status": "error",
            "error": "exclude_columns must be an array or comma-separated string.",
        }

    raw_date_columns = payload.get("date_columns", [])
    if isinstance(raw_date_columns, str):
        date_columns = [part.strip() for part in raw_date_columns.split(",") if part.strip()]
    elif isinstance(raw_date_columns, list):
        date_columns = [str(item).strip() for item in raw_date_columns if str(item).strip()]
    else:
        return 400, {
            "status": "error",
            "error": "date_columns must be an array or comma-separated string.",
        }

    try:
        test_size = float(payload.get("test_size", 0.2))
        epochs = int(payload.get("epochs", 60))
        batch_size = int(payload.get("batch_size", 64))
        learning_rate = float(payload.get("learning_rate", 1e-3))
        hidden_dim = (
            int(payload.get("student_hidden_dim"))
            if payload.get("student_hidden_dim") is not None
            else None
        )
        num_hidden_layers = (
            int(payload.get("student_num_hidden_layers"))
            if payload.get("student_num_hidden_layers") is not None
            else None
        )
        student_dropout = (
            float(payload.get("student_dropout"))
            if payload.get("student_dropout") is not None
            else None
        )
        temperature = float(payload.get("temperature", 2.5))
        alpha = float(payload.get("alpha", 0.5))
        random_seed = int(payload.get("random_seed", 42))
    except (TypeError, ValueError):
        return 400, {"status": "error", "error": "Invalid numeric distillation parameters."}

    if not 0 < test_size < 1:
        return 400, {"status": "error", "error": "test_size must be > 0 and < 1."}
    if not 1 <= epochs <= 500:
        return 400, {"status": "error", "error": "epochs must be between 1 and 500."}
    if not 1 <= batch_size <= 200:
        return 400, {"status": "error", "error": "batch_size must be between 1 and 200."}
    if hidden_dim is not None and not 8 <= hidden_dim <= 500:
        return 400, {"status": "error", "error": "student_hidden_dim must be between 8 and 500."}
    if num_hidden_layers is not None and not 1 <= num_hidden_layers <= 15:
        return 400, {"status": "error", "error": "student_num_hidden_layers must be between 1 and 15."}
    if not 0 < learning_rate <= 1:
        return 400, {"status": "error", "error": "learning_rate must be > 0 and <= 1."}
    if not 0 < temperature <= 20:
        return 400, {"status": "error", "error": "temperature must be > 0 and <= 20."}
    if not 0 <= alpha <= 1:
        return 400, {"status": "error", "error": "alpha must be between 0 and 1."}
    training_mode = str(payload.get("training_mode", "mlp_dense"))
    if training_mode == "mlp":
        training_mode = "mlp_dense"
    if training_mode not in {
        "mlp_dense",
        "linear_glm_baseline",
        "wide_and_deep",
        "imbalance_aware",
        "quantile_regression",
        "calibrated_classifier",
        "entity_embeddings",
        "autoencoder_head",
        "multi_task_learning",
        "time_aware_tabular",
    }:
        return 400, {
            "status": "error",
            "error": "training_mode must be 'mlp_dense', 'linear_glm_baseline', 'wide_and_deep', 'imbalance_aware', 'quantile_regression', 'calibrated_classifier', 'entity_embeddings', 'autoencoder_head', 'multi_task_learning', or 'time_aware_tabular'.",
        }
    if training_mode in {
        "imbalance_aware",
        "quantile_regression",
        "calibrated_classifier",
        "entity_embeddings",
        "autoencoder_head",
        "multi_task_learning",
        "time_aware_tabular",
    }:
        return 400, {
            "status": "error",
            "error": f"Distillation is not yet supported for training_mode '{training_mode}'. Use mlp_dense, linear_glm_baseline, or wide_and_deep.",
        }

    cfg = TrainingConfig(
        target_column=str(target_column),
        training_mode=training_mode,
        task=payload.get("task", "auto"),
        test_size=test_size,
        random_seed=random_seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim or 128,
        num_hidden_layers=num_hidden_layers or 2,
        dropout=student_dropout if student_dropout is not None else 0.1,
    )

    try:
        teacher_bundle = _load_in_memory_bundle(str(teacher_run_id)) if teacher_run_id else None
        if teacher_run_id and teacher_bundle is None:
            return 404, {"status": "error", "error": "Teacher run not found or expired."}
        bundle, metrics = _distill_model_from_file_impl(
            data_path=data_path,
            cfg=cfg,
            teacher_path=teacher_model_path,
            teacher_bundle=teacher_bundle,
            sheet_name=sheet_name,
            exclude_columns=exclude_columns,
            date_columns=date_columns,
            device=payload.get("device"),
            temperature=temperature,
            alpha=alpha,
            student_hidden_dim=hidden_dim,
            student_num_hidden_layers=num_hidden_layers,
            student_dropout=student_dropout,
        )
        model_path = save_bundle(bundle, model_dir, metrics) if save_model and model_dir else None
        teacher_model_size_bytes = _safe_file_size(str(teacher_model_path))
        student_model_size_bytes = _safe_file_size(str(model_path) if model_path else None)
        size_saved_bytes = (
            teacher_model_size_bytes - student_model_size_bytes
            if teacher_model_size_bytes is not None and student_model_size_bytes is not None
            else None
        )
        size_saved_percent = (
            (float(size_saved_bytes) / float(teacher_model_size_bytes)) * 100.0
            if size_saved_bytes is not None and teacher_model_size_bytes and teacher_model_size_bytes > 0
            else None
        )
        teacher_model_for_stats = (
            teacher_bundle if teacher_bundle is not None else load_bundle(str(teacher_model_path))
        )
        teacher_param_count = _parameter_count(teacher_model_for_stats.model)
        student_param_count = _parameter_count(bundle.model)
        param_saved_count = teacher_param_count - student_param_count
        param_saved_percent = (
            (float(param_saved_count) / float(teacher_param_count)) * 100.0
            if teacher_param_count > 0
            else None
        )
        if teacher_model_size_bytes is None:
            teacher_model_size_bytes = _serialized_model_size_bytes(teacher_model_for_stats.model)
        if student_model_size_bytes is None:
            student_model_size_bytes = _serialized_model_size_bytes(bundle.model)
            if teacher_model_size_bytes is not None and student_model_size_bytes is not None:
                size_saved_bytes = teacher_model_size_bytes - student_model_size_bytes
                size_saved_percent = (
                    (float(size_saved_bytes) / float(teacher_model_size_bytes)) * 100.0
                    if teacher_model_size_bytes > 0
                    else None
                )
        run_id = _store_in_memory_bundle(bundle)
        return 200, {
            "status": "ok",
            "run_id": run_id,
            "model_id": model_id,
            "model_path": str(model_path) if model_path else None,
            "metrics": asdict(metrics),
            "teacher_input_dim": int(teacher_model_for_stats.input_dim),
            "teacher_output_dim": int(teacher_model_for_stats.output_dim),
            "student_input_dim": int(bundle.input_dim),
            "student_output_dim": int(bundle.output_dim),
            "teacher_model_size_bytes": teacher_model_size_bytes,
            "student_model_size_bytes": student_model_size_bytes,
            "size_saved_bytes": size_saved_bytes,
            "size_saved_percent": size_saved_percent,
            "teacher_param_count": teacher_param_count,
            "student_param_count": student_param_count,
            "param_saved_count": param_saved_count,
            "param_saved_percent": param_saved_percent,
        }
    except Exception as exc:
        return 400, {"status": "error", "error": str(exc)}


def distill_model_from_file(**kwargs: Any) -> tuple[ModelBundle, Metrics]:
    return shared_distill_model_from_file("tensorflow", **kwargs)


def handle_distill_request(
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
) -> tuple[int, dict[str, Any]]:
    return shared_handle_distill_request("tensorflow", payload, resolve_dataset_path, artifacts_dir)


def handle_train_request(
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
) -> tuple[int, dict[str, Any]]:
    data_path = payload.get("data_path")
    dataset_id = payload.get("dataset_id")
    target_column = payload.get("target_column")
    if not data_path and dataset_id:
        resolved_path = resolve_dataset_path(str(dataset_id))
        if resolved_path is None:
            return 404, {"status": "error", "error": "Dataset not found in ai/ml/data."}
        data_path = str(resolved_path)

    if not data_path or not target_column:
        return 400, {
            "status": "error",
            "error": "target_column and either data_path or dataset_id are required.",
        }

    save_model = bool(payload.get("save_model", False))
    model_id = (payload.get("model_id") or str(uuid.uuid4())) if save_model else None
    model_dir = (artifacts_dir / model_id) if model_id else None
    sheet_name = payload.get("sheet_name")

    raw_exclude_columns = payload.get("exclude_columns", [])
    if isinstance(raw_exclude_columns, str):
        exclude_columns = [part.strip() for part in raw_exclude_columns.split(",") if part.strip()]
    elif isinstance(raw_exclude_columns, list):
        exclude_columns = [str(item).strip() for item in raw_exclude_columns if str(item).strip()]
    else:
        return 400, {
            "status": "error",
            "error": "exclude_columns must be an array or comma-separated string.",
        }

    raw_date_columns = payload.get("date_columns", [])
    if isinstance(raw_date_columns, str):
        date_columns = [part.strip() for part in raw_date_columns.split(",") if part.strip()]
    elif isinstance(raw_date_columns, list):
        date_columns = [str(item).strip() for item in raw_date_columns if str(item).strip()]
    else:
        return 400, {
            "status": "error",
            "error": "date_columns must be an array or comma-separated string.",
        }

    try:
        test_size = float(payload.get("test_size", 0.2))
        epochs = int(payload.get("epochs", 500))
        batch_size = int(payload.get("batch_size", 64))
        learning_rate = float(payload.get("learning_rate", 1e-3))
        training_mode = str(payload.get("training_mode", "mlp_dense"))
        hidden_dim = int(payload.get("hidden_dim", 128))
        num_hidden_layers = int(payload.get("num_hidden_layers", 2))
        dropout = float(payload.get("dropout", 0.1))
    except (TypeError, ValueError):
        return 400, {"status": "error", "error": "Invalid numeric training parameters."}

    if not 0 < test_size < 1:
        return 400, {"status": "error", "error": "test_size must be > 0 and < 1."}
    if not 1 <= epochs <= 500:
        return 400, {"status": "error", "error": "epochs must be between 1 and 500."}
    if not 1 <= batch_size <= 200:
        return 400, {"status": "error", "error": "batch_size must be between 1 and 200."}
    if not 0 < learning_rate <= 1:
        return 400, {"status": "error", "error": "learning_rate must be > 0 and <= 1."}
    if training_mode == "mlp":
        training_mode = "mlp_dense"
    if training_mode not in {
        "mlp_dense",
        "linear_glm_baseline",
        "wide_and_deep",
        "imbalance_aware",
        "quantile_regression",
        "calibrated_classifier",
        "entity_embeddings",
        "autoencoder_head",
        "multi_task_learning",
        "time_aware_tabular",
    }:
        return 400, {
            "status": "error",
            "error": "training_mode must be 'mlp_dense', 'linear_glm_baseline', 'wide_and_deep', 'imbalance_aware', 'quantile_regression', 'calibrated_classifier', 'entity_embeddings', 'autoencoder_head', 'multi_task_learning', or 'time_aware_tabular'.",
        }
    if training_mode != "linear_glm_baseline" and not 8 <= hidden_dim <= 500:
        return 400, {"status": "error", "error": "hidden_dim must be between 8 and 500."}
    if training_mode != "linear_glm_baseline" and not 1 <= num_hidden_layers <= 15:
        return 400, {"status": "error", "error": "num_hidden_layers must be between 1 and 15."}
    if training_mode != "linear_glm_baseline" and not 0 <= dropout <= 0.9:
        return 400, {"status": "error", "error": "dropout must be between 0 and 0.9."}

    cfg = TrainingConfig(
        target_column=str(target_column),
        training_mode=training_mode,
        task=payload.get("task", "auto"),
        test_size=test_size,
        random_seed=int(payload.get("random_seed", 42)),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
    )

    try:
        bundle, metrics = train_model_from_file(
            data_path=data_path,
            cfg=cfg,
            sheet_name=sheet_name,
            exclude_columns=exclude_columns,
            date_columns=date_columns,
            device=payload.get("device"),
        )
        run_id = _store_in_memory_bundle(bundle)
        model_path = save_bundle(bundle, model_dir, metrics) if save_model and model_dir else None
        return 200, {
            "status": "ok",
            "run_id": run_id,
            "model_id": model_id,
            "model_path": str(model_path) if model_path else None,
            "metrics": asdict(metrics),
        }
    except Exception as exc:
        return 400, {"status": "error", "error": str(exc)}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/test a TensorFlow MLP on CSV/XLS/XLSX tabular data")
    parser.add_argument("--data", required=True, help="Path to .csv, .xls, or .xlsx file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    parser.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default=None, help="Directory to save model bundle + metrics")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = TrainingConfig(
        target_column=args.target,
        task=args.task,
        test_size=args.test_size,
        random_seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.hidden_layers,
        dropout=args.dropout,
    )

    bundle, metrics = train_model_from_file(
        data_path=args.data,
        cfg=cfg,
        sheet_name=args.sheet,
    )

    print(json.dumps(asdict(metrics), indent=2))

    if args.save_dir:
        model_path = save_bundle(bundle, args.save_dir, metrics)
        print(f"Saved bundle to {model_path}")


if __name__ == "__main__":
    main()
