from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from ...core.preprocessing import impute_train_test_non_finite
    from ...file_util import load_tabular_file, split_features_target
    from ...ml_util import TaskType, expand_date_columns
except ImportError:  # pragma: no cover
    from core.preprocessing import impute_train_test_non_finite  # type: ignore
    from file_util import load_tabular_file, split_features_target  # type: ignore
    from ml_util import TaskType, expand_date_columns  # type: ignore

from ...core.types import Metrics, ModelBundle, TrainingConfig
from .models import build_model, model_dropout, model_hidden_dim, model_num_hidden_layers, set_seed
from .serialization import load_bundle


def distill_model_from_file(
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
    set_seed(cfg.random_seed)
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

    from sklearn.model_selection import train_test_split

    x_train_rows, x_test_rows, y_train_raw, y_test_raw = train_test_split(
        x_rows,
        y_raw,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=stratify,
    )

    x_train_np = teacher.vectorizer.transform(x_train_rows).astype(np.float32)
    x_test_np = teacher.vectorizer.transform(x_test_rows).astype(np.float32)
    x_train_np, x_test_np, col_medians = impute_train_test_non_finite(x_train_np, x_test_np)
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

    teacher_hidden_dim = model_hidden_dim(teacher)
    teacher_layers = model_num_hidden_layers(teacher)
    teacher_dropout = model_dropout(teacher)

    student_hd = student_hidden_dim or max(16, teacher_hidden_dim // 2)
    student_layers = student_num_hidden_layers or max(1, teacher_layers - 1)
    student_do = student_dropout if student_dropout is not None else min(0.5, teacher_dropout + 0.05)

    student = build_model(
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

    return (
        ModelBundle(
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
        ),
        metrics,
    )
