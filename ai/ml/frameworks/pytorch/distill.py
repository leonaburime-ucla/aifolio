from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

try:
    from ...core.preprocessing import impute_train_test_non_finite
    from ...file_util import load_tabular_file, split_features_target
    from ...ml_util import TaskType, batch_indices, expand_date_columns
except ImportError:  # pragma: no cover
    from core.preprocessing import impute_train_test_non_finite  # type: ignore
    from file_util import load_tabular_file, split_features_target  # type: ignore
    from ml_util import TaskType, batch_indices, expand_date_columns  # type: ignore

from ...core.types import Metrics, ModelBundle, TrainingConfig
from .models import build_model, model_dropout, model_hidden_dim, model_num_hidden_layers
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
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

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
    x_train_np, x_test_np, col_medians = impute_train_test_non_finite(x_train_np, x_test_np)
    x_train_np = teacher.scaler.transform(x_train_np).astype(np.float32)
    x_test_np = teacher.scaler.transform(x_test_np).astype(np.float32)

    if task == "classification":
        if teacher.label_encoder is None:
            raise ValueError("Teacher model is missing a label encoder.")
        y_train_np = teacher.label_encoder.transform([str(v) for v in y_train_raw]).astype(np.int64)
        y_test_np = teacher.label_encoder.transform([str(v) for v in y_test_raw]).astype(np.int64)
    else:
        y_train_np = np.array([float(v) for v in y_train_raw], dtype=np.float32).reshape(-1, 1)
        y_test_np = np.array([float(v) for v in y_test_raw], dtype=np.float32).reshape(-1, 1)
        if teacher.target_scaler is not None:
            y_train_np = teacher.target_scaler.transform(y_train_np).astype(np.float32)
            y_test_np = teacher.target_scaler.transform(y_test_np).astype(np.float32)

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    if task == "classification":
        y_train = torch.tensor(y_train_np, dtype=torch.long)
        y_test = torch.tensor(y_test_np, dtype=torch.long)
    else:
        y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

    teacher_hidden_dim = model_hidden_dim(teacher.model)
    teacher_layers = model_num_hidden_layers(teacher.model)
    teacher_dropout = model_dropout(teacher.model)

    student_hd = student_hidden_dim or max(16, teacher_hidden_dim // 2)
    student_layers = student_num_hidden_layers or max(1, teacher_layers - 1)
    student_do = student_dropout if student_dropout is not None else min(0.5, teacher_dropout + 0.05)

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    teacher.model.to(torch_device)
    teacher.model.eval()

    student = build_model(
        input_dim=int(x_train.shape[1]),
        output_dim=int(teacher.output_dim),
        training_mode=cfg.training_mode,
        hidden_dim=int(student_hd),
        num_hidden_layers=int(student_layers),
        dropout=float(student_do),
    ).to(torch_device)

    x_train = x_train.to(torch_device)
    x_test = x_test.to(torch_device)
    y_train = y_train.to(torch_device)
    y_test = y_test.to(torch_device)

    hard_criterion: nn.Module = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    student.train()
    update_steps = 0
    for _ in range(cfg.epochs):
        for batch in batch_indices(len(x_train), cfg.batch_size):
            xb = x_train[batch]
            yb = y_train[batch]
            if xb.shape[0] < 2:
                continue

            optimizer.zero_grad()
            student_out = student(xb)
            with torch.no_grad():
                teacher_out = teacher.model(xb)

            if task == "classification":
                hard_loss = hard_criterion(student_out, yb)
                soft_loss = nn.functional.kl_div(
                    nn.functional.log_softmax(student_out / temperature, dim=1),
                    nn.functional.softmax(teacher_out / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature**2)
            else:
                hard_loss = hard_criterion(student_out, yb)
                soft_loss = nn.functional.mse_loss(student_out, teacher_out)

            loss = alpha * hard_loss + (1 - alpha) * soft_loss
            loss.backward()
            optimizer.step()
            update_steps += 1

    if update_steps == 0:
        raise ValueError("No valid distillation batches. Increase training rows or reduce batch size / test_size.")

    student.eval()
    with torch.no_grad():
        train_loss = hard_criterion(student(x_train), y_train).item()
        test_loss = hard_criterion(student(x_test), y_test).item()

        if task == "classification":
            preds = student(x_test).argmax(dim=1)
            accuracy = (preds == y_test).float().mean().item()
            metrics = Metrics(
                task=task,
                train_loss=float(train_loss),
                test_loss=float(test_loss),
                test_metric_name="accuracy",
                test_metric_value=float(accuracy),
            )
        else:
            preds = student(x_test)
            preds_np = preds.cpu().numpy()
            y_test_np_t = y_test.cpu().numpy()
            if teacher.target_scaler is not None:
                preds_original = teacher.target_scaler.inverse_transform(preds_np)
                y_test_original = teacher.target_scaler.inverse_transform(y_test_np_t)
            else:
                preds_original = preds_np
                y_test_original = y_test_np_t
            rmse = float(np.sqrt(np.mean((preds_original - y_test_original) ** 2)))
            metrics = Metrics(
                task=task,
                train_loss=float(train_loss),
                test_loss=float(test_loss),
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
            input_dim=int(x_train.shape[1]),
            output_dim=int(teacher.output_dim),
            class_names=teacher.class_names,
        ),
        metrics,
    )
