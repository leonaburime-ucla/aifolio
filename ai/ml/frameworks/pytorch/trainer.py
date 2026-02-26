from __future__ import annotations

"""PyTorch runtime trainer implementation."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from ...file_util import coerce_value, load_tabular_file, split_features_target
    from ...ml_util import TaskType, batch_indices, expand_date_columns, infer_task
except ImportError:  # pragma: no cover
    from file_util import coerce_value, load_tabular_file, split_features_target  # type: ignore
    from ml_util import TaskType, batch_indices, expand_date_columns, infer_task  # type: ignore

from ...core.preprocessing import impute_non_finite_with_reference_medians
from ...core.types import Metrics, ModelBundle, TrainingConfig
from .data import prepare_tensors
from .distill import distill_model_from_file
from .models import build_model, compute_class_weights, compute_loss
from .serialization import load_bundle as _load_bundle, save_bundle


def train_model_from_file(
    data_path: str | Path,
    cfg: TrainingConfig,
    sheet_name: str | None = None,
    exclude_columns: list[str] | None = None,
    date_columns: list[str] | None = None,
    device: str | None = None,
) -> tuple[ModelBundle, Metrics]:
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

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
    ) = prepare_tensors(x_rows, y_raw, task, cfg)

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(
        input_dim=input_dim,
        output_dim=output_dim,
        training_mode=cfg.training_mode,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        dropout=cfg.dropout,
    ).to(torch_device)

    x_train = x_train.to(torch_device)
    x_test = x_test.to(torch_device)
    y_train = y_train.to(torch_device)
    y_test = y_test.to(torch_device)

    if task == "classification":
        if cfg.training_mode == "imbalance_aware":
            class_weights = compute_class_weights(y_train, output_dim=output_dim, device=torch_device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif cfg.training_mode == "calibrated_classifier":
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    tree_teacher_probs: torch.Tensor | None = None
    tree_teacher_preds: torch.Tensor | None = None
    tree_temperature = 2.0
    if cfg.training_mode == "tree_teacher_distillation":
        x_train_np = x_train.detach().cpu().numpy()
        if task == "classification":
            y_train_np = y_train.detach().cpu().numpy()
            teacher = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=cfg.random_seed, n_jobs=-1)
            teacher.fit(x_train_np, y_train_np)
            probs = teacher.predict_proba(x_train_np)
            tree_teacher_probs = torch.tensor(np.clip(probs, 1e-6, 1.0), dtype=torch.float32, device=torch_device)
        else:
            y_train_np = y_train.detach().cpu().numpy().reshape(-1)
            teacher = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=cfg.random_seed, n_jobs=-1)
            teacher.fit(x_train_np, y_train_np)
            preds = teacher.predict(x_train_np).reshape(-1, 1)
            tree_teacher_preds = torch.tensor(preds, dtype=torch.float32, device=torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    model.train()
    update_steps = 0
    for _ in range(cfg.epochs):
        for batch in batch_indices(len(x_train), cfg.batch_size):
            xb = x_train[batch]
            yb = y_train[batch]
            if xb.shape[0] < 2:
                continue

            optimizer.zero_grad()
            if cfg.training_mode == "tree_teacher_distillation":
                student_logits = model(xb)
                hard_loss = compute_loss(model, xb, yb, criterion)
                if task == "classification":
                    assert tree_teacher_probs is not None
                    soft_targets = tree_teacher_probs[batch]
                    soft_loss = nn.functional.kl_div(
                        nn.functional.log_softmax(student_logits / tree_temperature, dim=1),
                        soft_targets,
                        reduction="batchmean",
                    ) * (tree_temperature**2)
                else:
                    assert tree_teacher_preds is not None
                    soft_loss = nn.functional.mse_loss(student_logits, tree_teacher_preds[batch])
                loss = 0.5 * hard_loss + 0.5 * soft_loss
            else:
                loss = compute_loss(model, xb, yb, criterion)
            loss.backward()
            optimizer.step()
            update_steps += 1

    if update_steps == 0:
        raise ValueError("No valid training batches. Increase training rows or reduce batch size / test_size.")

    model.eval()
    with torch.no_grad():
        train_loss = compute_loss(model, x_train, y_train, criterion).item()
        test_loss = compute_loss(model, x_test, y_test, criterion).item()

        if task == "classification":
            preds = model(x_test).argmax(dim=1)
            accuracy = (preds == y_test).float().mean().item()
            metrics = Metrics(
                task=task,
                train_loss=float(train_loss),
                test_loss=float(test_loss),
                test_metric_name="accuracy",
                test_metric_value=float(accuracy),
            )
        else:
            preds = model(x_test)
            preds_np = preds.cpu().numpy()
            y_test_np = y_test.cpu().numpy()
            if target_scaler is not None:
                preds_original = target_scaler.inverse_transform(preds_np)
                y_test_original = target_scaler.inverse_transform(y_test_np)
            else:
                preds_original = preds_np
                y_test_original = y_test_np
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
        ),
        metrics,
    )


def predict_rows(bundle: ModelBundle, rows: list[dict[str, Any]], device: str | None = None) -> list[Any]:
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

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bundle.model.to(torch_device)
    bundle.model.eval()

    x = torch.tensor(x_np, dtype=torch.float32, device=torch_device)
    with torch.no_grad():
        outputs = bundle.model(x)

    if bundle.task == "classification":
        indices = outputs.argmax(dim=1).cpu().numpy().tolist()
        if bundle.label_encoder is not None:
            return bundle.label_encoder.inverse_transform(np.array(indices)).tolist()
        return indices

    preds = outputs.cpu().numpy()
    if bundle.target_scaler is not None:
        preds = bundle.target_scaler.inverse_transform(preds)
    return preds.squeeze(1).tolist()


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
