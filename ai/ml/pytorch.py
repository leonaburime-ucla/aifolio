from __future__ import annotations

import argparse
import io
import json
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn as nn
try:
    from .distill import (
        InMemoryBundleRegistry,
        distill_model_from_file as shared_distill_model_from_file,
        handle_distill_request as shared_handle_distill_request,
    )
    from .file_util import coerce_value, load_tabular_file, split_features_target
    from .ml_util import (
        TaskType,
        batch_indices,
        expand_date_columns,
        infer_task,
    )
except ImportError:  # pragma: no cover
    from distill import (
        InMemoryBundleRegistry,
        distill_model_from_file as shared_distill_model_from_file,
        handle_distill_request as shared_handle_distill_request,
    )
    from file_util import coerce_value, load_tabular_file, split_features_target
    from ml_util import (
        TaskType,
        batch_indices,
        expand_date_columns,
        infer_task,
    )
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


TrainingMode = Literal[
    "mlp_dense",
    "linear_glm_baseline",
    "tabresnet",
    "imbalance_aware",
    "calibrated_classifier",
    "tree_teacher_distillation",
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
    model: nn.Module
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


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        in_features = input_dim
        for _ in range(max(1, num_hidden_layers)):
            layers.append(nn.Linear(in_features, hidden_dim))
            # BatchNorm stabilizes training by normalizing layer outputs to zero mean
            # and unit variance. This prevents internal covariate shift and allows
            # higher learning rates without gradient explosion.
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearBaseline(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity
        out = torch.relu(out)
        return out


class TabResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        blocks = [ResidualBlock(hidden_dim, dropout) for _ in range(max(1, num_hidden_layers))]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input(x)
        out = self.input_bn(out)
        out = torch.relu(out)
        out = self.blocks(out)
        out = self.head(out)
        return out


def build_model(
    input_dim: int,
    output_dim: int,
    training_mode: TrainingMode,
    hidden_dim: int,
    num_hidden_layers: int,
    dropout: float,
) -> nn.Module:
    if training_mode == "linear_glm_baseline":
        return LinearBaseline(input_dim=input_dim, output_dim=output_dim)
    if training_mode == "tabresnet":
        return TabResNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
    return MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
    )


def _prepare_tensors(
    x_rows: list[dict[str, Any]],
    y_raw: list[Any],
    task: TaskType,
    cfg: TrainingConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
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

    # Impute missing values (NaN/inf) with training set column medians.
    # Why median instead of mean? Median is robust to outliers - a few extreme
    # values won't skew the imputed value. We compute medians from training data
    # only to prevent data leakage from test set. Test set uses the same medians
    # computed from training data.
    col_medians = np.nanmedian(np.where(np.isfinite(x_train_np), x_train_np, np.nan), axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)  # fallback to 0 if entire column is NaN
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
        y_train_np = encoder.fit_transform([str(v) for v in y_train_raw]).astype(np.int64)
        y_test_np = encoder.transform([str(v) for v in y_test_raw]).astype(np.int64)
        output_dim = int(len(encoder.classes_))
        class_names = [str(c) for c in encoder.classes_]
    else:
        encoder = None
        # CRITICAL: Scale the target variable for regression tasks.
        # Without scaling, targets like house prices ($100k-$750k) cause MSELoss to
        # compute (prediction - 750000)^2 = billions, leading to gradient explosion
        # and meaningless loss values. StandardScaler normalizes to ~0 mean, ~1 std,
        # making loss values interpretable and gradients stable. We inverse-transform
        # predictions later to get values in original units (dollars).
        target_scaler = StandardScaler()
        y_train_np = np.array([float(v) for v in y_train_raw], dtype=np.float32).reshape(-1, 1)
        y_test_np = np.array([float(v) for v in y_test_raw], dtype=np.float32).reshape(-1, 1)
        y_train_np = target_scaler.fit_transform(y_train_np).astype(np.float32).flatten()
        y_test_np = target_scaler.transform(y_test_np).astype(np.float32).flatten()
        output_dim = 1
        class_names = None

    x_train_t = torch.tensor(x_train_np, dtype=torch.float32)
    x_test_t = torch.tensor(x_test_np, dtype=torch.float32)

    if task == "classification":
        y_train_t = torch.tensor(y_train_np, dtype=torch.long)
        y_test_t = torch.tensor(y_test_np, dtype=torch.long)
    else:
        y_train_t = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
        y_test_t = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

    input_dim = x_train_t.shape[1]

    return (
        x_train_t,
        x_test_t,
        y_train_t,
        y_test_t,
        vectorizer,
        scaler,
        col_medians,
        encoder,
        target_scaler,
        input_dim,
        output_dim,
        class_names,
    )


def _compute_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    task: TaskType,
    criterion: nn.Module,
) -> torch.Tensor:
    logits = model(x)
    if task == "classification":
        return criterion(logits, y)
    return criterion(logits, y)


def _compute_class_weights(y_train: torch.Tensor, output_dim: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(y_train.detach().cpu(), minlength=output_dim).float()
    safe_counts = torch.clamp(counts, min=1.0)
    total = float(torch.sum(safe_counts).item())
    weights = total / (float(output_dim) * safe_counts)
    return weights.to(device)


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
    ) = _prepare_tensors(x_rows, y_raw, task, cfg)

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
            class_weights = _compute_class_weights(y_train, output_dim=output_dim, device=torch_device)
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
            teacher = RandomForestClassifier(
                n_estimators=120,
                max_depth=8,
                random_state=cfg.random_seed,
                n_jobs=-1,
            )
            teacher.fit(x_train_np, y_train_np)
            probs = teacher.predict_proba(x_train_np)
            tree_teacher_probs = torch.tensor(
                np.clip(probs, 1e-6, 1.0),
                dtype=torch.float32,
                device=torch_device,
            )
        else:
            y_train_np = y_train.detach().cpu().numpy().reshape(-1)
            teacher = RandomForestRegressor(
                n_estimators=120,
                max_depth=10,
                random_state=cfg.random_seed,
                n_jobs=-1,
            )
            teacher.fit(x_train_np, y_train_np)
            preds = teacher.predict(x_train_np).reshape(-1, 1)
            tree_teacher_preds = torch.tensor(preds, dtype=torch.float32, device=torch_device)
    # weight_decay adds L2 regularization to prevent overfitting by penalizing
    # large weights. This encourages the model to learn simpler patterns that
    # generalize better to unseen data. 1e-4 is a moderate regularization strength.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    model.train()
    update_steps = 0
    for _ in range(cfg.epochs):
        for batch in batch_indices(len(x_train), cfg.batch_size):
            xb = x_train[batch]
            yb = y_train[batch]
            # BatchNorm requires at least 2 samples in training mode.
            if xb.shape[0] < 2:
                continue

            optimizer.zero_grad()
            if cfg.training_mode == "tree_teacher_distillation":
                student_logits = model(xb)
                hard_loss = _compute_loss(model, xb, yb, task, criterion)
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
                loss = _compute_loss(model, xb, yb, task, criterion)
            loss.backward()
            optimizer.step()
            update_steps += 1

    if update_steps == 0:
        raise ValueError(
            "No valid training batches. Increase training rows or reduce batch size / test_size."
        )

    model.eval()
    with torch.no_grad():
        train_loss = _compute_loss(model, x_train, y_train, task, criterion).item()
        test_loss = _compute_loss(model, x_test, y_test, task, criterion).item()

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
            # Inverse transform predictions and targets back to original scale.
            # Since we trained on normalized targets (~0 mean, ~1 std), the model
            # outputs are also in normalized space. To report RMSE in meaningful
            # units (e.g., dollars for house prices), we convert back to original
            # scale. An RMSE of $20k is interpretable; an RMSE of 0.15 is not.
            preds_np = preds.cpu().numpy()
            y_test_np = y_test.cpu().numpy()
            if target_scaler is not None:
                preds_original = target_scaler.inverse_transform(preds_np)
                y_test_original = target_scaler.inverse_transform(y_test_np)
            else:
                preds_original = preds_np
                y_test_original = y_test_np
            rmse = float(np.sqrt(np.mean((preds_original - y_test_original) ** 2)))
            metric_name = "rmse"
            metric_value = rmse
            metrics = Metrics(
                task=task,
                train_loss=float(train_loss),
                test_loss=float(test_loss),
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
    )

    return bundle, metrics


def _model_hidden_dim(model: nn.Module) -> int:
    if isinstance(model, LinearBaseline):
        return 0
    if isinstance(model, MLP):
        first_linear = next((layer for layer in model.net if isinstance(layer, nn.Linear)), None)
        if first_linear is not None:
            return int(first_linear.out_features)
    if isinstance(model, TabResNet):
        return int(model.input.out_features)
    return 128


def _model_num_hidden_layers(model: nn.Module) -> int:
    if isinstance(model, LinearBaseline):
        return 0
    if isinstance(model, MLP):
        linear_count = sum(1 for layer in model.net if isinstance(layer, nn.Linear))
        return max(1, linear_count - 1)
    if isinstance(model, TabResNet):
        return int(len(model.blocks))
    return 2


def _model_dropout(model: nn.Module) -> float:
    if isinstance(model, LinearBaseline):
        return 0.0
    if isinstance(model, MLP):
        dropout_layer = next((layer for layer in model.net if isinstance(layer, nn.Dropout)), None)
        if dropout_layer is not None:
            return float(dropout_layer.p)
    if isinstance(model, TabResNet):
        first_block = next(iter(model.blocks), None)
        if first_block is not None and isinstance(first_block.dropout, nn.Dropout):
            return float(first_block.dropout.p)
    return 0.0


def _model_training_mode(model: nn.Module) -> TrainingMode:
    if isinstance(model, LinearBaseline):
        return "linear_glm_baseline"
    if isinstance(model, TabResNet):
        return "tabresnet"
    return "mlp_dense"


def _safe_file_size(path: str | None) -> int | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    return int(file_path.stat().st_size)


def _parameter_count(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def _serialized_model_size_bytes(model: nn.Module) -> int | None:
    buffer = io.BytesIO()
    try:
        torch.save(model.state_dict(), buffer)
        return int(buffer.tell())
    except Exception:
        return None


def _store_in_memory_bundle(bundle: ModelBundle) -> str:
    return _BUNDLE_REGISTRY.store(bundle)


def _load_in_memory_bundle(run_id: str) -> ModelBundle | None:
    return _BUNDLE_REGISTRY.load(run_id)


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

    # Impute missing values using training set medians (same as train_model_from_file).
    # Teacher's scaler expects clean data, so we must handle NaN/inf before scaling.
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
        y_train_np = teacher.label_encoder.transform([str(v) for v in y_train_raw]).astype(np.int64)
        y_test_np = teacher.label_encoder.transform([str(v) for v in y_test_raw]).astype(np.int64)
    else:
        y_train_np = np.array([float(v) for v in y_train_raw], dtype=np.float32).reshape(-1, 1)
        y_test_np = np.array([float(v) for v in y_test_raw], dtype=np.float32).reshape(-1, 1)
        if teacher.target_scaler is not None:
            y_train_np = teacher.target_scaler.transform(y_train_np).astype(np.float32)
            y_test_np = teacher.target_scaler.transform(y_test_np).astype(np.float32)
        y_train_np = y_train_np.flatten()
        y_test_np = y_test_np.flatten()

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    if task == "classification":
        y_train = torch.tensor(y_train_np, dtype=torch.long)
        y_test = torch.tensor(y_test_np, dtype=torch.long)
    else:
        y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

    teacher_hidden_dim = _model_hidden_dim(teacher.model)
    teacher_layers = _model_num_hidden_layers(teacher.model)
    teacher_dropout = _model_dropout(teacher.model)

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
    # Use same weight_decay as train_model_from_file for consistent regularization
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    student.train()
    update_steps = 0
    for _ in range(cfg.epochs):
        for batch in batch_indices(len(x_train), cfg.batch_size):
            xb = x_train[batch]
            yb = y_train[batch]
            # BatchNorm requires at least 2 samples in training mode.
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
        raise ValueError(
            "No valid distillation batches. Increase training rows or reduce batch size / test_size."
        )

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

    bundle = ModelBundle(
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
    )

    return bundle, metrics


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
        teacher_model_path = str(artifacts_dir / str(teacher_model_id) / "model_bundle.pt")

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

    training_mode = str(payload.get("training_mode", "mlp_dense"))
    if training_mode == "mlp":
        training_mode = "mlp_dense"

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
    if training_mode not in {
        "mlp_dense",
        "linear_glm_baseline",
        "tabresnet",
        "imbalance_aware",
        "calibrated_classifier",
        "tree_teacher_distillation",
    }:
        return 400, {
            "status": "error",
            "error": "training_mode must be 'mlp_dense', 'linear_glm_baseline', 'tabresnet', 'imbalance_aware', 'calibrated_classifier', or 'tree_teacher_distillation'.",
        }
    if training_mode in {"imbalance_aware", "calibrated_classifier", "tree_teacher_distillation"}:
        return 400, {
            "status": "error",
            "error": f"Distillation is not yet supported for training_mode '{training_mode}'. Use mlp_dense, linear_glm_baseline, or tabresnet.",
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
    return shared_distill_model_from_file("pytorch", **kwargs)


def handle_distill_request(
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
) -> tuple[int, dict[str, Any]]:
    return shared_handle_distill_request("pytorch", payload, resolve_dataset_path, artifacts_dir)


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
        "tabresnet",
        "imbalance_aware",
        "calibrated_classifier",
        "tree_teacher_distillation",
    }:
        return 400, {
            "status": "error",
            "error": "training_mode must be 'mlp_dense', 'linear_glm_baseline', 'tabresnet', 'imbalance_aware', 'calibrated_classifier', or 'tree_teacher_distillation'.",
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


def predict_rows(bundle: ModelBundle, rows: list[dict[str, Any]], device: str | None = None) -> list[Any]:
    if not rows:
        return []

    x_rows = [{k: coerce_value(v) for k, v in row.items()} for row in rows]
    x_np = bundle.vectorizer.transform(x_rows).astype(np.float32)

    # Impute missing values using training medians for consistency with training.
    # Fallback to 0 for legacy bundles without stored medians.
    if bundle.feature_medians is not None and bundle.feature_medians.shape[0] == x_np.shape[1]:
        medians = bundle.feature_medians
    else:
        medians = np.zeros(x_np.shape[1], dtype=np.float32)
    for col_idx in range(x_np.shape[1]):
        mask = ~np.isfinite(x_np[:, col_idx])
        x_np[mask, col_idx] = medians[col_idx]

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

    # Inverse transform regression predictions to original scale
    preds = outputs.cpu().numpy()
    if bundle.target_scaler is not None:
        preds = bundle.target_scaler.inverse_transform(preds)
    return preds.squeeze(1).tolist()


def save_bundle(bundle: ModelBundle, output_dir: str | Path, metrics: Metrics | None = None) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": bundle.model.state_dict(),
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
        "model_config": {
            "training_mode": _model_training_mode(bundle.model),
            "hidden_dim": _model_hidden_dim(bundle.model),
            "num_hidden_layers": _model_num_hidden_layers(bundle.model),
            "dropout": _model_dropout(bundle.model),
        },
    }

    model_path = out / "model_bundle.pt"
    torch.save(payload, model_path)

    if metrics is not None:
        (out / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    return model_path


def load_bundle(path: str | Path, map_location: str = "cpu") -> ModelBundle:
    payload = torch.load(Path(path), map_location=map_location)
    model_cfg = payload.get("model_config", {})

    model = build_model(
        input_dim=int(payload["input_dim"]),
        output_dim=int(payload["output_dim"]),
        training_mode=model_cfg.get("training_mode", "mlp_dense"),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["state_dict"])

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
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/test a PyTorch MLP on CSV/XLS/XLSX tabular data")
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
