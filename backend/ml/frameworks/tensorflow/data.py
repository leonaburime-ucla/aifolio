from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from ...core.preprocessing import impute_train_test_non_finite
    from ...ml_util import TaskType
except ImportError:  # pragma: no cover
    from core.preprocessing import impute_train_test_non_finite  # type: ignore
    from ml_util import TaskType  # type: ignore

from ...core.types import TrainingConfig


def prepare_arrays(
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

    x_train_np, x_test_np, col_medians = impute_train_test_non_finite(x_train_np, x_test_np)

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
