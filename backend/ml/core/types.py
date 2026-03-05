from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from ..ml_util import TaskType
except ImportError:  # pragma: no cover
    from ml_util import TaskType  # type: ignore


TrainingTask = TaskType | Literal["auto"]


@dataclass
class TrainingConfig:
    target_column: str
    training_mode: str = "mlp_dense"
    task: TrainingTask = "auto"
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
    model: Any
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
    model_config: dict[str, Any] | None = None
