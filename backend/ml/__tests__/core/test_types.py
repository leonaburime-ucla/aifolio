import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.types import Metrics, ModelBundle, TrainingConfig


def test_training_config_defaults_are_stable():
    cfg = TrainingConfig(target_column="target")

    assert cfg.training_mode == "mlp_dense"
    assert cfg.task == "auto"
    assert cfg.epochs == 500
    assert cfg.batch_size == 64


def test_metrics_dataclass_preserves_named_fields():
    metrics = Metrics(
        task="classification",
        train_loss=0.1,
        test_loss=0.2,
        test_metric_name="accuracy",
        test_metric_value=0.9,
    )

    assert metrics.test_metric_name == "accuracy"
    assert metrics.test_metric_value == 0.9


def test_model_bundle_allows_optional_model_config():
    vectorizer = DictVectorizer(sparse=False)
    vectorizer.fit([{"a": 1.0}])
    scaler = StandardScaler().fit(np.array([[1.0]], dtype=np.float32))

    bundle = ModelBundle(
        model=object(),
        task="regression",
        vectorizer=vectorizer,
        scaler=scaler,
        feature_medians=np.array([0.0], dtype=np.float32),
        label_encoder=None,
        target_scaler=None,
        target_column="target",
        input_dim=1,
        output_dim=1,
        class_names=None,
    )

    assert bundle.model_config is None
    assert bundle.input_dim == 1
