import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

AI_ROOT = Path(__file__).resolve().parents[4]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.types import ModelBundle
from ml.frameworks.pytorch.models import build_model
from ml.frameworks.pytorch.serialization import load_bundle, save_bundle


def test_save_and_load_bundle_roundtrip(tmp_path):
    model = build_model(2, 2, "mlp_dense", 8, 1, 0.0)
    vectorizer = DictVectorizer()
    vectorizer.fit([{"a": 1.0, "b": 2.0}])
    scaler = StandardScaler().fit(np.array([[1.0, 2.0]], dtype=np.float32))
    bundle = ModelBundle(
        model=model,
        task="classification",
        vectorizer=vectorizer,
        scaler=scaler,
        feature_medians=np.array([0.0, 0.0], dtype=np.float32),
        label_encoder=None,
        target_scaler=None,
        target_column="y",
        input_dim=2,
        output_dim=2,
        class_names=["a", "b"],
    )

    model_path = save_bundle(bundle, tmp_path)
    restored = load_bundle(model_path)
    assert restored.task == "classification"
    assert restored.input_dim == 2
    assert restored.output_dim == 2
