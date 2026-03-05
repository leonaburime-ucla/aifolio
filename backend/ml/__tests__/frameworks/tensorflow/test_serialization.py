import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

AI_ROOT = Path(__file__).resolve().parents[4]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.types import ModelBundle
from ml.frameworks.tensorflow.serialization import load_bundle, save_bundle


class _FakeModel:
    def save(self, path):
        Path(path).write_text("fake-model", encoding="utf-8")


def test_save_and_load_bundle_roundtrip_with_mocked_loader(tmp_path, monkeypatch):
    vectorizer = DictVectorizer()
    vectorizer.fit([{"a": 1.0}])
    scaler = StandardScaler().fit(np.array([[1.0]], dtype=np.float32))
    bundle = ModelBundle(
        model=_FakeModel(),
        task="regression",
        vectorizer=vectorizer,
        scaler=scaler,
        feature_medians=np.array([0.0], dtype=np.float32),
        label_encoder=None,
        target_scaler=None,
        target_column="y",
        input_dim=1,
        output_dim=1,
        class_names=None,
        model_config={"hidden_dim": 8},
    )

    model_path = save_bundle(bundle, tmp_path)
    monkeypatch.setattr("ml.frameworks.tensorflow.serialization.tf.keras.models.load_model", lambda _p: _FakeModel())
    restored = load_bundle(model_path)
    assert restored.task == "regression"
    assert restored.model_config == {"hidden_dim": 8}
