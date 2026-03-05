import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

AI_ROOT = Path(__file__).resolve().parents[4]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.types import ModelBundle, TrainingConfig
from ml.frameworks.pytorch import trainer


class _FakeModel:
    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        import torch

        return torch.tensor([[0.1, 0.9] for _ in range(x.shape[0])], dtype=torch.float32)


class _FakeRegressionModel:
    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        import torch

        return torch.tensor([[-1.0], [1.0]][: x.shape[0]], dtype=torch.float32)


def _write_csv(path: Path, rows: list[tuple[float, object]], include_dropme: bool = False) -> None:
    if include_dropme:
        lines = ["feature,dropme,target"]
        lines.extend(f"{feature},x,{target}" for feature, target in rows)
    else:
        lines = ["feature,target"]
        lines.extend(f"{feature},{target}" for feature, target in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_train_model_from_file_rejects_excluding_target(monkeypatch):
    cfg = TrainingConfig(target_column="y")
    monkeypatch.setattr(trainer, "load_tabular_file", lambda *_args, **_kwargs: [{"a": 1, "y": 0}])
    with pytest.raises(ValueError):
        trainer.train_model_from_file("dummy.csv", cfg, exclude_columns=["y"])


def test_predict_rows_returns_decoded_labels_when_encoder_present():
    vectorizer = DictVectorizer(sparse=False)
    vectorizer.fit([{"a": 1.0}])
    scaler = StandardScaler().fit(np.array([[1.0]], dtype=np.float32))

    class _Encoder:
        def inverse_transform(self, values):
            return np.array(["yes" if int(v) else "no" for v in values])

    bundle = ModelBundle(
        model=_FakeModel(),
        task="classification",
        vectorizer=vectorizer,
        scaler=scaler,
        feature_medians=np.array([0.0], dtype=np.float32),
        label_encoder=_Encoder(),
        target_scaler=None,
        target_column="y",
        input_dim=1,
        output_dim=2,
        class_names=["no", "yes"],
    )
    preds = trainer.predict_rows(bundle, [{"a": 1.0}])
    assert preds == ["yes"]


def test_train_model_from_file_rejects_classification_only_mode_for_regression(monkeypatch):
    cfg = TrainingConfig(target_column="y", training_mode="imbalance_aware", task="regression")
    monkeypatch.setattr(
        trainer,
        "load_tabular_file",
        lambda *_args, **_kwargs: [{"a": 1.0, "y": 1.25}, {"a": 2.0, "y": 2.5}],
    )

    with pytest.raises(ValueError, match="classification-only"):
        trainer.train_model_from_file("dummy.csv", cfg)


def test_build_tree_teacher_targets_classification_uses_classifier_cls():
    import torch

    called = {}

    class _Classifier:
        def __init__(self, **kwargs):
            called["kwargs"] = kwargs

        def fit(self, x, y):
            called["fit"] = (x.shape, y.tolist())

        def predict_proba(self, x):
            return np.array([[0.2, 0.8] for _ in range(x.shape[0])], dtype=np.float32)

    probs, preds = trainer._build_tree_teacher_targets(
        x_train=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        y_train=torch.tensor([0, 1], dtype=torch.long),
        task="classification",
        random_seed=7,
        torch_device=torch.device("cpu"),
        classifier_cls=_Classifier,
    )

    assert preds is None
    assert probs is not None
    assert tuple(probs.shape) == (2, 2)
    assert called["kwargs"]["random_state"] == 7


def test_build_tree_teacher_targets_regression_uses_regressor_cls():
    import torch

    called = {}

    class _Regressor:
        def __init__(self, **kwargs):
            called["kwargs"] = kwargs

        def fit(self, x, y):
            called["fit"] = (x.shape, y.tolist())

        def predict(self, x):
            return np.array([1.5 for _ in range(x.shape[0])], dtype=np.float32)

    probs, preds = trainer._build_tree_teacher_targets(
        x_train=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        y_train=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        task="regression",
        random_seed=9,
        torch_device=torch.device("cpu"),
        regressor_cls=_Regressor,
    )

    assert probs is None
    assert preds is not None
    assert tuple(preds.shape) == (2, 1)
    assert called["kwargs"]["random_state"] == 9


def test_regression_predictions_to_list_without_scaler_returns_raw_values():
    assert trainer._regression_predictions_to_list(np.array([[1.0], [2.0]], dtype=np.float32), None) == [1.0, 2.0]


def test_predict_rows_regression_uses_target_scaler():
    vectorizer = DictVectorizer(sparse=False)
    vectorizer.fit([{"a": 1.0}])
    scaler = StandardScaler().fit(np.array([[1.0]], dtype=np.float32))
    target_scaler = StandardScaler().fit(np.array([[10.0], [20.0]], dtype=np.float32))

    bundle = ModelBundle(
        model=_FakeRegressionModel(),
        task="regression",
        vectorizer=vectorizer,
        scaler=scaler,
        feature_medians=np.array([0.0], dtype=np.float32),
        label_encoder=None,
        target_scaler=target_scaler,
        target_column="y",
        input_dim=1,
        output_dim=1,
        class_names=None,
    )

    preds = trainer.predict_rows(bundle, [{"a": 1.0}, {"a": 1.0}])
    assert preds == [10.0, 20.0]


def test_train_model_from_file_classification_end_to_end(tmp_path):
    path = tmp_path / "classification.csv"
    _write_csv(path, [(0.0, "no"), (1.0, "yes"), (0.1, "no"), (1.1, "yes"), (0.2, "no"), (1.2, "yes")], include_dropme=True)
    cfg = TrainingConfig(
        target_column="target",
        task="classification",
        test_size=0.5,
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        num_hidden_layers=1,
        dropout=0.0,
    )

    bundle, metrics = trainer.train_model_from_file(path, cfg, exclude_columns=["dropme"])
    assert bundle.task == "classification"
    assert metrics.test_metric_name == "accuracy"


def test_train_model_from_file_regression_end_to_end(tmp_path):
    path = tmp_path / "regression.csv"
    _write_csv(path, [(0.0, 10.0), (1.0, 20.0), (0.1, 12.0), (1.1, 22.0), (0.2, 14.0), (1.2, 24.0)])
    cfg = TrainingConfig(
        target_column="target",
        task="regression",
        test_size=0.5,
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        num_hidden_layers=1,
        dropout=0.0,
    )

    bundle, metrics = trainer.train_model_from_file(path, cfg)
    assert bundle.task == "regression"
    assert metrics.test_metric_name == "rmse"
    assert metrics.test_metric_value >= 0.0


def test_train_model_from_file_tree_teacher_distillation_end_to_end(tmp_path):
    path = tmp_path / "classification.csv"
    _write_csv(path, [(0.0, "no"), (1.0, "yes"), (0.1, "no"), (1.1, "yes"), (0.2, "no"), (1.2, "yes")])
    cfg = TrainingConfig(
        target_column="target",
        task="classification",
        training_mode="tree_teacher_distillation",
        test_size=0.5,
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        num_hidden_layers=1,
        dropout=0.0,
    )

    bundle, metrics = trainer.train_model_from_file(path, cfg)
    assert bundle.task == "classification"
    assert metrics.test_metric_name == "accuracy"


def test_train_model_from_file_raises_when_no_valid_batches(tmp_path):
    path = tmp_path / "regression.csv"
    _write_csv(path, [(0.0, 10.0), (1.0, 20.0)])
    cfg = TrainingConfig(target_column="target", task="regression", test_size=0.5, epochs=1, batch_size=4)

    with pytest.raises(ValueError, match="No valid training batches"):
        trainer.train_model_from_file(path, cfg)


def test_predict_rows_classification_returns_indices_without_label_encoder():
    vectorizer = DictVectorizer(sparse=False)
    vectorizer.fit([{"a": 1.0}])
    scaler = StandardScaler().fit(np.array([[1.0]], dtype=np.float32))

    bundle = ModelBundle(
        model=_FakeModel(),
        task="classification",
        vectorizer=vectorizer,
        scaler=scaler,
        feature_medians=None,
        label_encoder=None,
        target_scaler=None,
        target_column="y",
        input_dim=1,
        output_dim=2,
        class_names=None,
    )
    assert trainer.predict_rows(bundle, [{"a": 1.0}]) == [1]


def test_predict_rows_returns_empty_for_no_rows():
    vectorizer = DictVectorizer(sparse=False)
    vectorizer.fit([{"a": 1.0}])
    scaler = StandardScaler().fit(np.array([[1.0]], dtype=np.float32))

    bundle = ModelBundle(
        model=_FakeModel(),
        task="classification",
        vectorizer=vectorizer,
        scaler=scaler,
        feature_medians=np.array([0.0], dtype=np.float32),
        label_encoder=None,
        target_scaler=None,
        target_column="y",
        input_dim=1,
        output_dim=2,
        class_names=None,
    )
    assert trainer.predict_rows(bundle, []) == []


def test_load_bundle_delegates_to_runtime_loader(monkeypatch):
    calls = {}

    def _fake_load_bundle(*, path, map_location):
        calls["args"] = {"path": path, "map_location": map_location}
        return "bundle"

    monkeypatch.setattr(trainer, "_load_bundle", _fake_load_bundle)

    assert trainer.load_bundle("bundle.pt", map_location="cuda:0") == "bundle"
    assert calls["args"] == {"path": "bundle.pt", "map_location": "cuda:0"}
