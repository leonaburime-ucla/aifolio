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
from ml.frameworks.tensorflow import trainer


class _FakeModel:
    def predict(self, x, verbose=0):
        _ = verbose
        return np.array([[0.1, 0.9] for _ in range(len(x))], dtype=np.float32)


class _FakeRegressionModel:
    def predict(self, x, verbose=0):
        _ = verbose
        return np.array([[-1.0], [1.0]][: len(x)], dtype=np.float32)


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


def test_predict_rows_decodes_indices_with_label_encoder():
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


def test_train_model_from_file_rejects_quantile_regression_for_classification(monkeypatch):
    cfg = TrainingConfig(target_column="y", training_mode="quantile_regression")
    monkeypatch.setattr(
        trainer,
        "load_tabular_file",
        lambda *_args, **_kwargs: [{"a": 1.0, "y": 0}, {"a": 2.0, "y": 1}],
    )

    with pytest.raises(ValueError, match="regression-only"):
        trainer.train_model_from_file("dummy.csv", cfg)


def test_train_model_from_file_rejects_classification_only_mode_for_regression(monkeypatch):
    cfg = TrainingConfig(target_column="y", task="regression", training_mode="imbalance_aware")
    monkeypatch.setattr(
        trainer,
        "load_tabular_file",
        lambda *_args, **_kwargs: [{"a": 1.0, "y": 1.5}, {"a": 2.0, "y": 2.5}],
    )

    with pytest.raises(ValueError, match="classification-only"):
        trainer.train_model_from_file("dummy.csv", cfg)


def test_build_fit_targets_for_multi_task_learning():
    y_train_fit, y_test_fit = trainer._build_fit_targets(
        "multi_task_learning",
        x_train=np.array([[1.0], [2.0]], dtype=np.float32),
        x_test=np.array([[3.0]], dtype=np.float32),
        y_train=np.array([0, 1], dtype=np.int32),
        y_test=np.array([1], dtype=np.int32),
    )
    assert set(y_train_fit.keys()) == {"main_output", "aux_output"}
    assert set(y_test_fit.keys()) == {"main_output", "aux_output"}
    assert y_train_fit["main_output"].tolist() == [0, 1]


def test_build_fit_targets_for_autoencoder_head():
    x_train = np.array([[1.0], [2.0]], dtype=np.float32)
    x_test = np.array([[3.0]], dtype=np.float32)
    y_train_fit, y_test_fit = trainer._build_fit_targets(
        "autoencoder_head",
        x_train=x_train,
        x_test=x_test,
        y_train=np.array([0, 1], dtype=np.int32),
        y_test=np.array([1], dtype=np.int32),
    )
    assert set(y_train_fit.keys()) == {"main_output", "reconstruction_output"}
    assert y_train_fit["reconstruction_output"].tolist() == x_train.tolist()
    assert y_test_fit["reconstruction_output"].tolist() == x_test.tolist()


def test_build_fit_targets_returns_plain_targets_for_standard_mode():
    y_train = np.array([0, 1], dtype=np.int32)
    y_test = np.array([1], dtype=np.int32)
    y_train_fit, y_test_fit = trainer._build_fit_targets(
        "mlp",
        x_train=np.array([[1.0], [2.0]], dtype=np.float32),
        x_test=np.array([[3.0]], dtype=np.float32),
        y_train=y_train,
        y_test=y_test,
    )
    assert y_train_fit is y_train
    assert y_test_fit is y_test


def test_build_class_weight_map_scales_inverse_frequency():
    weights = trainer._build_class_weight_map(np.array([0, 0, 1], dtype=np.int32), output_dim=2)
    assert weights[1] > weights[0]


def test_main_output_handles_dict_list_and_array():
    arr = np.array([[1.0]], dtype=np.float32)
    assert trainer._main_output({"main_output": arr}).shape == (1, 1)
    assert trainer._main_output({"other": arr}).shape == (1, 1)
    assert trainer._main_output([arr]).shape == (1, 1)
    assert trainer._main_output(arr).shape == (1, 1)


def test_regression_metric_returns_rmse_or_pinball():
    preds = np.array([[0.0], [1.0]], dtype=np.float32)
    truth = np.array([[0.0], [1.0]], dtype=np.float32)
    assert trainer._regression_metric(preds, truth, None, "mlp") == ("rmse", 0.0)

    metric_name, metric_value = trainer._regression_metric(
        np.array([[0.2], [0.8]], dtype=np.float32),
        np.array([[0.0], [1.0]], dtype=np.float32),
        None,
        "quantile_regression",
    )
    assert metric_name == "pinball_p80"
    assert metric_value >= 0.0


def test_regression_metric_uses_target_scaler_when_present():
    target_scaler = StandardScaler().fit(np.array([[10.0], [20.0]], dtype=np.float32))
    preds = target_scaler.transform(np.array([[10.0], [20.0]], dtype=np.float32))
    truth = target_scaler.transform(np.array([[10.0], [20.0]], dtype=np.float32))

    assert trainer._regression_metric(preds, truth, target_scaler, "mlp") == ("rmse", 0.0)


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


def test_predict_rows_returns_empty_for_no_rows():
    bundle = ModelBundle(
        model=_FakeModel(),
        task="classification",
        vectorizer=DictVectorizer(sparse=False).fit([{"a": 1.0}]),
        scaler=StandardScaler().fit(np.array([[1.0]], dtype=np.float32)),
        feature_medians=np.array([0.0], dtype=np.float32),
        label_encoder=None,
        target_scaler=None,
        target_column="y",
        input_dim=1,
        output_dim=2,
        class_names=None,
    )
    assert trainer.predict_rows(bundle, []) == []


def test_predict_rows_classification_returns_indices_without_encoder():
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


def test_train_model_from_file_imbalance_aware_end_to_end(tmp_path):
    path = tmp_path / "classification.csv"
    _write_csv(
        path,
        [(0.0, "no"), (0.1, "no"), (0.2, "no"), (0.3, "no"), (1.0, "yes"), (1.1, "yes")],
        include_dropme=True,
    )
    cfg = TrainingConfig(
        target_column="target",
        task="classification",
        training_mode="imbalance_aware",
        test_size=0.5,
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        num_hidden_layers=1,
        dropout=0.0,
    )

    bundle, metrics = trainer.train_model_from_file(path, cfg, exclude_columns=["dropme"])
    assert bundle.task == "classification"
    assert bundle.model_config["training_mode"] == "imbalance_aware"
    assert metrics.test_metric_name == "accuracy"


def test_train_model_from_file_multi_task_learning_end_to_end(tmp_path):
    path = tmp_path / "classification.csv"
    _write_csv(path, [(0.0, "no"), (1.0, "yes"), (0.1, "no"), (1.1, "yes"), (0.2, "no"), (1.2, "yes")])
    cfg = TrainingConfig(
        target_column="target",
        task="classification",
        training_mode="multi_task_learning",
        test_size=0.5,
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        num_hidden_layers=1,
        dropout=0.0,
    )

    bundle, metrics = trainer.train_model_from_file(path, cfg)
    assert bundle.task == "classification"
    assert bundle.model.output_names == ["main_output", "aux_output"]
    assert metrics.test_metric_name == "accuracy"


def test_train_model_from_file_autoencoder_head_end_to_end(tmp_path):
    path = tmp_path / "regression.csv"
    _write_csv(path, [(0.0, 10.0), (1.0, 20.0), (0.1, 12.0), (1.1, 22.0), (0.2, 14.0), (1.2, 24.0)])
    cfg = TrainingConfig(
        target_column="target",
        task="regression",
        training_mode="autoencoder_head",
        test_size=0.5,
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        num_hidden_layers=1,
        dropout=0.0,
    )

    bundle, metrics = trainer.train_model_from_file(path, cfg)
    assert bundle.task == "regression"
    assert bundle.model.output_names == ["main_output", "reconstruction_output"]
    assert metrics.test_metric_name == "rmse"
    assert metrics.test_metric_value >= 0.0


def test_train_model_from_file_quantile_regression_end_to_end(tmp_path):
    path = tmp_path / "regression.csv"
    _write_csv(path, [(0.0, 10.0), (1.0, 20.0), (0.1, 12.0), (1.1, 22.0), (0.2, 14.0), (1.2, 24.0)])
    cfg = TrainingConfig(
        target_column="target",
        task="regression",
        training_mode="quantile_regression",
        test_size=0.5,
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        num_hidden_layers=1,
        dropout=0.0,
    )

    bundle, metrics = trainer.train_model_from_file(path, cfg)
    assert bundle.task == "regression"
    assert metrics.test_metric_name == "pinball_p80"
    assert metrics.test_metric_value >= 0.0


def test_load_bundle_delegates_to_runtime_loader(monkeypatch):
    calls = {}

    def _fake_load_bundle(*, path, map_location):
        calls["args"] = {"path": path, "map_location": map_location}
        return "bundle"

    monkeypatch.setattr(trainer, "_load_bundle", _fake_load_bundle)

    assert trainer.load_bundle("bundle.keras", map_location="gpu") == "bundle"
    assert calls["args"] == {"path": "bundle.keras", "map_location": "gpu"}
