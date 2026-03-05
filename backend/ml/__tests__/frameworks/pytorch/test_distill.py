import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import torch

AI_ROOT = Path(__file__).resolve().parents[4]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.types import ModelBundle, TrainingConfig
from ml.frameworks.pytorch import distill
from ml.frameworks.pytorch.distill import distill_model_from_file
from ml.frameworks.pytorch.models import build_model


def _write_csv(path: Path, rows: list[tuple[float, object]], include_dropme: bool = False) -> None:
    if include_dropme:
        lines = ["feature,dropme,target"]
        lines.extend(f"{feature},x,{target}" for feature, target in rows)
    else:
        lines = ["feature,target"]
        lines.extend(f"{feature},{target}" for feature, target in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_torch_teacher_bundle(task: str, use_target_scaler: bool = True) -> ModelBundle:
    features = [{"feature": 0.0}, {"feature": 1.0}, {"feature": 0.2}, {"feature": 1.2}]
    vectorizer = DictVectorizer(sparse=False)
    x_np = vectorizer.fit_transform(features).astype(np.float32)
    scaler = StandardScaler().fit(x_np)

    if task == "classification":
        encoder = LabelEncoder().fit(["no", "yes"])
        model = build_model(1, 2, "mlp_dense", hidden_dim=8, num_hidden_layers=1, dropout=0.0)
        return ModelBundle(
            model=model,
            task="classification",
            vectorizer=vectorizer,
            scaler=scaler,
            feature_medians=np.zeros(x_np.shape[1], dtype=np.float32),
            label_encoder=encoder,
            target_scaler=None,
            target_column="target",
            input_dim=1,
            output_dim=2,
            class_names=["no", "yes"],
        )

    model = build_model(1, 1, "mlp_dense", hidden_dim=8, num_hidden_layers=1, dropout=0.0)
    target_scaler = (
        StandardScaler().fit(np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32))
        if use_target_scaler
        else None
    )
    return ModelBundle(
        model=model,
        task="regression",
        vectorizer=vectorizer,
        scaler=scaler,
        feature_medians=np.zeros(x_np.shape[1], dtype=np.float32),
        label_encoder=None,
        target_scaler=target_scaler,
        target_column="target",
        input_dim=1,
        output_dim=1,
        class_names=None,
    )


def test_distill_rejects_invalid_temperature_before_runtime_work():
    cfg = TrainingConfig(target_column="y")
    with pytest.raises(ValueError, match="temperature must be > 0"):
        distill_model_from_file("dummy.csv", cfg, teacher_bundle=object(), temperature=0)


def test_distill_requires_teacher_path_or_bundle():
    cfg = TrainingConfig(target_column="y")
    with pytest.raises(ValueError, match="teacher_path or teacher_bundle is required"):
        distill_model_from_file("dummy.csv", cfg)


def test_distill_rejects_invalid_alpha_before_runtime_work():
    cfg = TrainingConfig(target_column="y")
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        distill_model_from_file("dummy.csv", cfg, teacher_bundle=object(), alpha=2.0)


def test_distill_rejects_task_mismatch_before_data_loading():
    cfg = TrainingConfig(target_column="y", task="regression")
    teacher = SimpleNamespace(task="classification")

    with pytest.raises(ValueError, match="Requested task does not match teacher task"):
        distill_model_from_file("dummy.csv", cfg, teacher_bundle=teacher)


def test_classification_soft_loss_returns_positive_scalar():
    student_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    teacher_out = torch.tensor([[1.5, 1.0]], dtype=torch.float32)
    loss = distill._classification_soft_loss(student_out, teacher_out, temperature=2.0)
    assert float(loss.item()) > 0.0


def test_regression_rmse_uses_target_scaler_when_present():
    scaler = StandardScaler().fit(np.array([[10.0], [20.0]], dtype=np.float32))
    preds = scaler.transform(np.array([[10.0], [20.0]], dtype=np.float32))
    truth = scaler.transform(np.array([[10.0], [20.0]], dtype=np.float32))

    assert distill._regression_rmse(preds, truth, scaler) == 0.0


def test_regression_rmse_without_target_scaler_uses_raw_values():
    preds = np.array([[1.0], [2.0]], dtype=np.float32)
    truth = np.array([[1.0], [2.0]], dtype=np.float32)
    assert distill._regression_rmse(preds, truth, None) == 0.0


def test_distill_classification_end_to_end_with_teacher_bundle(tmp_path):
    path = tmp_path / "classification.csv"
    _write_csv(path, [(0.0, "no"), (1.0, "yes"), (0.1, "no"), (1.1, "yes"), (0.2, "no"), (1.2, "yes")])
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

    bundle, metrics = distill_model_from_file(path, cfg, teacher_bundle=_build_torch_teacher_bundle("classification"))
    assert bundle.task == "classification"
    assert metrics.test_metric_name == "accuracy"


def test_distill_regression_end_to_end_without_target_scaler(tmp_path):
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

    bundle, metrics = distill_model_from_file(path, cfg, teacher_bundle=_build_torch_teacher_bundle("regression", use_target_scaler=False))
    assert bundle.task == "regression"
    assert metrics.test_metric_name == "rmse"
    assert metrics.test_metric_value >= 0.0


def test_distill_regression_end_to_end_with_target_scaler_and_excluded_columns(tmp_path):
    path = tmp_path / "regression.csv"
    _write_csv(
        path,
        [(0.0, 10.0), (1.0, 20.0), (0.1, 12.0), (1.1, 22.0), (0.2, 14.0), (1.2, 24.0)],
        include_dropme=True,
    )
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

    bundle, metrics = distill_model_from_file(
        path,
        cfg,
        teacher_bundle=_build_torch_teacher_bundle("regression", use_target_scaler=True),
        exclude_columns=["dropme"],
    )
    assert bundle.task == "regression"
    assert metrics.test_metric_name == "rmse"
    assert metrics.test_metric_value >= 0.0


def test_distill_loads_teacher_bundle_from_path(monkeypatch, tmp_path):
    path = tmp_path / "classification.csv"
    _write_csv(path, [(0.0, "no"), (1.0, "yes"), (0.1, "no"), (1.1, "yes")])
    cfg = TrainingConfig(target_column="target", task="classification", test_size=0.5, epochs=1, batch_size=2)
    monkeypatch.setattr(distill, "load_bundle", lambda teacher_path: _build_torch_teacher_bundle("classification"))

    bundle, metrics = distill_model_from_file(path, cfg, teacher_path="teacher.pt")
    assert bundle.task == "classification"
    assert metrics.test_metric_name == "accuracy"


def test_distill_classification_requires_teacher_label_encoder(tmp_path):
    path = tmp_path / "classification.csv"
    _write_csv(path, [(0.0, "no"), (1.0, "yes"), (0.1, "no"), (1.1, "yes")])
    teacher = _build_torch_teacher_bundle("classification")
    teacher.label_encoder = None
    cfg = TrainingConfig(target_column="target", task="classification", test_size=0.5, epochs=1, batch_size=2)

    with pytest.raises(ValueError, match="label encoder"):
        distill_model_from_file(path, cfg, teacher_bundle=teacher)


def test_distill_rejects_excluding_target_column(tmp_path):
    path = tmp_path / "classification.csv"
    _write_csv(path, [(0.0, "no"), (1.0, "yes"), (0.1, "no"), (1.1, "yes")])
    cfg = TrainingConfig(target_column="target", task="classification", test_size=0.5, epochs=1, batch_size=2)

    with pytest.raises(ValueError, match="target_column cannot be excluded"):
        distill_model_from_file(
            path,
            cfg,
            teacher_bundle=_build_torch_teacher_bundle("classification"),
            exclude_columns=["target"],
        )


def test_distill_raises_when_no_valid_batches(tmp_path):
    path = tmp_path / "regression.csv"
    _write_csv(path, [(0.0, 10.0), (1.0, 20.0)])
    cfg = TrainingConfig(target_column="target", task="regression", test_size=0.5, epochs=1, batch_size=4)

    with pytest.raises(ValueError, match="No valid distillation batches"):
        distill_model_from_file(path, cfg, teacher_bundle=_build_torch_teacher_bundle("regression", use_target_scaler=False))
