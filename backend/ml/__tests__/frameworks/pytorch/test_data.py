import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[4]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.types import TrainingConfig
from ml.frameworks.pytorch.data import prepare_tensors


def test_prepare_tensors_classification_returns_expected_shapes():
    x_rows = [{"a": float(i), "b": float(i % 2)} for i in range(12)]
    y = ["yes" if i % 2 else "no" for i in range(12)]
    cfg = TrainingConfig(target_column="y", test_size=0.25, random_seed=7)

    x_train, x_test, y_train, y_test, _vec, _scaler, _medians, encoder, _target_scaler, input_dim, output_dim, names = prepare_tensors(
        x_rows, y, "classification", cfg
    )
    assert x_train.shape[1] == input_dim
    assert output_dim == 2
    assert encoder is not None
    assert sorted(names or []) == ["no", "yes"]
    assert y_train.ndim == 1
    assert y_test.ndim == 1


def test_prepare_tensors_regression_outputs_column_target():
    x_rows = [{"a": float(i), "b": float(i * 2)} for i in range(12)]
    y = [float(i) for i in range(12)]
    cfg = TrainingConfig(target_column="y", test_size=0.25, random_seed=7)

    _x_train, _x_test, y_train, y_test, _vec, _scaler, _medians, encoder, target_scaler, _in_dim, output_dim, names = prepare_tensors(
        x_rows, y, "regression", cfg
    )
    assert output_dim == 1
    assert encoder is None
    assert target_scaler is not None
    assert names is None
    assert y_train.ndim == 2
    assert y_test.ndim == 2
