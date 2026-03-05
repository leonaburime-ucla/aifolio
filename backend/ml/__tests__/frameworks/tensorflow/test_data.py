import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[4]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.types import TrainingConfig
from ml.frameworks.tensorflow.data import prepare_arrays


def test_prepare_arrays_classification_outputs_label_encoder():
    x_rows = [{"a": float(i), "b": float(i % 3)} for i in range(15)]
    y = ["yes" if i % 2 else "no" for i in range(15)]
    cfg = TrainingConfig(target_column="y", test_size=0.2, random_seed=3)

    _x_train, _x_test, y_train, y_test, _vec, _scaler, _medians, encoder, target_scaler, input_dim, output_dim, class_names = prepare_arrays(
        x_rows, y, "classification", cfg
    )
    assert input_dim > 0
    assert output_dim == 2
    assert encoder is not None
    assert target_scaler is None
    assert sorted(class_names or []) == ["no", "yes"]
    assert y_train.ndim == 1
    assert y_test.ndim == 1


def test_prepare_arrays_regression_outputs_scaled_targets():
    x_rows = [{"a": float(i), "b": float(i * 2)} for i in range(15)]
    y = [float(i) for i in range(15)]
    cfg = TrainingConfig(target_column="y", test_size=0.2, random_seed=3)

    _x_train, _x_test, y_train, y_test, _vec, _scaler, _medians, encoder, target_scaler, _input_dim, output_dim, class_names = prepare_arrays(
        x_rows, y, "regression", cfg
    )
    assert output_dim == 1
    assert encoder is None
    assert target_scaler is not None
    assert class_names is None
    assert y_train.ndim == 2
    assert y_test.ndim == 2
