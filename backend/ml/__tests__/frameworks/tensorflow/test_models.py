import sys
from pathlib import Path

import numpy as np
import pytest

AI_ROOT = Path(__file__).resolve().parents[4]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.frameworks.tensorflow import models


def test_set_seed_calls_tensorflow_seed(monkeypatch):
    called = {"seed": None}

    class _Random:
        @staticmethod
        def set_seed(seed):
            called["seed"] = seed

    monkeypatch.setattr(models.tf, "random", _Random)
    models.set_seed(7)
    assert called["seed"] == 7


def test_build_model_linear_classification_predicts_shape():
    model = models.build_model(3, 2, "classification", "linear_glm_baseline", 8, 1, 0.0, 0.01)
    preds = model.predict(np.zeros((2, 3), dtype=np.float32), verbose=0)
    assert preds.shape == (2, 2)


def test_build_model_multi_task_classification_returns_two_outputs():
    model = models.build_model(3, 2, "classification", "multi_task_learning", 8, 1, 0.0, 0.01)
    preds = model.predict(np.zeros((2, 3), dtype=np.float32), verbose=0)
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert preds[0].shape == (2, 2)
    assert preds[1].shape == (2, 2)


def test_model_config_accessors_use_bundle_model_config():
    class _Bundle:
        model_config = {"hidden_dim": 99, "num_hidden_layers": 4, "dropout": 0.25}

    assert models.model_hidden_dim(_Bundle()) == 99
    assert models.model_num_hidden_layers(_Bundle()) == 4
    assert models.model_dropout(_Bundle()) == 0.25


def test_quantile_pinball_loss_returns_expected_value():
    y_true = models.tf.constant([[2.0], [4.0]], dtype=models.tf.float32)
    y_pred = models.tf.constant([[1.0], [5.0]], dtype=models.tf.float32)
    loss = models.quantile_pinball_loss(0.8, y_true, y_pred)
    assert float(loss.numpy()) == pytest.approx(0.5)


def test_classification_and_regression_loss_helpers_choose_expected_losses():
    classification_loss = models._classification_loss("calibrated_classifier")
    assert isinstance(classification_loss, models.tf.keras.losses.Loss)
    assert models._classification_loss("mlp") == "sparse_categorical_crossentropy"

    regression_loss = models._regression_loss("quantile_regression", 0.8)
    y_true = models.tf.constant([[2.0]], dtype=models.tf.float32)
    y_pred = models.tf.constant([[1.0]], dtype=models.tf.float32)
    assert float(regression_loss(y_true, y_pred).numpy()) == pytest.approx(0.8)
    assert models._regression_loss("mlp", 0.8) == "mse"


@pytest.mark.parametrize(
    "training_mode",
    ["entity_embeddings", "wide_and_deep", "time_aware_tabular"],
)
def test_build_model_special_classification_modes_predict_shape(training_mode):
    model = models.build_model(4, 3, "classification", training_mode, 8, 2, 0.1, 0.01)
    preds = model.predict(np.zeros((2, 4), dtype=np.float32), verbose=0)
    assert preds.shape == (2, 3)


def test_build_model_autoencoder_head_classification_returns_prediction_and_reconstruction():
    model = models.build_model(4, 3, "classification", "autoencoder_head", 8, 1, 0.0, 0.01)
    preds = model.predict(np.zeros((2, 4), dtype=np.float32), verbose=0)
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert preds[0].shape == (2, 3)
    assert preds[1].shape == (2, 4)


def test_build_model_default_regression_predicts_single_output():
    model = models.build_model(3, 1, "regression", "mlp", 8, 2, 0.1, 0.01)
    preds = model.predict(np.zeros((2, 3), dtype=np.float32), verbose=0)
    assert preds.shape == (2, 1)


def test_build_model_autoencoder_head_regression_returns_prediction_and_reconstruction():
    model = models.build_model(4, 1, "regression", "autoencoder_head", 8, 1, 0.0, 0.01)
    preds = model.predict(np.zeros((2, 4), dtype=np.float32), verbose=0)
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert preds[0].shape == (2, 1)
    assert preds[1].shape == (2, 4)


def test_build_model_multi_task_regression_returns_two_outputs():
    model = models.build_model(3, 1, "regression", "multi_task_learning", 8, 1, 0.0, 0.01)
    preds = model.predict(np.zeros((2, 3), dtype=np.float32), verbose=0)
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert preds[0].shape == (2, 1)
    assert preds[1].shape == (2, 1)


def test_build_model_calibrated_classifier_uses_smoothed_cross_entropy():
    model = models.build_model(3, 2, "classification", "calibrated_classifier", 8, 1, 0.0, 0.01)
    assert isinstance(model.loss, models.tf.keras.losses.SparseCategoricalCrossentropy)


def test_build_model_quantile_regression_uses_pinball_loss():
    model = models.build_model(3, 1, "regression", "quantile_regression", 8, 1, 0.0, 0.01)
    y_true = models.tf.constant([[2.0], [4.0]], dtype=models.tf.float32)
    y_pred = models.tf.constant([[1.0], [5.0]], dtype=models.tf.float32)
    assert float(model.loss(y_true, y_pred).numpy()) == pytest.approx(0.5)


def test_model_config_accessors_use_defaults_when_missing():
    class _Bundle:
        model_config = None

    assert models.model_hidden_dim(_Bundle()) == 128
    assert models.model_num_hidden_layers(_Bundle()) == 2
    assert models.model_dropout(_Bundle()) == 0.1
