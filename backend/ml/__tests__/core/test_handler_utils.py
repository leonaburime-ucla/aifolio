import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core import handler_utils


def test_parse_train_numeric_returns_error_on_invalid_number():
    parsed, error = handler_utils.parse_train_numeric({"epochs": "oops"})
    assert parsed is None
    assert error == "Invalid numeric training parameters."


def test_parse_train_numeric_and_distill_numeric_return_defaults_and_optional_fields():
    parsed_train, train_error = handler_utils.parse_train_numeric({})
    assert train_error is None
    assert parsed_train == {
        "test_size": 0.2,
        "epochs": 500,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "training_mode": "mlp_dense",
        "hidden_dim": 128,
        "num_hidden_layers": 2,
        "dropout": 0.1,
        "random_seed": 42,
    }

    parsed_distill, distill_error = handler_utils.parse_distill_numeric({"student_hidden_dim": 32, "student_dropout": 0.25})
    assert distill_error is None
    assert parsed_distill["hidden_dim"] == 32
    assert parsed_distill["num_hidden_layers"] is None
    assert parsed_distill["student_dropout"] == 0.25


def test_parse_distill_numeric_returns_error_on_invalid_number():
    parsed, error = handler_utils.parse_distill_numeric({"temperature": "oops"})
    assert parsed is None
    assert error == "Invalid numeric distillation parameters."


def test_hidden_config_error_allows_linear_baseline_only():
    assert handler_utils.hidden_config_error("linear_glm_baseline", 1, 0, 5.0) is None


def test_hidden_config_error_validates_hidden_dim_layers_and_dropout():
    assert handler_utils.hidden_config_error("mlp_dense", 1, 2, 0.1) == "hidden_dim must be between 8 and 500."
    assert handler_utils.hidden_config_error("mlp_dense", 32, 0, 0.1) == "num_hidden_layers must be between 1 and 15."
    assert handler_utils.hidden_config_error("mlp_dense", 32, 2, 1.0) == "dropout must be between 0 and 0.9."


def test_runtime_unavailable_response_uses_framework_specific_package_hint():
    status, payload = handler_utils.runtime_unavailable_response("PyTorch", "missing torch")
    assert status == 503
    assert payload["hint"].endswith("install torch in the interpreter running the server.")

    status, payload = handler_utils.runtime_unavailable_response("TensorFlow", "missing tensorflow")
    assert status == 503
    assert payload["hint"].endswith("install tensorflow in the interpreter running the server.")


def test_compute_distill_stats_uses_fallback_serialized_sizes():
    stats = handler_utils.compute_distill_stats(
        teacher_model_for_stats="teacher",
        student_model="student",
        teacher_model_size_bytes=None,
        student_model_size_bytes=None,
        parameter_count_fn=lambda model: 100 if model == "teacher" else 25,
        serialized_size_fn=lambda model: 1000 if model == "teacher" else 400,
    )
    assert stats["size_saved_bytes"] == 600
    assert stats["param_saved_count"] == 75
    assert stats["param_saved_percent"] == 75.0


def test_compute_distill_stats_handles_missing_sizes_and_zero_teacher_params():
    stats = handler_utils.compute_distill_stats(
        teacher_model_for_stats="teacher",
        student_model="student",
        teacher_model_size_bytes=None,
        student_model_size_bytes=None,
        parameter_count_fn=lambda model: 0 if model == "teacher" else 10,
        serialized_size_fn=lambda model: None,
    )
    assert stats["size_saved_bytes"] is None
    assert stats["size_saved_percent"] is None
    assert stats["param_saved_percent"] is None
