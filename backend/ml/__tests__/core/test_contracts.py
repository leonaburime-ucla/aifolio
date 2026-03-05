import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

import pytest

from ml.core import contracts


def test_parse_string_list_field_supports_string_and_list_inputs():
    assert contracts.parse_string_list_field("a, b , ,c", "x") == ["a", "b", "c"]
    assert contracts.parse_string_list_field(["a", "  b", 3, ""], "x") == ["a", "b", "3"]


def test_parse_string_list_field_rejects_invalid_type():
    with pytest.raises(ValueError):
        contracts.parse_string_list_field({"a": 1}, "x")


def test_validate_common_distill_bounds_catches_alpha_range():
    error = contracts.validate_common_distill_bounds(
        test_size=0.2,
        epochs=10,
        batch_size=16,
        learning_rate=0.01,
        hidden_dim=128,
        num_hidden_layers=2,
        temperature=2.0,
        alpha=1.5,
    )
    assert error == "alpha must be between 0 and 1."


def test_normalize_training_mode_maps_frontend_alias():
    assert contracts.normalize_training_mode("mlp") == "mlp_dense"
    assert contracts.normalize_training_mode("tabresnet") == "tabresnet"


def test_resolve_data_path_from_payload_prefers_data_path_then_dataset_lookup():
    resolver = lambda dataset_id: Path(f"/tmp/{dataset_id}.csv")

    assert contracts.resolve_data_path_from_payload("/tmp/data.csv", "ignored", resolver) == ("/tmp/data.csv", False)
    assert contracts.resolve_data_path_from_payload(None, "dataset", resolver) == ("/tmp/dataset.csv", False)
    assert contracts.resolve_data_path_from_payload(None, "missing", lambda _dataset_id: None) == (None, True)
    assert contracts.resolve_data_path_from_payload(None, None, resolver) == (None, False)


def test_validate_common_train_bounds_catches_each_invalid_numeric_bound():
    assert (
        contracts.validate_common_train_bounds(test_size=0.0, epochs=10, batch_size=16, learning_rate=0.01)
        == "test_size must be > 0 and < 1."
    )
    assert (
        contracts.validate_common_train_bounds(test_size=0.2, epochs=0, batch_size=16, learning_rate=0.01)
        == "epochs must be between 1 and 500."
    )
    assert (
        contracts.validate_common_train_bounds(test_size=0.2, epochs=10, batch_size=0, learning_rate=0.01)
        == "batch_size must be between 1 and 200."
    )
    assert (
        contracts.validate_common_train_bounds(test_size=0.2, epochs=10, batch_size=16, learning_rate=0.0)
        == "learning_rate must be > 0 and <= 1."
    )


def test_validate_common_distill_bounds_catches_hidden_layers_and_temperature_ranges():
    assert (
        contracts.validate_common_distill_bounds(
            test_size=0.2,
            epochs=10,
            batch_size=16,
            learning_rate=0.01,
            hidden_dim=4,
            num_hidden_layers=2,
            temperature=2.0,
            alpha=0.5,
        )
        == "student_hidden_dim must be between 8 and 500."
    )
    assert (
        contracts.validate_common_distill_bounds(
            test_size=0.2,
            epochs=10,
            batch_size=16,
            learning_rate=0.01,
            hidden_dim=32,
            num_hidden_layers=0,
            temperature=2.0,
            alpha=0.5,
        )
        == "student_num_hidden_layers must be between 1 and 15."
    )
    assert (
        contracts.validate_common_distill_bounds(
            test_size=0.2,
            epochs=10,
            batch_size=16,
            learning_rate=0.01,
            hidden_dim=32,
            num_hidden_layers=2,
            temperature=0.0,
            alpha=0.5,
        )
        == "temperature must be > 0 and <= 20."
    )
