from __future__ import annotations

"""
Shared payload and validation helpers used by multiple ML frameworks.

These helpers intentionally avoid framework imports (torch/tensorflow)
so they remain lightweight and deterministic for contract-level checks.
"""

from pathlib import Path
from typing import Any, Callable


def parse_string_list_field(raw_value: Any, field_name: str) -> list[str]:
    """
    Parse a payload field that can be either a list of strings or a comma-separated string.
    """
    if isinstance(raw_value, str):
        return [part.strip() for part in raw_value.split(",") if part.strip()]
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    raise ValueError(f"{field_name} must be an array or comma-separated string.")


def normalize_training_mode(training_mode: str) -> str:
    """
    Normalize aliases used by frontend payloads.
    """
    return "mlp_dense" if training_mode == "mlp" else training_mode


def resolve_data_path_from_payload(
    data_path: Any,
    dataset_id: Any,
    resolve_dataset_path: Callable[[str], Path | None],
) -> tuple[str | None, bool]:
    """
    Resolve an explicit data_path or a dataset_id into an absolute path string.

    Returns:
    - resolved data path (or None)
    - dataset lookup attempted but missing flag
    """
    if data_path:
        return str(data_path), False
    if dataset_id:
        resolved_path = resolve_dataset_path(str(dataset_id))
        if resolved_path is None:
            return None, True
        return str(resolved_path), False
    return None, False


def validate_common_train_bounds(
    *,
    test_size: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> str | None:
    """
    Validate numeric bounds shared by train handlers.

    Returns an error string on failure, otherwise None.
    """
    if not 0 < test_size < 1:
        return "test_size must be > 0 and < 1."
    if not 1 <= epochs <= 500:
        return "epochs must be between 1 and 500."
    if not 1 <= batch_size <= 200:
        return "batch_size must be between 1 and 200."
    if not 0 < learning_rate <= 1:
        return "learning_rate must be > 0 and <= 1."
    return None


def validate_common_distill_bounds(
    *,
    test_size: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int | None,
    num_hidden_layers: int | None,
    temperature: float,
    alpha: float,
) -> str | None:
    """
    Validate numeric bounds shared by distill handlers.

    Returns an error string on failure, otherwise None.
    """
    train_error = validate_common_train_bounds(
        test_size=test_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    if train_error:
        return train_error
    if hidden_dim is not None and not 8 <= hidden_dim <= 500:
        return "student_hidden_dim must be between 8 and 500."
    if num_hidden_layers is not None and not 1 <= num_hidden_layers <= 15:
        return "student_num_hidden_layers must be between 1 and 15."
    if not 0 < temperature <= 20:
        return "temperature must be > 0 and <= 20."
    if not 0 <= alpha <= 1:
        return "alpha must be between 0 and 1."
    return None
