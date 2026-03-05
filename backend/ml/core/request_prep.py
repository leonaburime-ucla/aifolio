from __future__ import annotations

"""Shared request-preparation layer for ML train/distill handlers.

This module centralizes payload parsing, validation, and TrainingConfig assembly
that is common to both PyTorch and TensorFlow handler flows.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .artifacts import resolve_model_artifact_target
from .contracts import normalize_training_mode, validate_common_distill_bounds, validate_common_train_bounds
from .handler_utils import hidden_config_error, parse_distill_numeric, parse_train_numeric
from .mode_catalog import allowed_modes_error, unsupported_distill_mode_error
from .request_helpers import parse_feature_columns, resolve_data_target, resolve_teacher_model_path
from .types import TrainingConfig


@dataclass
class PreparedTrainRequest:
    """Normalized train-request inputs consumed by execution helpers/handlers."""

    data_path: str | Path
    exclude_columns: list[str]
    date_columns: list[str]
    cfg: TrainingConfig
    save_model: bool
    model_id: str | None
    model_dir: Path | None


@dataclass
class PreparedDistillRequest:
    """Normalized distill-request inputs consumed by execution helpers/handlers."""

    data_path: str | Path
    exclude_columns: list[str]
    date_columns: list[str]
    cfg: TrainingConfig
    numeric: dict[str, Any]
    teacher_run_id: Any
    teacher_model_path: str | None
    save_model: bool
    model_id: str | None
    model_dir: Path | None


def prepare_train_request(
    *,
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
    allowed_training_modes: set[str],
) -> tuple[PreparedTrainRequest | None, tuple[int, dict[str, Any]] | None]:
    """Validate and normalize a train payload into a prepared request object.

    Args:
        payload: Raw request body provided to the framework train endpoint.
        resolve_dataset_path: Callback that resolves a dataset ID to a file path.
        artifacts_dir: Base directory for optional model artifact output.
        allowed_training_modes: Framework-supported training mode identifiers.

    Returns:
        Tuple of `(prepared_request, error_response)` where one side is always
        `None`. `error_response` uses `(status_code, body)` shape.
    """
    data_path, target_column, data_error = resolve_data_target(payload, resolve_dataset_path)
    if data_error:
        return None, data_error

    save_model, model_id, model_dir = resolve_model_artifact_target(payload, artifacts_dir)

    exclude_columns, date_columns, list_error = parse_feature_columns(payload)
    if list_error:
        return None, (400, {"status": "error", "error": list_error})

    numeric, numeric_error = parse_train_numeric(payload)
    if numeric_error:
        return None, (400, {"status": "error", "error": numeric_error})
    assert numeric is not None

    bounds_error = validate_common_train_bounds(
        test_size=numeric["test_size"],
        epochs=numeric["epochs"],
        batch_size=numeric["batch_size"],
        learning_rate=numeric["learning_rate"],
    )
    if bounds_error:
        return None, (400, {"status": "error", "error": bounds_error})

    training_mode = normalize_training_mode(numeric["training_mode"])
    if training_mode not in allowed_training_modes:
        return None, (400, {"status": "error", "error": allowed_modes_error(sorted(allowed_training_modes))})

    config_error = hidden_config_error(
        training_mode=training_mode,
        hidden_dim=numeric["hidden_dim"],
        num_hidden_layers=numeric["num_hidden_layers"],
        dropout=numeric["dropout"],
    )
    if config_error:
        return None, (400, {"status": "error", "error": config_error})

    cfg = TrainingConfig(
        target_column=str(target_column),
        training_mode=training_mode,
        task=payload.get("task", "auto"),
        test_size=numeric["test_size"],
        random_seed=numeric["random_seed"],
        epochs=numeric["epochs"],
        batch_size=numeric["batch_size"],
        learning_rate=numeric["learning_rate"],
        hidden_dim=numeric["hidden_dim"],
        num_hidden_layers=numeric["num_hidden_layers"],
        dropout=numeric["dropout"],
    )

    return (
        PreparedTrainRequest(
            data_path=data_path,
            exclude_columns=exclude_columns,
            date_columns=date_columns,
            cfg=cfg,
            save_model=save_model,
            model_id=model_id,
            model_dir=model_dir,
        ),
        None,
    )


def prepare_distill_request(
    *,
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
    artifact_filename: str,
    allowed_training_modes: set[str],
    unsupported_distill_modes: set[str],
    unsupported_distill_hint: str,
) -> tuple[PreparedDistillRequest | None, tuple[int, dict[str, Any]] | None]:
    """Validate and normalize a distill payload into a prepared request object.

    Args:
        payload: Raw request body provided to the framework distill endpoint.
        resolve_dataset_path: Callback that resolves a dataset ID to a file path.
        artifacts_dir: Base directory for optional model artifact output.
        artifact_filename: Framework-specific artifact filename (e.g. `.pt`/`.keras`).
        allowed_training_modes: Framework-supported training mode identifiers.
        unsupported_distill_modes: Modes explicitly blocked for distillation.
        unsupported_distill_hint: User-facing hint for supported distill modes.

    Returns:
        Tuple of `(prepared_request, error_response)` where one side is always
        `None`. `error_response` uses `(status_code, body)` shape.
    """
    data_path, target_column, data_error = resolve_data_target(payload, resolve_dataset_path)
    if data_error:
        return None, data_error

    teacher_run_id = payload.get("teacher_run_id")
    teacher_model_path = resolve_teacher_model_path(payload, artifacts_dir, artifact_filename)
    if not teacher_model_path and not teacher_run_id:
        return None, (
            400,
            {
                "status": "error",
                "error": "teacher_run_id, teacher_model_path, or teacher_model_id is required.",
            },
        )

    save_model, model_id, model_dir = resolve_model_artifact_target(payload, artifacts_dir)

    exclude_columns, date_columns, list_error = parse_feature_columns(payload)
    if list_error:
        return None, (400, {"status": "error", "error": list_error})

    numeric, numeric_error = parse_distill_numeric(payload)
    if numeric_error:
        return None, (400, {"status": "error", "error": numeric_error})
    assert numeric is not None

    bounds_error = validate_common_distill_bounds(
        test_size=numeric["test_size"],
        epochs=numeric["epochs"],
        batch_size=numeric["batch_size"],
        learning_rate=numeric["learning_rate"],
        hidden_dim=numeric["hidden_dim"],
        num_hidden_layers=numeric["num_hidden_layers"],
        temperature=numeric["temperature"],
        alpha=numeric["alpha"],
    )
    if bounds_error:
        return None, (400, {"status": "error", "error": bounds_error})

    training_mode = normalize_training_mode(numeric["training_mode"])
    if training_mode not in allowed_training_modes:
        return None, (400, {"status": "error", "error": allowed_modes_error(sorted(allowed_training_modes))})
    if training_mode in unsupported_distill_modes:
        return None, (
            400,
            {
                "status": "error",
                "error": unsupported_distill_mode_error(training_mode, unsupported_distill_hint),
            },
        )

    cfg = TrainingConfig(
        target_column=str(target_column),
        training_mode=training_mode,
        task=payload.get("task", "auto"),
        test_size=numeric["test_size"],
        random_seed=numeric["random_seed"],
        epochs=numeric["epochs"],
        batch_size=numeric["batch_size"],
        learning_rate=numeric["learning_rate"],
        hidden_dim=numeric["hidden_dim"] or 128,
        num_hidden_layers=numeric["num_hidden_layers"] or 2,
        dropout=numeric["student_dropout"] if numeric["student_dropout"] is not None else 0.1,
    )

    return (
        PreparedDistillRequest(
            data_path=data_path,
            exclude_columns=exclude_columns,
            date_columns=date_columns,
            cfg=cfg,
            numeric=numeric,
            teacher_run_id=teacher_run_id,
            teacher_model_path=teacher_model_path,
            save_model=save_model,
            model_id=model_id,
            model_dir=model_dir,
        ),
        None,
    )
