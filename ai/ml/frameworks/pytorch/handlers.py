from __future__ import annotations

from dataclasses import asdict
import io
from pathlib import Path
from typing import Any, Callable
import uuid

from ...core.artifacts import safe_file_size
from ...core.contracts import normalize_training_mode, validate_common_distill_bounds, validate_common_train_bounds
from ...core.handler_utils import (
    compute_distill_stats,
    hidden_config_error,
    parse_distill_numeric,
    parse_train_numeric,
    runtime_unavailable_response,
)
from ...core.mode_catalog import (
    PYTORCH_ALLOWED_TRAINING_MODES,
    PYTORCH_UNSUPPORTED_DISTILL_MODES,
    allowed_modes_error,
    unsupported_distill_mode_error,
)
from ...core.request_helpers import parse_feature_columns, resolve_data_target, resolve_teacher_model_path
from ...core.types import ModelBundle, TrainingConfig
from ...distill import InMemoryBundleRegistry

_BUNDLE_REGISTRY: InMemoryBundleRegistry[ModelBundle] = InMemoryBundleRegistry(ttl_seconds=900, max_items=128)


def _runtime_trainer() -> tuple[Any | None, str | None]:
    try:
        from . import trainer as runtime_trainer

        return runtime_trainer, None
    except ModuleNotFoundError as exc:
        return None, str(exc)


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _serialized_model_size_bytes(model: Any) -> int | None:
    try:
        import torch

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return int(buffer.tell())
    except Exception:
        return None


def _store_in_memory_bundle(bundle: ModelBundle) -> str:
    return _BUNDLE_REGISTRY.store(bundle)


def _load_in_memory_bundle(run_id: str) -> ModelBundle | None:
    return _BUNDLE_REGISTRY.load(run_id)


def handle_train_request(
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
) -> tuple[int, dict[str, Any]]:
    runtime_trainer, runtime_error = _runtime_trainer()
    if runtime_trainer is None:
        return runtime_unavailable_response("PyTorch", runtime_error)

    data_path, target_column, error = resolve_data_target(payload, resolve_dataset_path)
    if error:
        return error

    save_model = bool(payload.get("save_model", False))
    model_id = (payload.get("model_id") or str(uuid.uuid4())) if save_model else None
    model_dir = (artifacts_dir / str(model_id)) if model_id else None

    exclude_columns, date_columns, list_error = parse_feature_columns(payload)
    if list_error:
        return 400, {"status": "error", "error": list_error}

    numeric, numeric_error = parse_train_numeric(payload)
    if numeric_error:
        return 400, {"status": "error", "error": numeric_error}
    assert numeric is not None

    bounds_error = validate_common_train_bounds(
        test_size=numeric["test_size"],
        epochs=numeric["epochs"],
        batch_size=numeric["batch_size"],
        learning_rate=numeric["learning_rate"],
    )
    if bounds_error:
        return 400, {"status": "error", "error": bounds_error}

    training_mode = normalize_training_mode(numeric["training_mode"])
    if training_mode not in PYTORCH_ALLOWED_TRAINING_MODES:
        return 400, {"status": "error", "error": allowed_modes_error(sorted(PYTORCH_ALLOWED_TRAINING_MODES))}

    config_error = hidden_config_error(
        training_mode=training_mode,
        hidden_dim=numeric["hidden_dim"],
        num_hidden_layers=numeric["num_hidden_layers"],
        dropout=numeric["dropout"],
    )
    if config_error:
        return 400, {"status": "error", "error": config_error}

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

    try:
        bundle, metrics = runtime_trainer.train_model_from_file(
            data_path=data_path,
            cfg=cfg,
            sheet_name=payload.get("sheet_name"),
            exclude_columns=exclude_columns,
            date_columns=date_columns,
            device=payload.get("device"),
        )
        run_id = _store_in_memory_bundle(bundle)
        model_path = runtime_trainer.save_bundle(bundle, model_dir, metrics) if save_model and model_dir else None
        return 200, {
            "status": "ok",
            "run_id": run_id,
            "model_id": model_id,
            "model_path": str(model_path) if model_path else None,
            "metrics": asdict(metrics),
        }
    except Exception as exc:
        return 400, {"status": "error", "error": str(exc)}


def _handle_distill_request_impl(
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
) -> tuple[int, dict[str, Any]]:
    runtime_trainer, runtime_error = _runtime_trainer()
    if runtime_trainer is None:
        return runtime_unavailable_response("PyTorch", runtime_error)

    data_path, target_column, error = resolve_data_target(payload, resolve_dataset_path)
    if error:
        return error

    teacher_run_id = payload.get("teacher_run_id")
    teacher_model_path = resolve_teacher_model_path(payload, artifacts_dir, "model_bundle.pt")
    if not teacher_model_path and not teacher_run_id:
        return 400, {
            "status": "error",
            "error": "teacher_run_id, teacher_model_path, or teacher_model_id is required.",
        }

    save_model = bool(payload.get("save_model", False))
    model_id = (payload.get("model_id") or str(uuid.uuid4())) if save_model else None
    model_dir = (artifacts_dir / str(model_id)) if model_id else None

    exclude_columns, date_columns, list_error = parse_feature_columns(payload)
    if list_error:
        return 400, {"status": "error", "error": list_error}

    numeric, numeric_error = parse_distill_numeric(payload)
    if numeric_error:
        return 400, {"status": "error", "error": numeric_error}
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
        return 400, {"status": "error", "error": bounds_error}

    training_mode = normalize_training_mode(numeric["training_mode"])
    if training_mode not in PYTORCH_ALLOWED_TRAINING_MODES:
        return 400, {"status": "error", "error": allowed_modes_error(sorted(PYTORCH_ALLOWED_TRAINING_MODES))}
    if training_mode in PYTORCH_UNSUPPORTED_DISTILL_MODES:
        return 400, {
            "status": "error",
            "error": unsupported_distill_mode_error(training_mode, "mlp_dense, linear_glm_baseline, or tabresnet"),
        }

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

    try:
        teacher_bundle = _load_in_memory_bundle(str(teacher_run_id)) if teacher_run_id else None
        if teacher_run_id and teacher_bundle is None:
            return 404, {"status": "error", "error": "Teacher run not found or expired."}

        bundle, metrics = runtime_trainer.distill_model_from_file(
            data_path=data_path,
            cfg=cfg,
            teacher_path=teacher_model_path,
            teacher_bundle=teacher_bundle,
            sheet_name=payload.get("sheet_name"),
            exclude_columns=exclude_columns,
            date_columns=date_columns,
            device=payload.get("device"),
            temperature=numeric["temperature"],
            alpha=numeric["alpha"],
            student_hidden_dim=numeric["hidden_dim"],
            student_num_hidden_layers=numeric["num_hidden_layers"],
            student_dropout=numeric["student_dropout"],
        )

        model_path = runtime_trainer.save_bundle(bundle, model_dir, metrics) if save_model and model_dir else None

        teacher_model_size_bytes = safe_file_size(str(teacher_model_path))
        student_model_size_bytes = safe_file_size(str(model_path) if model_path else None)
        teacher_model_for_stats = teacher_bundle if teacher_bundle is not None else runtime_trainer.load_bundle(str(teacher_model_path))

        stats = compute_distill_stats(
            teacher_model_for_stats=teacher_model_for_stats.model,
            student_model=bundle.model,
            teacher_model_size_bytes=teacher_model_size_bytes,
            student_model_size_bytes=student_model_size_bytes,
            parameter_count_fn=_parameter_count,
            serialized_size_fn=_serialized_model_size_bytes,
        )

        run_id = _store_in_memory_bundle(bundle)
        return 200, {
            "status": "ok",
            "run_id": run_id,
            "model_id": model_id,
            "model_path": str(model_path) if model_path else None,
            "metrics": asdict(metrics),
            "teacher_input_dim": int(teacher_model_for_stats.input_dim),
            "teacher_output_dim": int(teacher_model_for_stats.output_dim),
            "student_input_dim": int(bundle.input_dim),
            "student_output_dim": int(bundle.output_dim),
            **stats,
        }
    except Exception as exc:
        return 400, {"status": "error", "error": str(exc)}


def handle_distill_request(
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
) -> tuple[int, dict[str, Any]]:
    return _handle_distill_request_impl(payload, resolve_dataset_path, artifacts_dir)
