from __future__ import annotations

"""Shared execution layer for ML train/distill handlers.

This module centralizes runtime trainer invocation, envelope shaping, and
error mapping that is common across framework handler implementations.
"""

from dataclasses import asdict
from typing import Any, Callable

from .artifacts import safe_file_size
from .handler_utils import compute_distill_stats
from .request_prep import PreparedDistillRequest, PreparedTrainRequest


def execute_train_request(
    *,
    runtime_trainer: Any,
    prepared: PreparedTrainRequest,
    payload: dict[str, Any],
    store_bundle: Callable[[Any], str],
) -> tuple[int, dict[str, Any]]:
    """Execute the shared training flow and build the HTTP response payload.

    Args:
        runtime_trainer: Framework trainer module/object with `train_model_from_file`
            and `save_bundle` methods.
        prepared: Normalized train inputs produced by `prepare_train_request`.
        payload: Original request payload (used for optional fields like `sheet_name`).
        store_bundle: Callback that stores in-memory bundle and returns a run ID.

    Returns:
        `(status_code, response_body)` compatible with framework handlers.
    """
    try:
        bundle, metrics = runtime_trainer.train_model_from_file(
            data_path=prepared.data_path,
            cfg=prepared.cfg,
            sheet_name=payload.get("sheet_name"),
            exclude_columns=prepared.exclude_columns,
            date_columns=prepared.date_columns,
            device=payload.get("device"),
        )
        run_id = store_bundle(bundle)
        model_path = (
            runtime_trainer.save_bundle(bundle, prepared.model_dir, metrics)
            if prepared.save_model and prepared.model_dir
            else None
        )
        return 200, {
            "status": "ok",
            "run_id": run_id,
            "model_id": prepared.model_id,
            "model_path": str(model_path) if model_path else None,
            "metrics": asdict(metrics),
        }
    except Exception as exc:
        return 400, {"status": "error", "error": str(exc)}


def execute_distill_request(
    *,
    runtime_trainer: Any,
    prepared: PreparedDistillRequest,
    payload: dict[str, Any],
    store_bundle: Callable[[Any], str],
    load_in_memory_bundle: Callable[[str], Any | None],
    parameter_count_fn: Callable[[Any], int],
    serialized_size_fn: Callable[[Any], int | None],
) -> tuple[int, dict[str, Any]]:
    """Execute the shared distillation flow and build the HTTP response payload.

    Args:
        runtime_trainer: Framework trainer module/object with distill and load/save
            bundle methods.
        prepared: Normalized distill inputs produced by `prepare_distill_request`.
        payload: Original request payload (used for optional fields like `sheet_name`).
        store_bundle: Callback that stores in-memory bundle and returns a run ID.
        load_in_memory_bundle: Callback that resolves teacher bundle by run ID.
        parameter_count_fn: Framework-specific model parameter counting function.
        serialized_size_fn: Framework-specific serialized-size fallback function.

    Returns:
        `(status_code, response_body)` compatible with framework handlers.
    """
    try:
        teacher_bundle = load_in_memory_bundle(str(prepared.teacher_run_id)) if prepared.teacher_run_id else None
        if prepared.teacher_run_id and teacher_bundle is None:
            return 404, {"status": "error", "error": "Teacher run not found or expired."}

        bundle, metrics = runtime_trainer.distill_model_from_file(
            data_path=prepared.data_path,
            cfg=prepared.cfg,
            teacher_path=prepared.teacher_model_path,
            teacher_bundle=teacher_bundle,
            sheet_name=payload.get("sheet_name"),
            exclude_columns=prepared.exclude_columns,
            date_columns=prepared.date_columns,
            device=payload.get("device"),
            temperature=prepared.numeric["temperature"],
            alpha=prepared.numeric["alpha"],
            student_hidden_dim=prepared.numeric["hidden_dim"],
            student_num_hidden_layers=prepared.numeric["num_hidden_layers"],
            student_dropout=prepared.numeric["student_dropout"],
        )

        model_path = (
            runtime_trainer.save_bundle(bundle, prepared.model_dir, metrics)
            if prepared.save_model and prepared.model_dir
            else None
        )

        teacher_model_size_bytes = safe_file_size(str(prepared.teacher_model_path))
        student_model_size_bytes = safe_file_size(str(model_path) if model_path else None)
        teacher_model_for_stats = (
            teacher_bundle
            if teacher_bundle is not None
            else runtime_trainer.load_bundle(str(prepared.teacher_model_path))
        )

        stats = compute_distill_stats(
            teacher_model_for_stats=teacher_model_for_stats.model,
            student_model=bundle.model,
            teacher_model_size_bytes=teacher_model_size_bytes,
            student_model_size_bytes=student_model_size_bytes,
            parameter_count_fn=parameter_count_fn,
            serialized_size_fn=serialized_size_fn,
        )

        run_id = store_bundle(bundle)
        return 200, {
            "status": "ok",
            "run_id": run_id,
            "model_id": prepared.model_id,
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
