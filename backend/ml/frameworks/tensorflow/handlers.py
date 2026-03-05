from __future__ import annotations

"""TensorFlow HTTP handler adapters for train/distill backend endpoints.

This module wires TensorFlow-specific runtime/model details into shared core
request-prep and execution flows.
"""

from pathlib import Path
from typing import Any, Callable

from ...core.execution import execute_distill_request, execute_train_request
from ...core.handler_utils import runtime_unavailable_response
from ...core.mode_catalog import (
    TENSORFLOW_ALLOWED_TRAINING_MODES,
    TENSORFLOW_UNSUPPORTED_DISTILL_MODES,
)
from ...core.request_prep import prepare_distill_request, prepare_train_request
from ...core.runtime_imports import import_runtime_trainer
from ...core.types import ModelBundle
from ...distill import InMemoryBundleRegistry

_BUNDLE_REGISTRY: InMemoryBundleRegistry[ModelBundle] = InMemoryBundleRegistry(ttl_seconds=900, max_items=128)


def _runtime_trainer() -> tuple[Any | None, str | None]:
    """Resolve the TensorFlow runtime trainer module and import error details.

    Returns:
        Tuple `(trainer_module, error)` where one side is `None`.
    """
    return import_runtime_trainer("ml.frameworks.tensorflow.trainer")


def _parameter_count(model: Any) -> int:
    """Count trainable parameters for a TensorFlow model.

    Args:
        model: TensorFlow model instance exposing `count_params()`.

    Returns:
        Integer parameter count.
    """
    return int(model.count_params())


def _serialized_model_size_bytes(model: Any) -> int | None:
    """Estimate serialized TensorFlow model size in bytes.

    Args:
        model: TensorFlow model instance exposing `get_weights()`.

    Returns:
        Serialized byte size or `None` when serialization fails.
    """
    try:
        return int(sum(len(weights.tobytes()) for weights in model.get_weights()))
    except Exception:
        return None


def _store_in_memory_bundle(bundle: ModelBundle) -> str:
    """Store a model bundle in the in-memory registry.

    Args:
        bundle: Trained or distilled model bundle.

    Returns:
        Registry run identifier.
    """
    return _BUNDLE_REGISTRY.store(bundle)


def _load_in_memory_bundle(run_id: str) -> ModelBundle | None:
    """Load a model bundle from in-memory registry by run ID.

    Args:
        run_id: Bundle registry identifier.

    Returns:
        Stored model bundle or `None` when not found/expired.
    """
    return _BUNDLE_REGISTRY.load(run_id)


def handle_train_request(
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
) -> tuple[int, dict[str, Any]]:
    """Handle TensorFlow train endpoint payload.

    Args:
        payload: Incoming request body.
        resolve_dataset_path: Dataset ID resolver callback.
        artifacts_dir: TensorFlow artifact output directory.

    Returns:
        Tuple `(status_code, response_body)`.
    """
    runtime_trainer, runtime_error = _runtime_trainer()
    if runtime_trainer is None:
        return runtime_unavailable_response("TensorFlow", runtime_error)
    prepared, prep_error = prepare_train_request(
        payload=payload,
        resolve_dataset_path=resolve_dataset_path,
        artifacts_dir=artifacts_dir,
        allowed_training_modes=TENSORFLOW_ALLOWED_TRAINING_MODES,
    )
    if prep_error:
        return prep_error

    return execute_train_request(
        runtime_trainer=runtime_trainer,
        prepared=prepared,
        payload=payload,
        store_bundle=_store_in_memory_bundle,
    )


def _handle_distill_request_impl(
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
) -> tuple[int, dict[str, Any]]:
    """Internal implementation for TensorFlow distill endpoint payload.

    Args:
        payload: Incoming request body.
        resolve_dataset_path: Dataset ID resolver callback.
        artifacts_dir: TensorFlow artifact output directory.

    Returns:
        Tuple `(status_code, response_body)`.
    """
    runtime_trainer, runtime_error = _runtime_trainer()
    if runtime_trainer is None:
        return runtime_unavailable_response("TensorFlow", runtime_error)
    prepared, prep_error = prepare_distill_request(
        payload=payload,
        resolve_dataset_path=resolve_dataset_path,
        artifacts_dir=artifacts_dir,
        artifact_filename="model_bundle.keras",
        allowed_training_modes=TENSORFLOW_ALLOWED_TRAINING_MODES,
        unsupported_distill_modes=TENSORFLOW_UNSUPPORTED_DISTILL_MODES,
        unsupported_distill_hint="mlp_dense, linear_glm_baseline, or wide_and_deep",
    )
    if prep_error:
        return prep_error

    return execute_distill_request(
        runtime_trainer=runtime_trainer,
        prepared=prepared,
        payload=payload,
        store_bundle=_store_in_memory_bundle,
        load_in_memory_bundle=_load_in_memory_bundle,
        parameter_count_fn=_parameter_count,
        serialized_size_fn=_serialized_model_size_bytes,
    )


def handle_distill_request(
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
) -> tuple[int, dict[str, Any]]:
    """Handle TensorFlow distill endpoint payload.

    Args:
        payload: Incoming request body.
        resolve_dataset_path: Dataset ID resolver callback.
        artifacts_dir: TensorFlow artifact output directory.

    Returns:
        Tuple `(status_code, response_body)`.
    """
    return _handle_distill_request_impl(payload, resolve_dataset_path, artifacts_dir)
