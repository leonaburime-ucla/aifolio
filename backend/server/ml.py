from __future__ import annotations

"""Shared FastAPI endpoint helpers for ML framework routes.

This module keeps `server.py` thin by centralizing repeated HTTP envelopes and
validation/error mapping logic used by PyTorch and TensorFlow endpoints.
"""

from pathlib import Path
from typing import Any, Callable

from fastapi.responses import JSONResponse

RuntimeHandler = Callable[[dict[str, Any], Callable[[str], Path | None], Path], tuple[int, dict[str, Any]]]
LoadBundleFn = Callable[[str], Any]
PredictRowsFn = Callable[[Any, list[Any], Any], list[Any]]


def runtime_unavailable_json(framework: str, package: str, details: str | None) -> JSONResponse:
    """Build a stable runtime-unavailable envelope for ML framework endpoints.

    Args:
        framework: Human-readable framework name (e.g. `PyTorch`).
        package: Runtime package name used in installation hint.
        details: Optional low-level import/runtime error details.

    Returns:
        `JSONResponse` with status 503 and standardized error payload.
    """
    return JSONResponse(
        status_code=503,
        content={
            "status": "error",
            "error": f"{framework} runtime is unavailable in this Python environment.",
            "details": details,
            "hint": f"Activate backend/.venv or install {package} in the interpreter running the server.",
        },
    )


def run_training_or_distill_endpoint(
    *,
    payload: dict[str, Any],
    handler: RuntimeHandler | None,
    framework: str,
    package: str,
    import_error: str | None,
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
):
    """Execute train/distill handler with consistent status/body handling.

    Args:
        payload: Incoming request body.
        handler: Framework train/distill handler callback.
        framework: Human-readable framework name.
        package: Runtime package name used in installation hint.
        import_error: Optional import error details when handler unavailable.
        resolve_dataset_path: Dataset ID resolver callback.
        artifacts_dir: Framework artifact root directory.

    Returns:
        Successful JSON body dict or `JSONResponse` on errors.
    """
    if handler is None:
        return runtime_unavailable_json(framework, package, import_error)

    status_code, response = handler(
        payload=payload,
        resolve_dataset_path=resolve_dataset_path,
        artifacts_dir=artifacts_dir,
    )
    if status_code != 200:
        return JSONResponse(status_code=status_code, content=response)
    return response


def run_predict_endpoint(
    *,
    payload: dict[str, Any],
    load_bundle: LoadBundleFn | None,
    predict_rows: PredictRowsFn | None,
    framework: str,
    package: str,
    import_error: str | None,
    artifacts_dir: Path,
    artifact_filename: str,
):
    """Execute prediction flow with shared validation/error envelopes.

    Args:
        payload: Incoming request body.
        load_bundle: Model bundle loader callback.
        predict_rows: Row prediction callback.
        framework: Human-readable framework name.
        package: Runtime package name used in installation hint.
        import_error: Optional import error details when runtime unavailable.
        artifacts_dir: Framework artifact root directory.
        artifact_filename: Default artifact filename when resolving `model_id`.

    Returns:
        Successful JSON body dict or `JSONResponse` on errors.
    """
    if load_bundle is None or predict_rows is None:
        return runtime_unavailable_json(framework, package, import_error)

    rows = payload.get("rows")
    if not isinstance(rows, list):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "rows must be an array of objects."},
        )

    model_path = payload.get("model_path")
    model_id = payload.get("model_id")
    if not model_path:
        if not model_id:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "model_path or model_id is required."},
            )
        model_path = str(artifacts_dir / model_id / artifact_filename)

    try:
        bundle = load_bundle(model_path)
        predictions = predict_rows(bundle, rows, device=payload.get("device"))
        return {
            "status": "ok",
            "model_path": model_path,
            "count": len(predictions),
            "predictions": predictions,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": str(exc)},
        )


def framework_status(*, import_error: str | None, package: str) -> dict[str, Any]:
    """Build a stable framework status payload for availability probes.

    Args:
        import_error: Import/runtime error details, if unavailable.
        package: Runtime package name used in installation hint.

    Returns:
        Status payload dictionary for `/ml/<framework>/status` endpoints.
    """
    available = import_error is None
    return {
        "status": "ok",
        "available": available,
        "error": import_error,
        "hint": None if available else f"Activate backend/.venv or install {package} in the interpreter running the server.",
    }
