from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from ml.ml_data import resolve_ml_dataset_path
from server.ml import framework_status as _framework_status
from server.ml import run_predict_endpoint as _run_predict_endpoint
from server.ml import run_training_or_distill_endpoint as _run_training_or_distill_endpoint
from server.routes.deps import resolve_http_override

router = APIRouter()

AI_ROOT = Path(__file__).resolve().parents[2]
PYTORCH_ARTIFACTS_DIR = AI_ROOT / "ml" / "artifacts"
TENSORFLOW_ARTIFACTS_DIR = AI_ROOT / "ml" / "tensorflow_artifacts"

PYTORCH_IMPORT_ERROR: str | None = None
PYTORCH_HANDLER_IMPORT_ERROR: str | None = None
PYTORCH_TRAINER_IMPORT_ERROR: str | None = None
try:
    from ml.frameworks.pytorch.handlers import (  # noqa: E402
        handle_distill_request as handle_pytorch_distill_request,
        handle_train_request as handle_pytorch_train_request,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    PYTORCH_HANDLER_IMPORT_ERROR = str(exc)
    handle_pytorch_distill_request = None  # type: ignore[assignment]
    handle_pytorch_train_request = None  # type: ignore[assignment]

try:
    from ml.frameworks.pytorch.trainer import (  # noqa: E402
        load_bundle as load_pytorch_bundle,
        predict_rows as predict_pytorch_rows,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    PYTORCH_TRAINER_IMPORT_ERROR = str(exc)
    load_pytorch_bundle = None  # type: ignore[assignment]
    predict_pytorch_rows = None  # type: ignore[assignment]
PYTORCH_IMPORT_ERROR = PYTORCH_HANDLER_IMPORT_ERROR or PYTORCH_TRAINER_IMPORT_ERROR

TENSORFLOW_IMPORT_ERROR: str | None = None
TENSORFLOW_HANDLER_IMPORT_ERROR: str | None = None
TENSORFLOW_TRAINER_IMPORT_ERROR: str | None = None
try:
    from ml.frameworks.tensorflow.handlers import (  # noqa: E402
        handle_distill_request as handle_tensorflow_distill_request,
        handle_train_request as handle_tensorflow_train_request,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    TENSORFLOW_HANDLER_IMPORT_ERROR = str(exc)
    handle_tensorflow_distill_request = None  # type: ignore[assignment]
    handle_tensorflow_train_request = None  # type: ignore[assignment]

try:
    from ml.frameworks.tensorflow.trainer import (  # noqa: E402
        load_bundle as load_tensorflow_bundle,
        predict_rows as predict_tensorflow_rows,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    TENSORFLOW_TRAINER_IMPORT_ERROR = str(exc)
    load_tensorflow_bundle = None  # type: ignore[assignment]
    predict_tensorflow_rows = None  # type: ignore[assignment]
TENSORFLOW_IMPORT_ERROR = TENSORFLOW_HANDLER_IMPORT_ERROR or TENSORFLOW_TRAINER_IMPORT_ERROR


def _runtime(name: str, fallback):
    return resolve_http_override(name, fallback)


@router.post("/ml/pytorch/train")
def pytorch_train(payload: dict):
    return _runtime("run_training_or_distill_endpoint", _run_training_or_distill_endpoint)(
        payload=payload,
        handler=_runtime("handle_pytorch_train_request", handle_pytorch_train_request),
        framework="PyTorch",
        package="torch",
        import_error=PYTORCH_IMPORT_ERROR,
        resolve_dataset_path=_runtime("resolve_ml_dataset_path", resolve_ml_dataset_path),
        artifacts_dir=PYTORCH_ARTIFACTS_DIR,
    )


@router.post("/ml/pytorch/distill")
def pytorch_distill(payload: dict):
    return _runtime("run_training_or_distill_endpoint", _run_training_or_distill_endpoint)(
        payload=payload,
        handler=_runtime("handle_pytorch_distill_request", handle_pytorch_distill_request),
        framework="PyTorch",
        package="torch",
        import_error=PYTORCH_IMPORT_ERROR,
        resolve_dataset_path=_runtime("resolve_ml_dataset_path", resolve_ml_dataset_path),
        artifacts_dir=PYTORCH_ARTIFACTS_DIR,
    )


@router.post("/ml/pytorch/predict")
def pytorch_predict(payload: dict):
    return _runtime("run_predict_endpoint", _run_predict_endpoint)(
        payload=payload,
        load_bundle=_runtime("load_pytorch_bundle", load_pytorch_bundle),
        predict_rows=_runtime("predict_pytorch_rows", predict_pytorch_rows),
        framework="PyTorch",
        package="torch",
        import_error=PYTORCH_IMPORT_ERROR,
        artifacts_dir=PYTORCH_ARTIFACTS_DIR,
        artifact_filename="model_bundle.pt",
    )


@router.get("/ml/pytorch/status")
def pytorch_status():
    return _runtime("framework_status", _framework_status)(import_error=PYTORCH_IMPORT_ERROR, package="torch")


@router.post("/ml/tensorflow/train")
def tensorflow_train(payload: dict):
    return _runtime("run_training_or_distill_endpoint", _run_training_or_distill_endpoint)(
        payload=payload,
        handler=_runtime("handle_tensorflow_train_request", handle_tensorflow_train_request),
        framework="TensorFlow",
        package="tensorflow",
        import_error=TENSORFLOW_IMPORT_ERROR,
        resolve_dataset_path=_runtime("resolve_ml_dataset_path", resolve_ml_dataset_path),
        artifacts_dir=TENSORFLOW_ARTIFACTS_DIR,
    )


@router.post("/ml/tensorflow/distill")
def tensorflow_distill(payload: dict):
    return _runtime("run_training_or_distill_endpoint", _run_training_or_distill_endpoint)(
        payload=payload,
        handler=_runtime("handle_tensorflow_distill_request", handle_tensorflow_distill_request),
        framework="TensorFlow",
        package="tensorflow",
        import_error=TENSORFLOW_IMPORT_ERROR,
        resolve_dataset_path=_runtime("resolve_ml_dataset_path", resolve_ml_dataset_path),
        artifacts_dir=TENSORFLOW_ARTIFACTS_DIR,
    )


@router.post("/ml/tensorflow/predict")
def tensorflow_predict(payload: dict):
    return _runtime("run_predict_endpoint", _run_predict_endpoint)(
        payload=payload,
        load_bundle=_runtime("load_tensorflow_bundle", load_tensorflow_bundle),
        predict_rows=_runtime("predict_tensorflow_rows", predict_tensorflow_rows),
        framework="TensorFlow",
        package="tensorflow",
        import_error=TENSORFLOW_IMPORT_ERROR,
        artifacts_dir=TENSORFLOW_ARTIFACTS_DIR,
        artifact_filename="model_bundle.keras",
    )


@router.get("/ml/tensorflow/status")
def tensorflow_status():
    return _runtime("framework_status", _framework_status)(import_error=TENSORFLOW_IMPORT_ERROR, package="tensorflow")
