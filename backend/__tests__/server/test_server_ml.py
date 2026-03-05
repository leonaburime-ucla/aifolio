import json
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from fastapi.responses import JSONResponse

from server.ml import (
    framework_status,
    run_predict_endpoint,
    run_training_or_distill_endpoint,
    runtime_unavailable_json,
)


def _json_body(response: JSONResponse) -> dict:
    return json.loads(response.body.decode("utf-8"))


def test_runtime_unavailable_json_contract():
    response = runtime_unavailable_json("PyTorch", "torch", "No module named torch")
    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    payload = _json_body(response)
    assert payload["status"] == "error"
    assert payload["error"] == "PyTorch runtime is unavailable in this Python environment."
    assert "install torch" in payload["hint"]


def test_run_training_or_distill_endpoint_handles_none_handler():
    response = run_training_or_distill_endpoint(
        payload={},
        handler=None,
        framework="TensorFlow",
        package="tensorflow",
        import_error="missing",
        resolve_dataset_path=lambda _: None,
        artifacts_dir=Path("/tmp"),
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 503


def test_run_training_or_distill_endpoint_passthrough_status():
    def _handler(payload, resolve_dataset_path, artifacts_dir):
        return 400, {"status": "error", "error": "bad request"}

    response = run_training_or_distill_endpoint(
        payload={},
        handler=_handler,
        framework="PyTorch",
        package="torch",
        import_error=None,
        resolve_dataset_path=lambda _: None,
        artifacts_dir=Path("/tmp"),
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert _json_body(response)["error"] == "bad request"


def test_run_training_or_distill_endpoint_returns_success_body():
    def _handler(payload, resolve_dataset_path, artifacts_dir):
        return 200, {"status": "ok", "framework": "pytorch"}

    response = run_training_or_distill_endpoint(
        payload={},
        handler=_handler,
        framework="PyTorch",
        package="torch",
        import_error=None,
        resolve_dataset_path=lambda _: None,
        artifacts_dir=Path("/tmp"),
    )
    assert response == {"status": "ok", "framework": "pytorch"}


def test_run_predict_endpoint_success_with_model_id_path_resolution():
    class _Bundle:
        pass

    def _load_bundle(path: str):
        assert path.endswith("abc/model_bundle.pt")
        return _Bundle()

    def _predict_rows(bundle, rows, device=None):
        assert rows == [{"x": 1}]
        assert device is None
        return [0.25]

    response = run_predict_endpoint(
        payload={"rows": [{"x": 1}], "model_id": "abc"},
        load_bundle=_load_bundle,
        predict_rows=_predict_rows,
        framework="PyTorch",
        package="torch",
        import_error=None,
        artifacts_dir=Path("/tmp"),
        artifact_filename="model_bundle.pt",
    )

    assert response["status"] == "ok"
    assert response["count"] == 1
    assert response["predictions"] == [0.25]


def test_run_predict_endpoint_returns_runtime_unavailable_when_runtime_missing():
    response = run_predict_endpoint(
        payload={"rows": []},
        load_bundle=None,
        predict_rows=None,
        framework="PyTorch",
        package="torch",
        import_error="missing",
        artifacts_dir=Path("/tmp"),
        artifact_filename="model_bundle.pt",
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 503


def test_run_predict_endpoint_validates_rows_and_model_reference():
    invalid_rows_response = run_predict_endpoint(
        payload={"rows": "bad"},
        load_bundle=lambda path: None,
        predict_rows=lambda bundle, rows, device=None: [],
        framework="PyTorch",
        package="torch",
        import_error=None,
        artifacts_dir=Path("/tmp"),
        artifact_filename="model_bundle.pt",
    )
    assert invalid_rows_response.status_code == 400
    assert _json_body(invalid_rows_response)["error"] == "rows must be an array of objects."

    missing_model_response = run_predict_endpoint(
        payload={"rows": []},
        load_bundle=lambda path: None,
        predict_rows=lambda bundle, rows, device=None: [],
        framework="PyTorch",
        package="torch",
        import_error=None,
        artifacts_dir=Path("/tmp"),
        artifact_filename="model_bundle.pt",
    )
    assert missing_model_response.status_code == 400
    assert _json_body(missing_model_response)["error"] == "model_path or model_id is required."


def test_run_predict_endpoint_maps_prediction_exceptions_to_400():
    response = run_predict_endpoint(
        payload={"rows": [{"x": 1}], "model_path": "/tmp/model.pt", "device": "cpu"},
        load_bundle=lambda path: {"path": path},
        predict_rows=lambda bundle, rows, device=None: (_ for _ in ()).throw(RuntimeError("predict exploded")),
        framework="PyTorch",
        package="torch",
        import_error=None,
        artifacts_dir=Path("/tmp"),
        artifact_filename="model_bundle.pt",
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert _json_body(response)["error"] == "predict exploded"


def test_framework_status_contract():
    assert framework_status(import_error=None, package="torch") == {
        "status": "ok",
        "available": True,
        "error": None,
        "hint": None,
    }


def test_framework_status_includes_hint_when_runtime_missing():
    payload = framework_status(import_error="missing", package="tensorflow")
    assert payload["available"] is False
    assert "install tensorflow" in payload["hint"]
