import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from server.http import app
from shared.google_gemini import normalize_model_id

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_llm_ping_roundtrip():
    payload = {"prompt": "Hello"}
    response = client.post("/llm/ping", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "LLM endpoint placeholder"
    assert data["received"] == payload


def test_chat_route_uses_mock_llm(monkeypatch):
    def _fake_unified_chat(payload: dict):
        assert payload.get("prompt") == "Hi Gemini!"
        return (
            "provider",
            {
                "message": "Hello from mock Gemini",
                "chartSpec": None,
                "actions": [],
            },
        )

    monkeypatch.setattr("server.http.run_unified_chat", _fake_unified_chat)

    payload = {"prompt": "Hi Gemini!"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    assert response.json()["message"] == "Hello from mock Gemini"


def test_run_chat_research_normalizes_message_and_actions(monkeypatch):
    monkeypatch.setattr(
        "server.http.run_unified_chat",
        lambda payload: ("provider", {"message": None, "chartSpec": None}),
    )
    from server.http import _run_chat_research

    result = _run_chat_research({"message": "hi"})
    assert result == {
        "status": "ok",
        "mode": "provider",
        "result": {"message": None, "chartSpec": None},
        "message": "",
        "chartSpec": None,
        "actions": [],
    }


def test_chat_get_uses_default_model(monkeypatch):
    monkeypatch.setattr("server.http.langchain_chat_response", lambda payload: payload)
    response = client.get("/chat?message=hello")
    assert response.status_code == 200
    assert response.json()["message"] == "hello"
    assert response.json()["model"] == normalize_model_id(None)


def test_agui_route_returns_stream_response(monkeypatch):
    monkeypatch.setattr("server.http.create_agui_stream_response", lambda payload: {"stream": payload})
    response = client.post("/agui", json={"run_id": "r1"})
    assert response.status_code == 200
    assert response.json() == {"stream": {"run_id": "r1"}}


def test_ml_data_list_returns_datasets():
    response = client.get("/ml-data")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["datasets"], list)
    assert any(item.get("id") == "customer_churn_telco.csv" for item in data["datasets"])


def test_ml_data_detail_returns_rows():
    response = client.get("/ml-data/customer_churn_telco.csv?row_limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["dataset"]["id"] == "customer_churn_telco.csv"
    assert isinstance(data["columns"], list)
    assert isinstance(data["rows"], list)
    assert data["rowCount"] <= 5


def test_gemini_models_includes_31_pro_option():
    response = client.get("/llm/gemini-models")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    model_ids = {model["id"] for model in data["models"]}
    assert "gemini-3.1-pro-preview" in model_ids


def test_normalize_model_id_maps_legacy_flash_id():
    assert normalize_model_id("gemini-3-flash") == "gemini-3-flash-preview"
    assert normalize_model_id("gemini-3-flash-preview") == "gemini-3-flash-preview"


def test_llm_ds_route_uses_coordinator_when_dataset_present(monkeypatch):
    monkeypatch.setattr("server.http.run_unified_chat", lambda payload: ("coordinator", {"message": "ok"}))
    response = client.post("/llm/ds", json={"dataset_id": "d1", "message": "analyze"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "coordinator", "result": {"message": "ok"}}


def test_llm_ds_route_uses_direct_tool_mode(monkeypatch):
    captured = {}

    def _run_tool(**kwargs):
        captured.update(kwargs)
        return {"message": "tool result", "chartSpec": None}

    monkeypatch.setattr("server.http.run_data_scientist_tool", _run_tool)
    response = client.post(
        "/llm/ds",
        json={"tool_name": "pca", "tool_args": {"n": 2}, "message": "run", "model": "m1"},
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "tool"
    assert captured == {
        "message": "run",
        "tool_name": "pca",
        "tool_args": {"n": 2},
        "model_id": "m1",
    }


def test_llm_ds_route_uses_chat_mode_without_dataset_or_tool(monkeypatch):
    monkeypatch.setattr(
        "server.http.run_data_scientist",
        lambda message, model_id: {"message": f"{message}:{model_id}"},
    )
    response = client.post("/llm/ds", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "chat", "result": {"message": "hello:gemini-3-flash-preview"}}


def test_llm_ds_get_returns_demo_payload(monkeypatch):
    monkeypatch.setattr("server.http.run_demo_pca_transform", lambda n_components: {"components": n_components})
    response = client.get("/llm/ds?n_components=3")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "pca-demo", "components": 3}


def test_agent_status_route_returns_status(monkeypatch):
    monkeypatch.setattr("server.http.get_status", lambda: {"agents": "ok"})
    response = client.get("/llm/agent-status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "data": {"agents": "ok"}}


def test_langsmith_trace_route_returns_404_for_error_report(monkeypatch):
    monkeypatch.setattr("server.http.get_trace_report", lambda **kwargs: {"status": "error", "error": "not found"})
    response = client.get("/llm/langsmith/trace/trace-1")
    assert response.status_code == 404
    assert response.json()["error"] == "not found"


def test_langsmith_trace_route_returns_500_on_exception(monkeypatch):
    monkeypatch.setattr("server.http.get_trace_report", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    response = client.get("/llm/langsmith/trace/trace-1")
    assert response.status_code == 500
    assert response.json()["error"] == "Failed to fetch LangSmith trace."


def test_sample_data_routes_handle_success_and_not_found(monkeypatch):
    monkeypatch.setattr("server.http.list_sample_datasets", lambda: [{"id": "s1"}])
    monkeypatch.setattr("server.http.load_sample_dataset", lambda dataset_id: {"status": "error", "error": "missing"})
    response = client.get("/sample-data")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "datasets": [{"id": "s1"}]}

    detail = client.get("/sample-data/missing")
    assert detail.status_code == 404
    assert detail.json()["error"] == "missing"


def test_ml_data_detail_returns_404_for_missing_dataset(monkeypatch):
    monkeypatch.setattr(
        "server.http.load_ml_dataset",
        lambda dataset_id, row_limit=None, sheet_name=None: {"status": "error", "error": "missing"},
    )
    response = client.get("/ml-data/missing")
    assert response.status_code == 404
    assert response.json()["error"] == "missing"


def test_sklearn_tools_route_returns_tools_and_schemas(monkeypatch):
    monkeypatch.setattr("server.http.sklearn_tools.list_available_tools", lambda: ["a"])
    monkeypatch.setattr("server.http.sklearn_tools.get_tools_schema", lambda: {"a": {}})
    response = client.get("/sklearn-tools")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "tools": ["a"], "schemas": {"a": {}}}


@pytest.mark.parametrize(
    ("path", "expected_framework", "expected_artifact"),
    [
        ("/ml/pytorch/train", "PyTorch", None),
        ("/ml/pytorch/distill", "PyTorch", None),
        ("/ml/tensorflow/train", "TensorFlow", None),
        ("/ml/tensorflow/distill", "TensorFlow", None),
    ],
)
def test_training_and_distill_routes_delegate_to_shared_endpoint(monkeypatch, path, expected_framework, expected_artifact):
    monkeypatch.setattr(
        "server.http.run_training_or_distill_endpoint",
        lambda **kwargs: {"framework": kwargs["framework"], "artifacts_dir": str(kwargs["artifacts_dir"])},
    )
    response = client.post(path, json={"dataset_id": "d1"})
    assert response.status_code == 200
    assert response.json()["framework"] == expected_framework


@pytest.mark.parametrize(
    ("path", "framework"),
    [
        ("/ml/pytorch/predict", "PyTorch"),
        ("/ml/tensorflow/predict", "TensorFlow"),
    ],
)
def test_predict_routes_delegate_to_shared_endpoint(monkeypatch, path, framework):
    monkeypatch.setattr(
        "server.http.run_predict_endpoint",
        lambda **kwargs: {"framework": kwargs["framework"], "artifact": kwargs["artifact_filename"]},
    )
    response = client.post(path, json={"rows": []})
    assert response.status_code == 200
    assert response.json()["framework"] == framework


@pytest.mark.parametrize(
    ("path", "package"),
    [
        ("/ml/pytorch/status", "torch"),
        ("/ml/tensorflow/status", "tensorflow"),
    ],
)
def test_framework_status_routes_delegate(monkeypatch, path, package):
    monkeypatch.setattr(
        "server.http.framework_status",
        lambda **kwargs: {"package": kwargs["package"], "import_error": kwargs["import_error"]},
    )
    response = client.get(path)
    assert response.status_code == 200
    assert response.json()["package"] == package
