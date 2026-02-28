import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from server import app
from google_gemini import normalize_model_id

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
    monkeypatch.setenv("GEMINI_AGENT_MODE", "mock")
    monkeypatch.setenv("GEMINI_MOCK_RESPONSE", "Hello from mock Gemini")

    payload = {"prompt": "Hi Gemini!"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    assert response.json()["message"] == "Hello from mock Gemini"


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


def test_gemini_models_excludes_commented_31_pro_option():
    response = client.get("/llm/gemini-models")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    model_ids = {model["id"] for model in data["models"]}
    assert "gemini-3.1-pro-preview" not in model_ids


def test_normalize_model_id_maps_legacy_flash_id():
    assert normalize_model_id("gemini-3-flash") == "gemini-3-flash-preview"
    assert normalize_model_id("gemini-3-flash-preview") == "gemini-3-flash-preview"
