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
