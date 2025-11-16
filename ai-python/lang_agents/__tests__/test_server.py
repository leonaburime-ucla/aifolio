import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
