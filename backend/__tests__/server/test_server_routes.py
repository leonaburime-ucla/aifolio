import sys
from pathlib import Path

from fastapi.testclient import TestClient

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

import server.http as server

client = TestClient(server.app)


def test_run_chat_research_normalizes_response(monkeypatch):
    monkeypatch.setattr(
        server,
        "run_unified_chat",
        lambda payload: ("provider", {"message": "hello", "chartSpec": [{"id": "c1"}], "actions": [{"name": "x"}]}),
    )

    result = server._run_chat_research({"prompt": "Hi"})
    assert result == {
        "status": "ok",
        "mode": "provider",
        "result": {"message": "hello", "chartSpec": [{"id": "c1"}], "actions": [{"name": "x"}]},
        "message": "hello",
        "chartSpec": [{"id": "c1"}],
        "actions": [{"name": "x"}],
    }


def test_chat_research_route_uses_shared_envelope(monkeypatch):
    monkeypatch.setattr(
        server,
        "run_unified_chat",
        lambda payload: ("provider", {"message": "researched", "chartSpec": None, "actions": []}),
    )

    response = client.post("/chat-research", json={"prompt": "research this"})
    assert response.status_code == 200
    assert response.json()["message"] == "researched"
