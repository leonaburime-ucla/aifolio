import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PYTHON_ROOT.parent
for path in (str(PYTHON_ROOT), str(AI_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

import shared.chat_application_service as cas


def test_extract_latest_user_text_uses_latest_user_string_content():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ignored"},
        {"role": "user", "content": "latest"},
    ]
    assert cas._extract_latest_user_text(messages) == "latest"


def test_extract_latest_user_text_returns_empty_for_non_list_or_non_string_content():
    assert cas._extract_latest_user_text("bad") == ""
    assert cas._extract_latest_user_text(
        [
            {"role": "assistant", "content": "ignored"},
            {"role": "user", "content": {"text": "not-a-string"}},
        ]
    ) == ""


def test_resolve_message_prefers_message_then_prompt_then_history():
    assert cas._resolve_message({"message": " direct "}) == "direct"
    assert cas._resolve_message({"prompt": " prompt "}) == "prompt"
    assert cas._resolve_message({"messages": [{"role": "user", "content": "history"}]}) == "history"


def test_available_tool_names_normalizes_and_filters_invalid_entries():
    names = cas._available_tool_names(
        [{"name": "Start_PyTorch_Training_Runs"}, {"name": ""}, "ignored", {"missing": True}]
    )
    assert names == {"start pytorch training runs"}


def test_is_probable_ui_action_request_detects_tool_name_in_prompt():
    assert cas._is_probable_ui_action_request(
        "please start pytorch training",
        [{"name": "start_pytorch_training_runs"}],
    )


def test_is_probable_ui_action_request_detects_pattern_without_named_tool():
    assert cas._is_probable_ui_action_request("please open the charts page", None) is True
    assert cas._is_probable_ui_action_request("", [{"name": "tool"}]) is False


def test_run_unified_action_plan_returns_empty_without_tools():
    assert cas.run_unified_action_plan({"message": "anything", "tools": []}) == {
        "actions": [],
        "planner_message": "",
    }


def test_run_unified_action_plan_uses_latest_user_text_when_message_missing(monkeypatch):
    captured = {}

    def _run_chat(payload):
        captured["message"] = payload.get("message")
        return '{"message":"planned","actions":[{"name":"x","args":{}}]}'

    monkeypatch.setattr(cas, "run_chat", _run_chat)
    monkeypatch.setattr(cas, "normalize_assistant_payload", lambda raw: {"message": "planned", "actions": [{"name": "x", "args": {}}]})

    result = cas.run_unified_action_plan(
        {
            "messages": [{"role": "user", "content": "from history"}],
            "tools": [{"name": "x"}],
        }
    )

    assert captured["message"] == "from history"
    assert result == {"actions": [{"name": "x", "args": {}}], "planner_message": "planned"}


def test_run_unified_action_plan_normalizes_non_list_actions(monkeypatch):
    monkeypatch.setattr(cas, "run_chat", lambda payload: '{"message":"planned","actions":{"bad":true}}')
    monkeypatch.setattr(cas, "normalize_assistant_payload", lambda raw: {"message": "planned", "actions": {"bad": True}})

    result = cas.run_unified_action_plan({"prompt": "plan", "tools": [{"name": "x"}]})

    assert result == {"actions": [], "planner_message": "planned"}


def test_run_unified_chat_uses_coordinator_for_dataset_queries(monkeypatch):
    monkeypatch.setattr(cas, "coordinator_agent", lambda payload: {"message": "coordinator", "chartSpec": None})

    mode, payload = cas.run_unified_chat({"dataset_id": "ds1", "message": "analyze this"})

    assert mode == "coordinator"
    assert payload == {"message": "coordinator", "chartSpec": None, "actions": []}


def test_run_unified_chat_uses_provider_when_forced(monkeypatch):
    monkeypatch.setattr(cas, "run_chat", lambda payload: "raw")
    monkeypatch.setattr(cas, "normalize_assistant_payload", lambda raw: {"message": "provider", "chartSpec": None, "actions": []})

    mode, payload = cas.run_unified_chat(
        {"dataset_id": "ds1", "message": "analyze this"},
        force_provider=True,
    )

    assert mode == "provider"
    assert payload["message"] == "provider"


def test_run_unified_chat_uses_provider_for_ui_action_requests(monkeypatch):
    monkeypatch.setattr(cas, "run_chat", lambda payload: "raw")
    monkeypatch.setattr(cas, "normalize_assistant_payload", lambda raw: {"message": "provider", "chartSpec": None, "actions": []})

    mode, payload = cas.run_unified_chat(
        {
            "dataset_id": "ds1",
            "messages": [{"role": "user", "content": "open the charts page"}],
            "tools": [{"name": "add_chart_spec"}],
        }
    )

    assert mode == "provider"
    assert payload == {"message": "provider", "chartSpec": None, "actions": []}
