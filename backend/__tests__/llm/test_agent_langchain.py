import os
import sys
from pathlib import Path

# Prevent import-time key checks from failing in tests.
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

PYTHON_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PYTHON_ROOT.parent
for path in (str(PYTHON_ROOT), str(AI_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

import shared.agent_langchain as al


def test_format_tools_for_prompt_handles_empty():
    assert al._format_tools_for_prompt({}) == "No frontend tools were provided for this run."


def test_format_tools_for_prompt_filters_invalid_entries():
    text = al._format_tools_for_prompt(
        {"tools": [{"name": "navigate", "description": "Go somewhere"}, {"name": " "}, "bad"]}
    )
    assert text == "- navigate: Go somewhere"


def test_format_context_for_prompt_formats_pairs():
    text = al._format_context_for_prompt({"context": [{"description": "tab", "value": "pytorch"}]})
    assert "tab: pytorch" in text


def test_format_context_for_prompt_handles_empty_and_invalid_items():
    assert al._format_context_for_prompt({}) == "No additional app context."
    text = al._format_context_for_prompt({"context": [{"description": "", "value": ""}, "bad"]})
    assert text == "No additional app context."


def test_format_message_uses_history_roles():
    state = al.formatMessage({"messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "u"}]})
    assert len(state["messages"]) == 2
    assert state["messages"][0].content == "a"
    assert state["messages"][1].content == "u"


def test_format_message_uses_message_and_attachments_when_no_history():
    state = al.formatMessage({"message": "hello", "attachments": [{"id": "a1"}]})
    assert state["messages"][0].content == "hello"
    assert state["messages"][0].additional_kwargs == {"attachments": [{"id": "a1"}]}


def test_build_system_prompt_includes_planning_mode_and_context():
    prompt = al._build_system_prompt(
        {
            "response_mode": "actions_only",
            "tools": [{"name": "navigate_to_page", "description": "Navigate"}],
            "context": [{"description": "tab", "value": "research"}],
        }
    )
    assert "PLANNING MODE" in prompt
    assert "navigate_to_page" in prompt
    assert "tab: research" in prompt


def test_run_chat_builds_system_prompt_and_invokes_model(monkeypatch):
    captured = {}

    class _Result:
        content = '{"message":"ok","chartSpec":null}'

    class _Model:
        def invoke(self, messages):
            captured["messages"] = messages
            return _Result()

    monkeypatch.setattr(al, "get_model", lambda model_id: _Model())
    result = al.run_chat({"message": "hello", "model": "m1", "response_mode": "actions_only"})
    assert result == '{"message":"ok","chartSpec":null}'
    assert "PLANNING MODE" in captured["messages"][0].content
    assert captured["messages"][1].content == "hello"


def test_strip_json_fences():
    assert al._strip_json_fences("```json\n{\"message\":\"x\"}\n```") == '{"message":"x"}'


def test_parse_llm_json_extracts_embedded_object():
    parsed = al._parse_llm_json("prefix {\"message\":\"x\",\"chartSpec\":null} suffix")
    assert parsed == {"message": "x", "chartSpec": None}


def test_parse_llm_json_returns_none_for_invalid_input():
    assert al._parse_llm_json("no json here") is None
    assert al._parse_llm_json('{"chartSpec": null}') is None


def test_run_chat_response_quota_error(monkeypatch):
    monkeypatch.setattr(al, "run_chat", lambda payload: (_ for _ in ()).throw(RuntimeError("429 RESOURCE_EXHAUSTED")))
    monkeypatch.setattr(al, "record_run", lambda **kwargs: None)

    resp = al.run_chat_response({"message": "x"})
    assert resp.status_code == 429


def test_run_chat_response_success_parses_json(monkeypatch):
    monkeypatch.setattr(al, "run_chat", lambda payload: '{"message":"ok","chartSpec":null}')
    monkeypatch.setattr(al, "record_run", lambda **kwargs: None)

    result = al.run_chat_response({"message": "x", "model": "m1"})
    assert result["status"] == "ok"
    assert result["result"]["message"] == "ok"
    assert result["model"] == "m1"


def test_run_chat_response_success_returns_raw_string_when_not_json(monkeypatch):
    monkeypatch.setattr(al, "run_chat", lambda payload: "plain text")
    monkeypatch.setattr(al, "record_run", lambda **kwargs: None)
    result = al.run_chat_response({"message": "x"})
    assert result["result"] == "plain text"


def test_run_chat_response_returns_500_for_non_quota_error(monkeypatch):
    monkeypatch.setattr(al, "run_chat", lambda payload: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(al, "record_run", lambda **kwargs: None)
    resp = al.run_chat_response({"message": "x"})
    assert resp.status_code == 500
