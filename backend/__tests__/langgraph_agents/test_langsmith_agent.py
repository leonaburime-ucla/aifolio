import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PYTHON_ROOT.parent
for path in (str(PYTHON_ROOT), str(AI_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from agents import langsmith_agent


class _Result:
    def __init__(self, content):
        self.content = content


class _Model:
    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        return _Result(self._content)


def test_deterministic_fallback_disabled():
    assert langsmith_agent._deterministic_fallback({"enabled": False}) == "[LangSmith] Tracing is disabled."


def test_deterministic_fallback_with_summary():
    msg = langsmith_agent._deterministic_fallback({"enabled": True, "summary": {"status": "success", "total_tokens": 10, "model_call_count": 1, "tool_call_count": 2, "error_count": 0, "retry_count": 0, "total_latency_ms": 20}})
    assert msg.startswith("[LangSmith]")
    assert "no errors or retries" in msg


def test_interpret_langsmith_observability_uses_model_and_validates_prefix(monkeypatch):
    monkeypatch.setattr(langsmith_agent, "get_model", lambda model_id: _Model('{"langsmith_message":"[LangSmith] all good."}'))
    msg = langsmith_agent.interpret_langsmith_observability({"enabled": True, "summary": {"status": "success"}})
    assert msg == "[LangSmith] all good."


def test_stringify_content_parts_flattens_mixed_items():
    text = langsmith_agent._stringify_content_parts(
        [{"text": "a"}, {"content": "b"}, "c", 4]
    )
    assert text == "a\nb\nc\n4"


def test_strip_json_fence_removes_markdown_wrapper():
    assert langsmith_agent._strip_json_fence("```json\n{\"a\": 1}\n```") == '{"a": 1}'


def test_json_object_slice_returns_empty_when_missing_braces():
    assert langsmith_agent._json_object_slice("not json") == ""


def test_safe_json_parse_handles_lists_and_embedded_json():
    parsed = langsmith_agent._safe_json_parse(
        [{"text": "prefix "}, {"content": '{"langsmith_message":"[LangSmith] ok"}'}]
    )
    assert parsed == {"langsmith_message": "[LangSmith] ok"}


def test_safe_json_parse_returns_empty_dict_for_invalid_payload():
    assert langsmith_agent._safe_json_parse("```json\nnope\n```") == {}


def test_deterministic_fallback_with_report_status_and_error():
    msg = langsmith_agent._deterministic_fallback(
        {"enabled": True, "report_status": "failed", "report_error": "timeout"}
    )
    assert msg == "[LangSmith] report_status=failed error=timeout"


def test_deterministic_fallback_with_model_and_tool_calls_without_tokens():
    msg = langsmith_agent._deterministic_fallback(
        {"enabled": True, "summary": {"model_call_count": 2, "tool_call_count": 1}}
    )
    assert "2 model call(s)" in msg
    assert "1 tool call(s)" in msg


def test_interpret_langsmith_observability_uses_fallback_when_prefix_is_invalid(monkeypatch):
    monkeypatch.setattr(
        langsmith_agent,
        "get_model",
        lambda model_id: _Model('{"langsmith_message":"not prefixed"}'),
    )
    msg = langsmith_agent.interpret_langsmith_observability(
        {"enabled": True, "summary": {"status": "success", "error_count": 0, "retry_count": 0}}
    )
    assert msg.startswith("[LangSmith] This request is success.")


def test_interpret_langsmith_observability_uses_fallback_on_model_exception(monkeypatch):
    class _ExplodingModel:
        def invoke(self, messages):
            raise RuntimeError("boom")

    monkeypatch.setattr(langsmith_agent, "get_model", lambda model_id: _ExplodingModel())
    msg = langsmith_agent.interpret_langsmith_observability(
        {"enabled": True, "summary": {"status": "running"}}
    )
    assert msg == "[LangSmith] This request is running."


def test_summarize_langsmith_observability_uses_default_model(monkeypatch):
    monkeypatch.setattr(langsmith_agent, "get_model", lambda model_id: _Model('{"langsmith_message":"[LangSmith] summary."}'))
    msg = langsmith_agent.summarize_langsmith_observability(
        {"enabled": True, "summary": {"status": "success"}}
    )
    assert msg == "[LangSmith] summary."
