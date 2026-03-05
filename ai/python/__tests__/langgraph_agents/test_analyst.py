import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PYTHON_ROOT.parent
for path in (str(PYTHON_ROOT), str(AI_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from backend.agents import analyst


class _Result:
    def __init__(self, content):
        self.content = content


class _Model:
    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        return _Result(self._content)


def test_safe_json_parse_handles_fenced_json():
    raw = "```json\n{\"analyst_summary\":\"ok\",\"findings\":[\"a\"]}\n```"
    parsed = analyst._safe_json_parse(raw)
    assert parsed["analyst_summary"] == "ok"


def test_stringify_content_parts_flattens_mixed_items():
    text = analyst._stringify_content_parts([{"text": "a"}, {"content": "b"}, "c"])
    assert text == "a\nb\nc"


def test_strip_json_fence_removes_wrapper():
    assert analyst._strip_json_fence("```json\n{\"a\": 1}\n```") == '{"a": 1}'


def test_json_object_slice_returns_empty_string_without_json():
    assert analyst._json_object_slice("plain text") == ""


def test_safe_json_parse_handles_list_and_embedded_json():
    parsed = analyst._safe_json_parse([{"text": "prefix "}, {"content": '{"analyst_summary":"ok"}'}])
    assert parsed == {"analyst_summary": "ok"}


def test_safe_json_parse_returns_empty_dict_for_invalid_text():
    assert analyst._safe_json_parse("not json") == {}


def test_format_conversation_history_truncates_and_limits():
    history = [{"role": "user", "content": "x" * 600}] * 8
    text = analyst._format_conversation_history(history)
    assert text.startswith("Previous conversation:")
    assert text.count("[USER]") == 6
    assert "..." in text


def test_format_conversation_history_returns_empty_string_for_no_history():
    assert analyst._format_conversation_history([]) == ""


def test_build_prompt_includes_history_and_context():
    prompt = analyst._build_prompt(
        user_request="What matters most?",
        dataset_label="Housing",
        data_scientist_message="Important coefficients found",
        charts=[{"type": "bar"}],
        non_chart_response={"r2": 0.91},
        dataset_metadata={"context": "prices"},
        conversation_history=[{"role": "user", "content": "previous question"}],
    )
    assert "Previous conversation:" in prompt
    assert "What matters most?" in prompt
    assert '"r2": 0.91' in prompt
    assert '"context": "prices"' in prompt


def test_interpret_analysis_returns_parsed_fields(monkeypatch):
    monkeypatch.setattr(analyst, "get_model", lambda model_id: _Model('{"analyst_summary":"done","findings":["f1"]}'))
    result = analyst.interpret_analysis("q", "d", "s", charts=[])
    assert result == {"analyst_summary": "done", "findings": ["f1"]}


def test_interpret_analysis_returns_defaults_for_unparseable_response(monkeypatch):
    monkeypatch.setattr(analyst, "get_model", lambda model_id: _Model("not-json"))
    result = analyst.interpret_analysis("q", "d", "s", charts=[])
    assert result == {"analyst_summary": "Could not parse response.", "findings": []}
