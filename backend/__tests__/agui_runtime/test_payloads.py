import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PYTHON_ROOT.parent
for path in (str(PYTHON_ROOT), str(AI_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from shared.agui_runtime import payloads


class _Msg:
    def __init__(self, role: str, content):
        self.role = role
        self.content = content


class _Tool:
    def __init__(self, name: str):
        self.name = name
        self.description = "desc"
        self.parameters = {"type": "object"}


class _Ctx:
    def __init__(self, description: str, value: str):
        self.description = description
        self.value = value


def test_extract_text_flattens_block_list():
    content = [{"text": "hello"}, {"content": "world"}, 123]
    assert payloads.extract_text(content) == "hello\nworld\n123"


def test_extract_text_handles_none_and_plain_string():
    assert payloads.extract_text(None) == ""
    assert payloads.extract_text("plain") == "plain"


def test_extract_attachments_filters_binary_blocks():
    content = [
        {"type": "binary", "mimeType": "text/csv", "filename": "x.csv", "url": "u"},
        {"type": "text", "text": "ignore"},
    ]
    result = payloads.extract_attachments(content)
    assert result == [{"type": "text/csv", "name": "x.csv", "url": "u", "data": None}]


def test_extract_attachments_returns_empty_for_non_list():
    assert payloads.extract_attachments("bad") == []


def test_decode_context_value_handles_json_wrapped_string():
    assert payloads.decode_context_value('"gemini-3"') == "gemini-3"


def test_decode_context_value_handles_scalars_and_invalid_json():
    assert payloads.decode_context_value("42") == "42"
    assert payloads.decode_context_value("true") == "True"
    assert payloads.decode_context_value("{bad") == "{bad"
    assert payloads.decode_context_value("") == ""


def test_extract_context_map_normalizes_descriptions():
    class _Ctx:
        def __init__(self, description: str, value: str):
            self.description = description
            self.value = value

    context_map = payloads.extract_context_map([_Ctx(" Tab ", '"pytorch"'), _Ctx("", "skip")])
    assert context_map == {"tab": "pytorch"}


def test_build_chat_payload_builds_expected_contract_and_logs():
    logs = []
    result = payloads.build_chat_payload(
        messages=[_Msg("user", "hello")],
        tools=[_Tool("switch_ag_ui_tab")],
        context=[_Ctx("tab", "\"pytorch\"")],
        requested_model="",
        default_model_id="m-default",
        debug_log=lambda event, **meta: logs.append((event, meta)),
    )

    assert result["model"] == "m-default"
    assert result["messages"][0]["content"] == "hello"
    assert result["tools"][0]["name"] == "switch_ag_ui_tab"
    assert result["context"][0]["description"] == "tab"
    assert any(event == "build_chat_payload.done" for event, _ in logs)


def test_build_chat_payload_preserves_attachments_and_requested_model():
    result = payloads.build_chat_payload(
        messages=[_Msg("user", [{"type": "binary", "mimeType": "text/plain", "filename": "a.txt", "data": "x"}])],
        tools=[],
        context=[],
        requested_model="m-requested",
        default_model_id="m-default",
    )
    assert result["model"] == "m-requested"
    assert result["messages"][0]["attachments"] == [{"type": "text/plain", "name": "a.txt", "url": None, "data": "x"}]
