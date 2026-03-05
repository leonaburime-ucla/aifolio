import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PYTHON_ROOT.parent
for path in (str(PYTHON_ROOT), str(AI_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

import shared.chartspec as chartspec


def test_extract_text_from_llm_output_handles_dict():
    assert chartspec.extract_text_from_llm_output({"message": "x"}) == '{"message": "x"}'


def test_extract_text_from_llm_output_handles_string_and_list():
    assert chartspec.extract_text_from_llm_output("plain text") == "plain text"
    assert chartspec.extract_text_from_llm_output([{"text": "a"}, "b", {"ignored": "c"}]) == "a\nb"


def test_strip_code_fences_removes_wrapper():
    assert chartspec.strip_code_fences("```json\n{\"message\":\"ok\"}\n```") == '{"message":"ok"}'


def test_parse_assistant_json_from_fenced_text():
    raw = "```json\n{\"message\":\"ok\",\"chartSpec\":null}\n```"
    parsed = chartspec.parse_assistant_json(raw)
    assert parsed == {"message": "ok", "chartSpec": None}


def test_parse_assistant_json_returns_none_for_invalid_payload():
    assert chartspec.parse_assistant_json("plain text") is None


def test_coerce_chart_data_rows_filters_non_dict_items():
    rows = chartspec._coerce_chart_data_rows([{"x": 1}, "bad", 3, {"x": 2}])
    assert rows == [{"x": 1}, {"x": 2}]


def test_normalize_chart_spec_rejects_invalid_type():
    assert chartspec.normalize_chart_spec({"type": "bad", "xKey": "x", "yKeys": ["y"], "data": [{"x": 1, "y": 2}]}) is None


def test_normalize_chart_spec_populates_defaults_and_optional_fields():
    normalized = chartspec.normalize_chart_spec(
        {
            "type": "line",
            "xKey": "date",
            "yKeys": ["sales", "", 1],
            "data": [{"date": "2024-01", "sales": 10}],
            "description": "Trend",
            "xLabel": "Month",
            "yLabel": "Sales",
            "zKey": "z",
            "colorKey": "region",
            "unit": "usd",
            "currency": "USD",
            "errorKeys": {"sales": "sales_error", "": "skip"},
            "timeframe": {"start": "2024-01", "end": "2024-12"},
            "source": {"provider": "internal", "url": "https://example.com"},
            "meta": {"datasetLabel": "Retail", "queryTimeMs": 12},
        }
    )
    assert normalized["id"] == "chart_1"
    assert normalized["title"] == "Line chart"
    assert normalized["yKeys"] == ["sales"]
    assert normalized["errorKeys"] == {"sales": "sales_error"}
    assert normalized["timeframe"] == {"start": "2024-01", "end": "2024-12"}
    assert normalized["source"] == {"provider": "internal", "url": "https://example.com"}
    assert normalized["meta"] == {"datasetLabel": "Retail", "queryTimeMs": 12}


def test_normalize_chart_spec_rejects_invalid_shapes():
    assert chartspec.normalize_chart_spec({"type": "line", "xKey": "", "yKeys": ["y"], "data": [{"x": 1}]}) is None
    assert chartspec.normalize_chart_spec({"type": "line", "xKey": "x", "yKeys": [], "data": [{"x": 1}]}) is None
    assert chartspec.normalize_chart_spec({"type": "line", "xKey": "x", "yKeys": ["y"], "data": []}) is None


def test_normalize_chart_spec_payload_list_filters_invalid_items():
    payload = [
        {"type": "line", "xKey": "x", "yKeys": ["y"], "data": [{"x": 1, "y": 2}]},
        {"type": "bad", "xKey": "x", "yKeys": ["y"], "data": [{"x": 1, "y": 2}]},
    ]
    normalized = chartspec.normalize_chart_spec_payload(payload)
    assert isinstance(normalized, list)
    assert len(normalized) == 1
    assert normalized[0]["type"] == "line"


def test_normalize_chart_spec_payload_handles_none_and_single_spec():
    assert chartspec.normalize_chart_spec_payload(None) is None
    normalized = chartspec.normalize_chart_spec_payload(
        {"type": "bar", "xKey": "x", "yKeys": ["y"], "data": [{"x": 1, "y": 2}]}
    )
    assert normalized["type"] == "bar"


def test_normalize_actions_payload_filters_invalid_actions():
    actions = chartspec.normalize_actions_payload(
        [
            {"name": "navigate", "args": {"route": "/charts"}},
            {"name": " ", "args": {"route": "/bad"}},
            {"name": "refresh", "args": "wrong"},
            "bad",
        ]
    )
    assert actions == [
        {"name": "navigate", "args": {"route": "/charts"}},
        {"name": "refresh", "args": {}},
    ]


def test_normalize_assistant_payload_fallbacks_to_text_message():
    result = chartspec.normalize_assistant_payload("plain text")
    assert result["type"] == "TextMessage"
    assert result["chartSpec"] is None


def test_normalize_assistant_payload_uses_result_wrapper_and_actions():
    result = chartspec.normalize_assistant_payload(
        {
            "result": {
                "message": "ok",
                "chartSpec": {"type": "bar", "xKey": "x", "yKeys": ["y"], "data": [{"x": 1, "y": 2}]},
                "actions": [{"name": "navigate", "args": {"route": "/charts"}}],
            }
        }
    )
    assert result["message"] == "ok"
    assert result["chartSpec"]["type"] == "bar"
    assert result["actions"] == [{"name": "navigate", "args": {"route": "/charts"}}]


def test_normalize_assistant_payload_uses_raw_text_when_parsed_message_is_invalid():
    result = chartspec.normalize_assistant_payload('{"message": 1, "chartSpec": null}')
    assert result["message"] == '{"message": 1, "chartSpec": null}'
    assert result["chartSpec"] is None


def test_format_assistant_json_text_returns_json_string():
    text = chartspec.format_assistant_json_text('{"message":"ok","chartSpec":null}')
    assert '"type": "TextMessage"' in text
