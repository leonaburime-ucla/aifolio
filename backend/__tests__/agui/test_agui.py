import sys
from pathlib import Path
from types import SimpleNamespace

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

import shared.agui as agui


def test_is_debug_enabled_reads_copilot_debug_flag(monkeypatch):
    monkeypatch.setenv("COPILOT_DEBUG", "true")
    assert agui._is_debug_enabled() is True

    monkeypatch.setenv("COPILOT_DEBUG", "0")
    assert agui._is_debug_enabled() is False


def test_summarize_chart_spec_handles_list_dict_and_other():
    count, types = agui._summarize_chart_spec([{"type": "line"}, {"type": "bar"}, "x"])
    assert count == 3
    assert types == ["line", "bar"]

    count, types = agui._summarize_chart_spec({"type": "scatter"})
    assert count == 1
    assert types == ["scatter"]

    count, types = agui._summarize_chart_spec("none")
    assert count == 0
    assert types == []


def test_filter_non_planned_known_actions_drops_duplicates_and_unknown_actions():
    action_calls = [
        {"name": "a", "args": {"x": 1}},
        {"name": "a", "args": {"x": 2}},
        {"name": "unknown", "args": {}},
    ]
    planned_actions = [{"name": "a", "args": {"x": 1}}]
    filtered = agui._filter_non_planned_known_actions(action_calls, planned_actions, {"a"})
    assert filtered == [{"name": "a", "args": {"x": 2}}]


def test_build_tool_call_events_returns_start_args_end_triplet():
    tool_call_id, args_json, events = agui._build_tool_call_events(
        run_id="r1",
        message_id="m1",
        action_name="switch_ag_ui_tab",
        action_args={"tab": "pytorch"},
        sequence=2,
    )

    assert tool_call_id == "tool_r1_2_switch_ag_ui_tab"
    assert '"tab": "pytorch"' in args_json
    assert [event.type for event in events] == ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END"]


def test_has_tool_messages_after_latest_user_detects_tool_turns():
    input_data = SimpleNamespace(
        messages=[
            SimpleNamespace(role="user"),
            SimpleNamespace(role="assistant"),
            SimpleNamespace(role="tool"),
        ]
    )

    assert agui._has_tool_messages_after_latest_user(input_data) is True


def test_extract_latest_user_text_returns_last_user_message():
    input_data = SimpleNamespace(
        messages=[
            SimpleNamespace(role="user", content="first"),
            SimpleNamespace(role="assistant", content="ignored"),
            SimpleNamespace(role="user", content="second"),
        ]
    )

    assert agui._extract_latest_user_text(input_data) == "second"


def test_filter_messages_for_surface_keeps_full_history_for_agentic_research():
    messages = [
        SimpleNamespace(role="user", content="first"),
        SimpleNamespace(role="assistant", content="answer"),
        SimpleNamespace(role="user", content="second"),
    ]

    assert agui._filter_messages_for_surface(messages, active_tab="agentic-research") == messages


def test_filter_messages_for_surface_keeps_only_latest_user_for_non_research_tabs():
    latest_user = SimpleNamespace(role="user", content="latest")
    messages = [
        SimpleNamespace(role="user", content="first"),
        SimpleNamespace(role="assistant", content="answer"),
        latest_user,
    ]

    assert agui._filter_messages_for_surface(messages, active_tab="charts") == [latest_user]
    assert agui._filter_messages_for_surface(messages, active_tab="pytorch") == [latest_user]


def test_resolve_surface_dataset_id_only_allows_agentic_research():
    context_map = {
        "agentic_research_selected_dataset_id": "research-dataset",
        "ml_selected_dataset_id": "ml-dataset",
    }

    assert agui._resolve_surface_dataset_id(context_map, active_tab="agentic-research") == "research-dataset"
    assert agui._resolve_surface_dataset_id(context_map, active_tab="charts") is None
    assert agui._resolve_surface_dataset_id(context_map, active_tab="tensorflow") is None


def test_resolve_dataset_from_action_args_normalizes_friendly_dataset_names():
    assert agui._resolve_dataset_from_action_args({"dataset": "fraud detection"}) == "fraud_detection_phishing_websites.csv"
    assert agui._resolve_dataset_from_action_args({"dataset_id": "house prices"}) == "house_prices_ames.csv"
