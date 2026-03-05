import json
import sys
from pathlib import Path
import asyncio

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

import shared.agui as agui


def _build_payload(user_text: str = "switch to pytorch and train") -> dict:
    return {
        "threadId": "t1",
        "runId": "r1",
        "state": {},
        "forwardedProps": {},
        "messages": [{"id": "u1", "role": "user", "content": user_text}],
        "tools": [
            {"name": "switch_ag_ui_tab", "description": "x", "parameters": {}},
            {"name": "set_pytorch_form_fields", "description": "x", "parameters": {}},
            {"name": "start_pytorch_training_runs", "description": "x", "parameters": {}},
        ],
        "context": [],
    }


def _decode_event_line(encoded: str) -> dict:
    assert encoded.startswith("data: ")
    return json.loads(encoded[len("data: ") :])


async def _agui_stream_emits_serial_tool_calls_before_text(monkeypatch):
    monkeypatch.setattr(
        agui,
        "run_unified_action_plan",
        lambda payload: {
            "actions": [
                {"name": "switch_ag_ui_tab", "args": {"tab": "pytorch"}},
                {"name": "set_pytorch_form_fields", "args": {"fields": {"run_sweep": False}}},
            ],
            "planner_message": "",
        },
    )
    monkeypatch.setattr(
        agui,
        "run_unified_chat",
        lambda payload, force_provider=False: (
            "provider",
            {
                "message": "ok",
                "chartSpec": None,
                "actions": [{"name": "start_pytorch_training_runs", "args": {}}],
            },
        ),
    )

    events: list[dict] = []
    async for encoded in agui.agui_event_stream(_build_payload()):
        events.append(_decode_event_line(encoded))

    assert events[0]["type"] == "RUN_STARTED"

    tool_starts = [event for event in events if event.get("type") == "TOOL_CALL_START"]
    assert [event["toolCallName"] for event in tool_starts] == [
        "switch_ag_ui_tab",
        "set_pytorch_form_fields",
        "start_pytorch_training_runs",
    ]

    first_text_index = next(i for i, event in enumerate(events) if event["type"] == "TEXT_MESSAGE_START")
    last_tool_index = max(i for i, event in enumerate(events) if event["type"] == "TOOL_CALL_END")
    assert last_tool_index < first_text_index

    assert events[-1]["type"] == "RUN_FINISHED"


async def _agui_stream_skips_unknown_planned_actions(monkeypatch):
    monkeypatch.setattr(
        agui,
        "run_unified_action_plan",
        lambda payload: {
            "actions": [
                {"name": "unknown_tool", "args": {}},
                {"name": "switch_ag_ui_tab", "args": {"tab": "pytorch"}},
            ],
            "planner_message": "",
        },
    )
    monkeypatch.setattr(
        agui,
        "run_unified_chat",
        lambda payload, force_provider=False: ("provider", {"message": "ok", "chartSpec": None, "actions": []}),
    )

    events: list[dict] = []
    async for encoded in agui.agui_event_stream(_build_payload("switch to pytorch")):
        events.append(_decode_event_line(encoded))

    tool_starts = [event for event in events if event.get("type") == "TOOL_CALL_START"]
    assert len(tool_starts) == 1
    assert tool_starts[0]["toolCallName"] == "switch_ag_ui_tab"


async def _agui_stream_emits_run_error_on_exception(monkeypatch):
    def _raise(_payload: dict):
        raise RuntimeError("planner exploded")

    monkeypatch.setattr(agui, "run_unified_action_plan", _raise)

    events: list[dict] = []
    async for encoded in agui.agui_event_stream(_build_payload()):
        events.append(_decode_event_line(encoded))

    assert events[0]["type"] == "RUN_STARTED"
    assert events[-1]["type"] == "RUN_ERROR"
    assert "planner exploded" in events[-1]["message"]


def test_contract_agui_stream_emits_serial_tool_calls_before_text(monkeypatch):
    asyncio.run(_agui_stream_emits_serial_tool_calls_before_text(monkeypatch))


def test_contract_agui_stream_skips_unknown_planned_actions(monkeypatch):
    asyncio.run(_agui_stream_skips_unknown_planned_actions(monkeypatch))


def test_contract_agui_stream_emits_run_error_on_exception(monkeypatch):
    asyncio.run(_agui_stream_emits_run_error_on_exception(monkeypatch))
