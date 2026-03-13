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


def _build_payload(
    user_text: str = "switch to pytorch and train",
    *,
    tools: list[dict] | None = None,
    context: list[dict] | None = None,
) -> dict:
    return {
        "threadId": "t1",
        "runId": "r1",
        "state": {},
        "forwardedProps": {},
        "messages": [{"id": "u1", "role": "user", "content": user_text}],
        "tools": tools if tools is not None else [
            {"name": "switch_ag_ui_tab", "description": "x", "parameters": {}},
            {"name": "set_pytorch_form_fields", "description": "x", "parameters": {}},
            {"name": "start_pytorch_training_runs", "description": "x", "parameters": {}},
        ],
        "context": context if context is not None else [],
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


async def _agui_stream_emits_agentic_research_remove_chart_tool(monkeypatch):
    monkeypatch.setattr(
        agui,
        "run_unified_action_plan",
        lambda payload: {
            "actions": [
                {"name": "ar-remove_chart_spec", "args": {"chart_id": "chart-1"}},
            ],
            "planner_message": "",
        },
    )
    monkeypatch.setattr(
        agui,
        "run_unified_chat",
        lambda payload, force_provider=False: ("provider", {"message": "removed", "chartSpec": None, "actions": []}),
    )

    payload = _build_payload(
        "Remove chart chart-1",
        tools=[
            {"name": "switch_ag_ui_tab", "description": "x", "parameters": {}},
            {"name": "ar-remove_chart_spec", "description": "x", "parameters": {}},
        ],
        context=[{"description": "ag_ui_active_tab", "value": "\"agentic-research\""}],
    )

    events: list[dict] = []
    async for encoded in agui.agui_event_stream(payload):
        events.append(_decode_event_line(encoded))

    tool_starts = [event for event in events if event.get("type") == "TOOL_CALL_START"]
    tool_args = [event["delta"] for event in events if event.get("type") == "TOOL_CALL_ARGS"]
    assert [event["toolCallName"] for event in tool_starts] == ["ar-remove_chart_spec"]
    assert tool_args == ['{"chart_id": "chart-1"}']
    assert events[-1]["type"] == "RUN_FINISHED"


async def _agui_stream_emits_agentic_research_reorder_chart_tool(monkeypatch):
    monkeypatch.setattr(
        agui,
        "run_unified_action_plan",
        lambda payload: {
            "actions": [
                {
                    "name": "ar-reorder_chart_specs",
                    "args": {"ordered_ids": ["chart-2", "chart-1"], "from_index": 1, "to_index": 0},
                },
            ],
            "planner_message": "",
        },
    )
    monkeypatch.setattr(
        agui,
        "run_unified_chat",
        lambda payload, force_provider=False: ("provider", {"message": "reordered", "chartSpec": None, "actions": []}),
    )

    payload = _build_payload(
        "Reorder charts so chart 2 comes before chart 1",
        tools=[
            {"name": "switch_ag_ui_tab", "description": "x", "parameters": {}},
            {"name": "ar-reorder_chart_specs", "description": "x", "parameters": {}},
        ],
        context=[{"description": "ag_ui_active_tab", "value": "\"agentic-research\""}],
    )

    events: list[dict] = []
    async for encoded in agui.agui_event_stream(payload):
        events.append(_decode_event_line(encoded))

    tool_starts = [event for event in events if event.get("type") == "TOOL_CALL_START"]
    tool_args = [event["delta"] for event in events if event.get("type") == "TOOL_CALL_ARGS"]
    assert [event["toolCallName"] for event in tool_starts] == ["ar-reorder_chart_specs"]
    assert tool_args == ['{"ordered_ids": ["chart-2", "chart-1"], "from_index": 1, "to_index": 0}']
    assert events[-1]["type"] == "RUN_FINISHED"


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


async def _agui_stream_skips_action_planning_for_agentic_research_pca_turn(monkeypatch):
    planned = {"called": False}
    captured = {}

    def _run_unified_action_plan(_payload: dict):
        planned["called"] = True
        return {"actions": [{"name": "set_pytorch_form_fields", "args": {"fields": {"run_sweep": False}}}]}

    def _run_unified_chat(payload, force_provider=False):
        captured["payload"] = payload
        captured["force_provider"] = force_provider
        return ("coordinator", {"message": "pca ready", "chartSpec": None, "actions": []})

    monkeypatch.setattr(agui, "run_unified_action_plan", _run_unified_action_plan)
    monkeypatch.setattr(agui, "run_unified_chat", _run_unified_chat)

    payload = _build_payload(
        "run pca",
        tools=[
            {"name": "switch_ag_ui_tab", "description": "x", "parameters": {}},
            {"name": "set_active_ml_form_fields", "description": "x", "parameters": {}},
            {"name": "start_active_ml_training_runs", "description": "x", "parameters": {}},
            {"name": "set_pytorch_form_fields", "description": "x", "parameters": {}},
            {"name": "start_pytorch_training_runs", "description": "x", "parameters": {}},
            {"name": "add_chart_spec", "description": "x", "parameters": {}},
            {"name": "ar-set_active_dataset", "description": "x", "parameters": {}},
        ],
        context=[
            {"description": "ag_ui_active_tab", "value": "\"agentic-research\""},
            {"description": "agentic_research_selected_dataset_id", "value": "\"telco-customer-churn\""},
        ],
    )

    events: list[dict] = []
    async for encoded in agui.agui_event_stream(payload):
        events.append(_decode_event_line(encoded))

    assert planned["called"] is False
    assert captured["force_provider"] is False
    assert captured["payload"]["dataset_id"] == "telco-customer-churn"
    assert [message["role"] for message in captured["payload"]["messages"]] == ["user"]
    assert {tool["name"] for tool in captured["payload"]["tools"]} == {
        "switch_ag_ui_tab",
        "add_chart_spec",
        "ar-set_active_dataset",
    }
    assert not any(event.get("type") == "TOOL_CALL_START" for event in events)
    assert events[-1]["type"] == "RUN_FINISHED"


def test_sanitize_runtime_tools_drops_stale_ml_tools_on_agentic_research():
    sanitized, dropped = agui._sanitize_runtime_tools(
        [
            {"name": "switch_ag_ui_tab", "description": "x", "parameters": {}},
            {"name": "set_active_ml_form_fields", "description": "x", "parameters": {}},
            {"name": "set_pytorch_form_fields", "description": "x", "parameters": {}},
            {"name": "add_chart_spec", "description": "x", "parameters": {}},
            {"name": "ar-set_active_dataset", "description": "x", "parameters": {}},
        ],
        active_tab="agentic-research",
        latest_user_text="run pca",
    )

    assert [tool["name"] for tool in sanitized] == [
        "switch_ag_ui_tab",
        "add_chart_spec",
        "ar-set_active_dataset",
    ]
    assert dropped == ["set_active_ml_form_fields", "set_pytorch_form_fields"]


async def _agui_stream_drops_chart_render_tool_actions(monkeypatch):
    monkeypatch.setattr(
        agui,
        "run_unified_action_plan",
        lambda payload: {
            "actions": [
                {"name": "add_chart_spec", "args": {"chartSpec": {"id": "from-plan"}}},
                {"name": "switch_ag_ui_tab", "args": {"tab": "charts"}},
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
                "message": "chart ready",
                "chartSpec": {
                    "id": "chart-1",
                    "type": "line",
                    "xKey": "month",
                    "yKeys": ["btc", "sol"],
                    "data": [{"month": "2025-10", "btc": 1, "sol": 2}],
                },
                "actions": [{"name": "add_chart_spec", "args": {"chartSpec": {"id": "from-response"}}}],
            },
        ),
    )

    events: list[dict] = []
    async for encoded in agui.agui_event_stream(
        _build_payload(
            "create a line chart of solana and bitcoin for the past 5 months",
            tools=[
                {"name": "switch_ag_ui_tab", "description": "x", "parameters": {}},
                {"name": "add_chart_spec", "description": "x", "parameters": {}},
            ],
            context=[{"description": "ag_ui_active_tab", "value": "\"charts\""}],
        )
    ):
        events.append(_decode_event_line(encoded))

    tool_starts = [event for event in events if event.get("type") == "TOOL_CALL_START"]
    assert [event["toolCallName"] for event in tool_starts] == ["switch_ag_ui_tab"]
    assert events[-1]["type"] == "RUN_FINISHED"


async def _agui_stream_strips_non_research_history_and_dataset(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        agui,
        "run_unified_chat",
        lambda payload, force_provider=False: (
            captured.update({"payload": payload, "force_provider": force_provider})
            or ("provider", {"message": "ok", "chartSpec": None, "actions": []})
        ),
    )

    payload = _build_payload(
        "make a scatter chart comparing bitcoin and ethereum returns over the last 30 days",
        tools=[
            {"name": "switch_ag_ui_tab", "description": "x", "parameters": {}},
            {"name": "add_chart_spec", "description": "x", "parameters": {}},
        ],
        context=[
            {"description": "ag_ui_active_tab", "value": "\"charts\""},
            {"description": "agentic_research_selected_dataset_id", "value": "\"customer_churn_telco.csv\""},
            {"description": "ml_selected_dataset_id", "value": "\"customer_churn_telco.csv\""},
        ],
    )
    payload["messages"] = [
        {"id": "u0", "role": "user", "content": "show rent chart"},
        {"id": "a0", "role": "assistant", "content": '{"type":"TextMessage","message":"previous"}'},
        {"id": "u1", "role": "user", "content": "make a scatter chart comparing bitcoin and ethereum returns over the last 30 days"},
    ]

    events: list[dict] = []
    async for encoded in agui.agui_event_stream(payload):
        events.append(_decode_event_line(encoded))

    assert captured["force_provider"] is False
    assert captured["payload"]["dataset_id"] is None
    assert captured["payload"]["messages"] == [
        {
            "role": "user",
            "content": "make a scatter chart comparing bitcoin and ethereum returns over the last 30 days",
            "attachments": [],
        }
    ]
    assert events[-1]["type"] == "RUN_FINISHED"


async def _agui_stream_does_not_reemit_tool_calls_after_tool_messages(monkeypatch):
    planned = {"called": False}

    def _run_unified_action_plan(_payload: dict):
        planned["called"] = True
        return {"actions": [{"name": "set_pytorch_form_fields", "args": {"fields": {"run_sweep": True}}}]}

    monkeypatch.setattr(agui, "run_unified_action_plan", _run_unified_action_plan)
    monkeypatch.setattr(
        agui,
        "run_unified_chat",
        lambda payload, force_provider=False: (
            "provider",
            {
                "message": "PyTorch already updated.",
                "chartSpec": None,
                "actions": [{"name": "start_pytorch_training_runs", "args": {}}],
            },
        ),
    )

    payload = _build_payload(
        "Use the fraud detection dataset. Switch the training algorithm from neural net to TabResNet.",
        tools=[
            {"name": "switch_ag_ui_tab", "description": "x", "parameters": {}},
            {"name": "set_active_ml_form_fields", "description": "x", "parameters": {}},
            {"name": "start_active_ml_training_runs", "description": "x", "parameters": {}},
            {"name": "set_pytorch_form_fields", "description": "x", "parameters": {}},
            {"name": "start_pytorch_training_runs", "description": "x", "parameters": {}},
        ],
        context=[{"description": "ag_ui_active_tab", "value": "\"pytorch\""}],
    )
    payload["messages"] = [
        {"id": "u1", "role": "user", "content": "Use the fraud detection dataset. Switch the training algorithm from neural net to TabResNet."},
        {"id": "a1", "role": "assistant", "content": '{"type":"TextMessage","message":"I updated the PyTorch configuration."}'},
        {
            "id": "t1",
            "role": "tool",
            "toolCallId": "tool-r1-1-set_pytorch_form_fields",
            "content": '{"status":"ok","applied":["dataset_id","training_mode"],"via":"bridge"}',
        },
    ]

    events: list[dict] = []
    async for encoded in agui.agui_event_stream(payload):
        events.append(_decode_event_line(encoded))

    assert planned["called"] is False
    assert not any(event.get("type") == "TOOL_CALL_START" for event in events)
    assert events[-1]["type"] == "RUN_FINISHED"


async def _agui_stream_prefers_serial_planned_ml_actions_over_provider_actions(monkeypatch):
    monkeypatch.setattr(
        agui,
        "run_unified_action_plan",
        lambda payload: {
            "actions": [
                {
                    "name": "set_active_ml_form_fields",
                    "args": {
                        "fields": {
                            "dataset": "fraud detection",
                            "model_type": "TabResNet",
                            "batch_size": [33, 40],
                            "hidden_dim": [64, 96],
                            "dropout": [0.1, 0.2],
                        }
                    },
                },
                {"name": "randomize_active_ml_form_fields", "args": {"value_count": 1}},
                {"name": "start_active_ml_training_runs", "args": {}},
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
                "message": "I updated the PyTorch configuration and started training.",
                "chartSpec": None,
                "actions": [
                    {
                        "name": "set_pytorch_form_fields",
                        "args": {"fields": {"model_architecture": "TabResNet"}},
                    },
                    {"name": "start_pytorch_training_runs", "args": {}},
                ],
            },
        ),
    )

    payload = _build_payload(
        "Use the fraud detection dataset. Switch the training algorithm from neural net to TabResNet. Randomize PyTorch form fields with one value each, and start training runs.",
        tools=[
            {"name": "switch_ag_ui_tab", "description": "x", "parameters": {}},
            {"name": "set_active_ml_form_fields", "description": "x", "parameters": {}},
            {"name": "randomize_active_ml_form_fields", "description": "x", "parameters": {}},
            {"name": "start_active_ml_training_runs", "description": "x", "parameters": {}},
            {"name": "set_pytorch_form_fields", "description": "x", "parameters": {}},
            {"name": "randomize_pytorch_form_fields", "description": "x", "parameters": {}},
            {"name": "start_pytorch_training_runs", "description": "x", "parameters": {}},
        ],
        context=[{"description": "ag_ui_active_tab", "value": "\"pytorch\""}],
    )

    events: list[dict] = []
    async for encoded in agui.agui_event_stream(payload):
        events.append(_decode_event_line(encoded))

    tool_starts = [event for event in events if event.get("type") == "TOOL_CALL_START"]
    assert [event["toolCallName"] for event in tool_starts] == [
        "set_pytorch_form_fields",
        "randomize_pytorch_form_fields",
        "start_pytorch_training_runs",
    ]

    tool_args = [event["delta"] for event in events if event.get("type") == "TOOL_CALL_ARGS"]
    assert '"training_mode": "tabresnet"' in tool_args[0]
    assert '"dataset_id": "fraud_detection_phishing_websites.csv"' in tool_args[0]
    assert len(tool_starts) == 3
    assert events[-1]["type"] == "RUN_FINISHED"


async def _agui_stream_strips_implicit_sweep_for_tensorflow_multi_value_prompt(monkeypatch):
    monkeypatch.setattr(
        agui,
        "run_unified_action_plan",
        lambda payload: {
            "actions": [
                {
                    "name": "set_tensorflow_form_fields",
                    "args": {
                        "fields": {
                            "dataset_id": "house_prices_ames.csv",
                            "training_mode": "wide_and_deep",
                            "test_sizes": [0.25, 0.3],
                            "batch_sizes": [32, 64],
                            "hidden_dims": [128, 256],
                            "set_sweep_values": True,
                            "run_sweep": True,
                        }
                    },
                }
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
                "message": "TensorFlow updated.",
                "chartSpec": None,
                "actions": [],
            },
        ),
    )

    payload = _build_payload(
        "Use the house prices dataset. Switch the training algorithm from neural net to wide and deep. Set test sizes to 0.25 and 0.3, batch sizes to 32 and 64, and hidden dims to 128 and 256.",
        tools=[
            {"name": "switch_ag_ui_tab", "description": "x", "parameters": {}},
            {"name": "set_active_ml_form_fields", "description": "x", "parameters": {}},
            {"name": "set_tensorflow_form_fields", "description": "x", "parameters": {}},
        ],
        context=[{"description": "ag_ui_active_tab", "value": "\"tensorflow\""}],
    )

    events: list[dict] = []
    async for encoded in agui.agui_event_stream(payload):
        events.append(_decode_event_line(encoded))

    tool_args = [event["delta"] for event in events if event.get("type") == "TOOL_CALL_ARGS"]
    assert len(tool_args) == 1
    assert '"set_sweep_values": true' not in tool_args[0]
    assert '"run_sweep": true' not in tool_args[0]
    assert '"test_sizes": [0.25, 0.3]' in tool_args[0]
    assert events[-1]["type"] == "RUN_FINISHED"


async def _agui_stream_strips_non_agentic_research_actions_from_assistant_payload(monkeypatch):
    monkeypatch.setattr(
        agui,
        "run_unified_action_plan",
        lambda payload: {
            "actions": [
                {"name": "ar-set_active_dataset", "args": {"dataset_id": "fraud_detection.csv"}},
            ],
            "planner_message": "",
        },
    )
    monkeypatch.setattr(
        agui,
        "run_unified_chat",
        lambda payload, force_provider=False: (
            "coordinator",
            {
                "message": "I have switched the active dataset to fraud detection and initiated a Lasso Regression training run.",
                "chartSpec": None,
                "actions": [
                    {"name": "ar-set_active_dataset", "args": {"dataset_id": "fraud_detection.csv"}},
                    {"name": "set_pytorch_form_fields", "args": {"fields": {"model_type": "lasso"}}},
                    {"name": "start_pytorch_training_runs", "args": {}},
                ],
            },
        ),
    )

    payload = _build_payload(
        "Change the dataset to fraud detection and run Lasso Regression",
        tools=[
            {"name": "switch_ag_ui_tab", "description": "x", "parameters": {}},
            {"name": "add_chart_spec", "description": "x", "parameters": {}},
            {"name": "ar-set_active_dataset", "description": "x", "parameters": {}},
        ],
        context=[{"description": "ag_ui_active_tab", "value": "\"agentic-research\""}],
    )

    events: list[dict] = []
    async for encoded in agui.agui_event_stream(payload):
        events.append(_decode_event_line(encoded))

    tool_starts = [event for event in events if event.get("type") == "TOOL_CALL_START"]
    assert [event["toolCallName"] for event in tool_starts] == ["ar-set_active_dataset"]

    text_events = [event for event in events if event.get("type") == "TEXT_MESSAGE_CONTENT"]
    assert len(text_events) == 1
    payload_text = text_events[0]["delta"]
    assert '"ar-set_active_dataset"' not in payload_text
    assert '"set_pytorch_form_fields"' not in payload_text
    assert '"start_pytorch_training_runs"' not in payload_text
    assert "Lasso Regression training run" in payload_text
    assert events[-1]["type"] == "RUN_FINISHED"


def test_contract_agui_stream_emits_serial_tool_calls_before_text(monkeypatch):
    asyncio.run(_agui_stream_emits_serial_tool_calls_before_text(monkeypatch))


def test_contract_agui_stream_skips_unknown_planned_actions(monkeypatch):
    asyncio.run(_agui_stream_skips_unknown_planned_actions(monkeypatch))


def test_contract_agui_stream_emits_agentic_research_remove_chart_tool(monkeypatch):
    asyncio.run(_agui_stream_emits_agentic_research_remove_chart_tool(monkeypatch))


def test_contract_agui_stream_emits_agentic_research_reorder_chart_tool(monkeypatch):
    asyncio.run(_agui_stream_emits_agentic_research_reorder_chart_tool(monkeypatch))


def test_contract_agui_stream_emits_run_error_on_exception(monkeypatch):
    asyncio.run(_agui_stream_emits_run_error_on_exception(monkeypatch))


def test_contract_agui_stream_skips_action_planning_for_agentic_research_pca_turn(monkeypatch):
    asyncio.run(_agui_stream_skips_action_planning_for_agentic_research_pca_turn(monkeypatch))


def test_contract_agui_stream_drops_chart_render_tool_actions(monkeypatch):
    asyncio.run(_agui_stream_drops_chart_render_tool_actions(monkeypatch))


def test_contract_agui_stream_strips_non_research_history_and_dataset(monkeypatch):
    asyncio.run(_agui_stream_strips_non_research_history_and_dataset(monkeypatch))


def test_contract_agui_stream_does_not_reemit_tool_calls_after_tool_messages(monkeypatch):
    asyncio.run(_agui_stream_does_not_reemit_tool_calls_after_tool_messages(monkeypatch))


def test_contract_agui_stream_prefers_serial_planned_ml_actions_over_provider_actions(monkeypatch):
    asyncio.run(_agui_stream_prefers_serial_planned_ml_actions_over_provider_actions(monkeypatch))


def test_contract_agui_stream_strips_implicit_sweep_for_tensorflow_multi_value_prompt(monkeypatch):
    asyncio.run(_agui_stream_strips_implicit_sweep_for_tensorflow_multi_value_prompt(monkeypatch))


def test_contract_agui_stream_strips_non_agentic_research_actions_from_assistant_payload(monkeypatch):
    asyncio.run(_agui_stream_strips_non_agentic_research_actions_from_assistant_payload(monkeypatch))
