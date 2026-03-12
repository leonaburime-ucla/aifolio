"""AG-UI backend streaming orchestration for tool calls and assistant output.

This module converts AG-UI run payloads into an SSE event stream, including
planned tool-call events, provider/coordinator output, and terminal run events.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Iterable

from fastapi.responses import StreamingResponse
from ag_ui.core import (
    EventType,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder

from shared.agent_langchain import DEFAULT_MODEL_ID
from shared.agui_runtime.actions import (
    PYTORCH_EXECUTION_TOOL_NAMES,
    build_enforced_pytorch_actions,
    normalize_action_calls,
)
from shared.agui_runtime.intents import (
    AG_UI_TAB_ALIASES,
    NAV_ROUTE_ALIASES,
    is_pure_tab_switch_intent,
    resolve_ag_ui_tab_target,
    resolve_navigation_target,
)
from shared.agui_runtime.payloads import (
    build_chat_payload as build_chat_payload_from_runtime,
    decode_context_value,
    extract_attachments as extract_attachments_from_runtime,
    extract_context_map as extract_context_map_from_runtime,
    extract_text as extract_text_from_runtime,
)
from shared.chat_application_service import (
    is_probable_ui_action_request,
    run_unified_action_plan,
    run_unified_chat,
)
from shared.chartspec import format_assistant_json_text

# Backward-compatible aliases for existing imports/tests.
_resolve_navigation_target = resolve_navigation_target
_resolve_ag_ui_tab_target = resolve_ag_ui_tab_target
_is_pure_tab_switch_intent = is_pure_tab_switch_intent
_normalize_action_calls = normalize_action_calls
_build_enforced_pytorch_actions = build_enforced_pytorch_actions
_decode_context_value = decode_context_value

BASE_AG_UI_TOOL_NAMES: set[str] = {"switch_ag_ui_tab"}
GLOBAL_CHART_TOOL_NAMES: set[str] = {"add_chart_spec", "clear_charts"}
PYTORCH_TOOL_NAMES: set[str] = {
    "set_active_ml_form_fields",
    "change_active_ml_target_column",
    "randomize_active_ml_form_fields",
    "start_active_ml_training_runs",
    "set_pytorch_form_fields",
    "change_pytorch_target_column",
    "randomize_pytorch_form_fields",
    "start_pytorch_training_runs",
    "train_pytorch_model",
}
TENSORFLOW_TOOL_NAMES: set[str] = {
    "set_active_ml_form_fields",
    "change_active_ml_target_column",
    "randomize_active_ml_form_fields",
    "start_active_ml_training_runs",
    "set_tensorflow_form_fields",
    "change_tensorflow_target_column",
    "randomize_tensorflow_form_fields",
    "start_tensorflow_training_runs",
    "train_tensorflow_model",
}
AGENTIC_RESEARCH_TOOL_NAMES: set[str] = {
    "ar-add_chart_spec",
    "ar-clear_charts",
    "ar-remove_chart_spec",
    "ar-reorder_chart_specs",
    "ar-set_active_dataset",
    "remove_chart_spec",
    "reorder_chart_specs",
    "set_active_dataset",
}
CHART_RENDER_TOOL_NAMES: set[str] = {"add_chart_spec", "ar-add_chart_spec"}
CONTEXT_BOUND_TOOL_NAMES: set[str] = {
    "navigate_to_page",
    *BASE_AG_UI_TOOL_NAMES,
    *GLOBAL_CHART_TOOL_NAMES,
    *PYTORCH_TOOL_NAMES,
    *TENSORFLOW_TOOL_NAMES,
    *AGENTIC_RESEARCH_TOOL_NAMES,
}
TAB_SCOPED_TOOL_NAMES: dict[str, set[str]] = {
    "charts": BASE_AG_UI_TOOL_NAMES | GLOBAL_CHART_TOOL_NAMES,
    "agentic-research": BASE_AG_UI_TOOL_NAMES | GLOBAL_CHART_TOOL_NAMES | AGENTIC_RESEARCH_TOOL_NAMES,
    "pytorch": BASE_AG_UI_TOOL_NAMES | PYTORCH_TOOL_NAMES,
    "tensorflow": BASE_AG_UI_TOOL_NAMES | TENSORFLOW_TOOL_NAMES,
}

def _is_debug_enabled() -> bool:
    """Return whether AG-UI debug logging is enabled.

    Returns:
        `True` when debug logs should be emitted.
    """
    value = os.getenv("COPILOT_DEBUG", "")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _debug_log(event: str, **meta: Any) -> None:
    """
    Emit debug logs only when `COPILOT_DEBUG=1`.
    """
    if not _is_debug_enabled():
        return
    try:
        print(f"[agui] {event} {json.dumps(meta, default=str)}", flush=True)
    except Exception:
        print(f"[agui] {event} {meta}", flush=True)


def _format_assistant_text(raw_output: Any) -> str:
    """Normalize provider output into assistant JSON text.

    Args:
        raw_output: Raw output object/string from backend chat service.

    Returns:
        JSON-string assistant payload in AG-UI expected shape.
    """
    return format_assistant_json_text(raw_output)


def _has_frontend_tool(input_data: RunAgentInput, tool_name: str) -> bool:
    """Check whether a frontend tool is available for this AG-UI run.

    Args:
        input_data: Validated AG-UI input payload.
        tool_name: Tool name to check.

    Returns:
        `True` when tool exists in the run's tool registry.
    """
    return any(getattr(tool, "name", None) == tool_name for tool in input_data.tools)


def _extract_latest_user_text(input_data: RunAgentInput) -> str:
    """Extract latest user message text from AG-UI message history.

    Args:
        input_data: Validated AG-UI input payload.

    Returns:
        Latest user text content or empty string when none exists.
    """
    for message in reversed(input_data.messages):
        if getattr(message, "role", None) != "user":
            continue
        return extract_text(getattr(message, "content", ""))
    return ""


def _serialize_runtime_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Normalize runtime tool entries into `{name, description, parameters}` dicts."""
    serialized: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            serialized.append(
                {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                }
            )
            continue

        name = str(getattr(tool, "name", "") or "").strip()
        if not name:
            continue
        serialized.append(
            {
                "name": name,
                "description": getattr(tool, "description", ""),
                "parameters": getattr(tool, "parameters", {}),
            }
        )
    return serialized


def _sanitize_runtime_tools(
    tools: list[dict[str, Any]],
    *,
    active_tab: str,
    latest_user_text: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop stale tab-scoped tools while preserving current and explicit target tabs."""
    current_tab = active_tab if active_tab in TAB_SCOPED_TOOL_NAMES else "charts"
    target_tab = resolve_ag_ui_tab_target(latest_user_text or "")
    allowed_tool_names = set(TAB_SCOPED_TOOL_NAMES[current_tab])
    if target_tab and target_tab in TAB_SCOPED_TOOL_NAMES:
        allowed_tool_names.update(TAB_SCOPED_TOOL_NAMES[target_tab])

    sanitized: list[dict[str, Any]] = []
    dropped: list[str] = []
    seen_names: set[str] = set()
    for tool in tools:
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        if name in CONTEXT_BOUND_TOOL_NAMES and name not in allowed_tool_names:
            dropped.append(name)
            continue
        if name in seen_names:
            continue
        seen_names.add(name)
        sanitized.append(tool)
    return sanitized, dropped


def _build_tool_call_events(
    *,
    run_id: str,
    message_id: str,
    action_name: str,
    action_args: dict[str, Any],
    sequence: int,
) -> tuple[str, str, list[Any]]:
    """Build a deterministic set of TOOL_CALL_* events for one action.

    Args:
        run_id: Current run identifier.
        message_id: Parent assistant message identifier.
        action_name: Tool/action name.
        action_args: Tool/action arguments.
        sequence: Sequence index for unique tool-call IDs.

    Returns:
        Tuple of `(tool_call_id, args_json, events)`.
    """
    tool_call_id = f"tool_{run_id}_{sequence}_{action_name}"
    action_args_json = json.dumps(action_args, ensure_ascii=False)
    events: list[Any] = [
        ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=tool_call_id,
            tool_call_name=action_name,
            parent_message_id=message_id,
        ),
        ToolCallArgsEvent(
            type=EventType.TOOL_CALL_ARGS,
            tool_call_id=tool_call_id,
            delta=action_args_json,
        ),
        ToolCallEndEvent(
            type=EventType.TOOL_CALL_END,
            tool_call_id=tool_call_id,
        ),
    ]
    return tool_call_id, action_args_json, events




def _has_tool_messages_after_latest_user(input_data: RunAgentInput) -> bool:
    """Detect if tool messages exist after the latest user turn.

    Args:
        input_data: Validated AG-UI input payload.

    Returns:
        `True` when any post-user message has role `tool`.
    """
    latest_user_index = -1
    for index, message in enumerate(input_data.messages):
        if getattr(message, "role", None) == "user":
            latest_user_index = index
    if latest_user_index < 0:
        return False
    for message in input_data.messages[latest_user_index + 1 :]:
        if getattr(message, "role", None) == "tool":
            return True
    return False


def _decode_context_value(raw_value: Any) -> str:
    """Compatibility wrapper for AG-UI context value decoding.

    Args:
        raw_value: Raw context value.

    Returns:
        Decoded scalar string value.
    """
    return decode_context_value(raw_value)


def _extract_context_map(input_data: RunAgentInput) -> dict[str, str]:
    """Build normalized context map from AG-UI readable context items.

    Args:
        input_data: Validated AG-UI input payload.

    Returns:
        Mapping of lowercase context descriptions to decoded values.
    """
    return extract_context_map_from_runtime(input_data.context)


def _pick_auto_chart_tool_name(input_data: RunAgentInput, active_tab: str) -> str | None:
    """Choose preferred chart tool name based on active tab and availability.

    Args:
        input_data: Validated AG-UI input payload.
        active_tab: Current AG-UI tab key.

    Returns:
        Preferred chart tool name or `None` if unavailable.
    """
    preferred = "ar-add_chart_spec" if active_tab == "agentic-research" else "add_chart_spec"
    if _has_frontend_tool(input_data, preferred):
        return preferred
    if _has_frontend_tool(input_data, "add_chart_spec"):
        return "add_chart_spec"
    if _has_frontend_tool(input_data, "ar-add_chart_spec"):
        return "ar-add_chart_spec"
    return None


def _summarize_chart_spec(chart_spec: Any) -> tuple[int, list[Any]]:
    """Summarize chart payload shape for debug logging."""
    if isinstance(chart_spec, list):
        return len(chart_spec), [item.get("type") for item in chart_spec if isinstance(item, dict)]
    if isinstance(chart_spec, dict):
        return 1, [chart_spec.get("type")]
    return 0, []


def _filter_non_planned_known_actions(
    action_calls: list[dict[str, Any]],
    planned_actions: list[dict[str, Any]],
    available_tool_names: set[str | None],
) -> list[dict[str, Any]]:
    """Keep only known tool actions that were not already emitted during planning."""
    filtered: list[dict[str, Any]] = []
    for action in action_calls:
        action_name = action["name"]
        action_args = action["args"]
        if action_name in CHART_RENDER_TOOL_NAMES:
            continue
        has_planned_match = any(
            action_name == planned_action.get("name")
            and action_args == planned_action.get("args")
            for planned_action in planned_actions
        )
        if has_planned_match:
            continue
        if action_name not in available_tool_names:
            continue
        filtered.append(action)
    return filtered


def extract_text(content: Any) -> str:
    """Compatibility wrapper for AG-UI content-to-text normalization.

    Args:
        content: AG-UI content field.

    Returns:
        Flattened text content.
    """
    return extract_text_from_runtime(content)


def extract_attachments(content: Any) -> list[dict[str, Any]]:
    """Compatibility wrapper for AG-UI attachment extraction.

    Args:
        content: AG-UI content field.

    Returns:
        List of normalized attachments.
    """
    return extract_attachments_from_runtime(content)


def build_chat_payload(
    input_data: RunAgentInput,
    requested_model: str | None = None,
) -> dict[str, Any]:
    """Build provider payload from AG-UI run input.

    Args:
        input_data: Validated AG-UI input payload.
        requested_model: Optional explicit model override.

    Returns:
        Provider-compatible payload dictionary.
    """
    return build_chat_payload_from_runtime(
        messages=input_data.messages,
        tools=input_data.tools,
        context=input_data.context,
        requested_model=requested_model or DEFAULT_MODEL_ID,
        default_model_id=DEFAULT_MODEL_ID,
        debug_log=_debug_log,
    )


async def agui_event_stream(
    payload: dict[str, Any],
) -> Iterable[str]:
    """
    Build an AG-UI-compliant server-sent event stream.

    Args:
        payload: Raw AG-UI run payload from transport.

    Returns:
        Async iterable of encoded SSE event lines.

    Event order contract (happy path):
    1. RUN_STARTED
    2. TEXT_MESSAGE_START
    3. TEXT_MESSAGE_CONTENT
    4. TEXT_MESSAGE_END
    5. RUN_FINISHED

    Error contract:
    - Emit RUN_ERROR once and stop immediately.
    - Do not emit any additional events after RUN_ERROR.
      This prevents client-side protocol violations.
    """
    input_data = RunAgentInput.model_validate(payload)
    _debug_log(
        "stream.start",
        payload_keys=sorted(list(payload.keys())),
        incoming_messages=len(input_data.messages),
    )
    run_id = input_data.run_id or str(uuid.uuid4())
    thread_id = input_data.thread_id or str(uuid.uuid4())
    input_data = input_data.model_copy(update={"run_id": run_id, "thread_id": thread_id})
    encoder = EventEncoder()
    yield encoder.encode(
        RunStartedEvent(
            type=EventType.RUN_STARTED,
            run_id=run_id,
            thread_id=thread_id,
            input=input_data,
        )
    )
    _debug_log(
        "stream.emit",
        run_id=run_id,
        event_type=EventType.RUN_STARTED,
    )

    try:
        message_id = f"msg_{run_id}"
        latest_user_text = _extract_latest_user_text(input_data)
        # Disabled by request: backend fast-path for `switch_ag_ui_tab`.
        # if _has_frontend_tool(input_data, "switch_ag_ui_tab"):
        #     tab_target = _resolve_ag_ui_tab_target(latest_user_text)
        #     if tab_target and _is_pure_tab_switch_intent(latest_user_text):
        #         ...
        #         return

        # Disabled by request: backend fast-path for `navigate_to_page`.
        # if _has_frontend_tool(input_data, "navigate_to_page"):
        #     nav_route = _resolve_navigation_target(latest_user_text)
        #     if nav_route:
        #         ...
        #         return

        context_map = _extract_context_map(input_data)
        selected_model_from_context = context_map.get("ag_ui_selected_model_id", "").strip() or None
        requested_model = selected_model_from_context or payload.get("model")
        active_tab = context_map.get("ag_ui_active_tab", "").strip().lower()
        serialized_tools = _serialize_runtime_tools(input_data.tools)
        sanitized_tools, dropped_tool_names = _sanitize_runtime_tools(
            serialized_tools,
            active_tab=active_tab,
            latest_user_text=latest_user_text,
        )
        chat_payload = build_chat_payload_from_runtime(
            messages=input_data.messages,
            tools=sanitized_tools,
            context=input_data.context,
            requested_model=requested_model or DEFAULT_MODEL_ID,
            default_model_id=DEFAULT_MODEL_ID,
            debug_log=_debug_log,
        )
        selected_dataset_id = context_map.get("agentic_research_selected_dataset_id", "").strip() or None
        service_payload = {
            **chat_payload,
            "message": latest_user_text,
            "dataset_id": selected_dataset_id,
            "model": requested_model or DEFAULT_MODEL_ID,
        }
        should_plan_actions = is_probable_ui_action_request(latest_user_text, sanitized_tools)
        _debug_log(
            "service.plan.start",
            run_id=run_id,
            thread_id=thread_id,
            requested_model=requested_model or DEFAULT_MODEL_ID,
            selected_model_from_context=selected_model_from_context,
            active_tab=active_tab,
            dataset_id=selected_dataset_id,
            available_tool_names=[tool.get("name") for tool in sanitized_tools],
            dropped_tool_names=dropped_tool_names,
            should_plan_actions=should_plan_actions,
        )
        plan_result = (
            run_unified_action_plan(service_payload)
            if should_plan_actions
            else {"actions": [], "planner_message": ""}
        )
        planned_actions = _normalize_action_calls(plan_result.get("actions"))
        planned_actions = [
            action for action in planned_actions if action.get("name") not in CHART_RENDER_TOOL_NAMES
        ]
        _debug_log(
            "service.plan.done",
            run_id=run_id,
            action_count=len(planned_actions),
            action_names=[item.get("name") for item in planned_actions],
        )

        available_tool_names = {tool.get("name") for tool in sanitized_tools}
        serial_tool_events_emitted = 0
        for index, action in enumerate(planned_actions):
            action_name = action["name"]
            action_args = action["args"]
            if action_name not in available_tool_names:
                _debug_log(
                    "stream.skip.unknown_planned_action",
                    run_id=run_id,
                    action_name=action_name,
                )
                continue
            tool_call_id, action_args_json, events = _build_tool_call_events(
                run_id=run_id,
                message_id=message_id,
                action_name=action_name,
                action_args=action_args,
                sequence=index + 1,
            )
            for event in events:
                _debug_log(
                    "stream.emit",
                    run_id=run_id,
                    event_type=getattr(event, "type", None),
                    tool_call_id=tool_call_id,
                )
                yield encoder.encode(event)
            serial_tool_events_emitted += 1
            _debug_log(
                "stream.emit.frontend_tool_call",
                run_id=run_id,
                tool_call_id=tool_call_id,
                tool_name=action_name,
                args_preview=action_args_json[:300],
            )

        _debug_log(
            "service.generate.start",
            run_id=run_id,
            thread_id=thread_id,
            requested_model=requested_model or DEFAULT_MODEL_ID,
            selected_model_from_context=selected_model_from_context,
            active_tab=active_tab,
            dataset_id=selected_dataset_id,
            force_provider=serial_tool_events_emitted > 0,
        )
        mode, raw_output = run_unified_chat(
            service_payload,
            force_provider=serial_tool_events_emitted > 0,
        )
        _debug_log(
            "service.generate.mode",
            run_id=run_id,
            mode=mode,
            dataset_id=selected_dataset_id,
        )
        result_text = _format_assistant_text(raw_output)
        parsed_payload: dict[str, Any] | None = None
        chart_spec: Any = None
        action_calls: list[dict[str, Any]] = []
        try:
            normalized_payload = json.loads(result_text)
            parsed_payload = normalized_payload if isinstance(normalized_payload, dict) else None
            chart_spec = parsed_payload.get("chartSpec") if parsed_payload else None
            action_calls = _normalize_action_calls(parsed_payload.get("actions") if parsed_payload else None)
            chart_count, chart_types = _summarize_chart_spec(chart_spec)
            _debug_log(
                "service.generate.normalized_payload",
                run_id=run_id,
                has_message=bool(normalized_payload.get("message")),
                has_chart_spec=chart_spec is not None,
                action_call_count=len(action_calls),
                chart_count=chart_count,
                chart_types=chart_types,
                chart_spec_preview=chart_spec,
                message_preview=str(normalized_payload.get("message", ""))[:220],
            )
        except Exception:
            _debug_log(
                "service.generate.normalized_payload_parse_failed",
                run_id=run_id,
                preview=result_text[:300],
            )
        _debug_log(
            "service.generate.done",
            run_id=run_id,
            output_chars=len(result_text),
        )

        frontend_tool_events: list[Any] = []
        # Disabled by request: backend PyTorch action enforcement/filtering.
        # has_tool_messages_after_latest_user = _has_tool_messages_after_latest_user(input_data)
        # action_calls = _build_enforced_pytorch_actions(
        #     latest_user_text=latest_user_text,
        #     has_tool_messages_after_latest_user=has_tool_messages_after_latest_user,
        #     action_calls=action_calls,
        #     available_tool_names=available_tool_names,
        # )
        # _debug_log(
        #     "stream.actions.enforced",
        #     run_id=run_id,
        #     has_tool_messages_after_latest_user=has_tool_messages_after_latest_user,
        #     action_count=len(action_calls),
        #     action_names=[item.get("name") for item in action_calls],
        # )

        filtered_action_calls = _filter_non_planned_known_actions(action_calls, planned_actions, available_tool_names)
        for index, action in enumerate(filtered_action_calls):
            action_name = action["name"]
            action_args = action["args"]
            tool_call_id, action_args_json, events = _build_tool_call_events(
                run_id=run_id,
                message_id=message_id,
                action_name=action_name,
                action_args=action_args,
                sequence=serial_tool_events_emitted + index + 1,
            )
            frontend_tool_events.extend(events)
            _debug_log(
                "stream.emit.frontend_tool_call",
                run_id=run_id,
                tool_call_id=tool_call_id,
                tool_name=action_name,
                args_preview=action_args_json[:300],
            )

        # Do not auto-emit add_chart_spec tool calls from chartSpec payloads.
        # Chart rendering is handled by frontend message parsing/chart bridge.
        # Auto tool calls can trigger recursive follow-up runs in Copilot transport.

        success_events = [
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=message_id,
                role="assistant",
            ),
            TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=message_id,
                delta=result_text,
            ),
            TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=message_id,
            ),
            RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                run_id=run_id,
                thread_id=thread_id,
            ),
        ]
        for event in [*frontend_tool_events, *success_events]:
            _debug_log(
                "stream.emit",
                run_id=run_id,
                event_type=getattr(event, "type", None),
            )
            yield encoder.encode(event)
    except Exception as exc:
        _debug_log(
            "stream.error",
            run_id=run_id,
            thread_id=thread_id,
            error=str(exc),
        )
        # Emit only RUN_ERROR on failures to avoid protocol violations.
        error_event = RunErrorEvent(
            type=EventType.RUN_ERROR,
            run_id=run_id,
            thread_id=thread_id,
            message=str(exc),
        )
        _debug_log(
            "stream.emit",
            run_id=run_id,
            event_type=getattr(error_event, "type", None),
        )
        yield encoder.encode(error_event)


def create_agui_stream_response(
    payload: dict[str, Any],
) -> StreamingResponse:
    """
    FastAPI response factory for AG-UI streams.

    Keeps route handlers simple:
    - `server.py` delegates route payload directly here.
    - feature-specific behavior remains in this module.

    Args:
        payload: Raw AG-UI run payload.

    Returns:
        `StreamingResponse` that emits AG-UI protocol events.
    """
    return StreamingResponse(
        agui_event_stream(payload),
        media_type="text/event-stream",
    )
