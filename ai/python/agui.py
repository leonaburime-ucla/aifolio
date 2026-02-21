from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

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

from agent_langchain import DEFAULT_MODEL_ID, run_chat
from chartspec import format_assistant_json_text

NAV_ROUTE_ALIASES: dict[str, str] = {
    "/": "/",
    "home": "/",
    "ai chat": "/",
    "chat": "/",
    "ag-ui": "/ag-ui",
    "agui": "/ag-ui",
    "agentic research": "/agentic-research",
    "agentic-research": "/agentic-research",
    "pytorch": "/ml/pytorch",
    "tensorflow": "/ml/tensorflow",
    "knowledge distillation": "/ml/knowledge-distillation",
    "knowledge-distillation": "/ml/knowledge-distillation",
}

def _is_debug_enabled() -> bool:
    # Temporary: always-on debug logging while chart pipeline is being verified.
    # Switch this back to an env-guard once debugging is complete.
    return True


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


class ChatProvider(Protocol):
    """
    Provider contract for AG-UI chat generation.

    Why this exists:
    - Keeps transport/event concerns (`/agui`) separate from model-provider concerns.
    - Makes provider selection (Gemini/OpenAI/Anthropic) an adapter swap instead of
      a route rewrite.

    Contract:
    - Input: normalized chat payload dictionary (history/messages/model).
    - Output: final assistant text to stream into `TEXT_MESSAGE_CONTENT`.
    - Errors: raise exceptions; the AG-UI stream wrapper converts to `RUN_ERROR`.
    """

    def generate(self, payload: dict[str, Any]) -> str:
        ...


@dataclass
class LangChainGeminiProvider:
    """
    Default provider using the existing LangChain/Gemini pipeline.

    This is intentionally thin so replacing it later is low-risk:
    - No AG-UI event logic here.
    - No FastAPI dependency.
    - Just "payload in -> text out".
    """

    def generate(self, payload: dict[str, Any]) -> str:
        return run_chat(payload)


def _format_assistant_text(raw_output: Any) -> str:
    return format_assistant_json_text(raw_output)


def _has_frontend_tool(input_data: RunAgentInput, tool_name: str) -> bool:
    return any(getattr(tool, "name", None) == tool_name for tool in input_data.tools)


def _extract_latest_user_text(input_data: RunAgentInput) -> str:
    for message in reversed(input_data.messages):
        if getattr(message, "role", None) != "user":
            continue
        return extract_text(getattr(message, "content", ""))
    return ""


def _resolve_navigation_target(text: str) -> str | None:
    normalized = text.strip().lower()
    if not normalized:
        return None

    direct = NAV_ROUTE_ALIASES.get(normalized)
    if direct:
        return direct

    patterns = [
        r"(?:navigate|go|open|take me)\s+(?:to\s+)?(.+)$",
        r"(?:can you|please)\s+(?:navigate|go|open)\s+(?:to\s+)?(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        target = match.group(1).strip(" .!?")
        resolved = NAV_ROUTE_ALIASES.get(target)
        if resolved:
            return resolved
        if target.startswith("/") and target in set(NAV_ROUTE_ALIASES.values()):
            return target
    return None


def _normalize_action_calls(raw_actions: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_actions, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        args = item.get("args")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(args, dict):
            args = {}
        normalized.append({"name": name.strip(), "args": args})
    return normalized


def extract_text(content: Any) -> str:
    """
    Normalize AG-UI message content into plain text.

    AG-UI message content can arrive as:
    - plain string
    - list of content blocks (dicts or primitives)
    - null

    This helper ensures provider adapters always receive text content.
    """
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def extract_attachments(content: Any) -> list[dict[str, Any]]:
    """
    Extract attachment-like items from AG-UI multimodal content blocks.

    AG-UI user messages may encode binary payloads inside `content` items:
    - { "type": "binary", "mimeType": "...", "filename": "...", "url": "...", "data": "..." }
    """
    if not isinstance(content, list):
        return []

    attachments: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "binary":
            continue
        attachments.append(
            {
                "type": item.get("mimeType") or item.get("type"),
                "name": item.get("filename") or "attachment",
                "url": item.get("url"),
                "data": item.get("data"),
            }
        )
    return attachments


def build_chat_payload(
    input_data: RunAgentInput,
    requested_model: str | None = None,
) -> dict[str, Any]:
    """
    Translate `RunAgentInput` (AG-UI shape) into provider payload shape.

    Output schema mirrors existing `agent_langchain.run_chat` expectations:
    - `messages`: [{ role, content, attachments }]
    - `model`: optional selected model id (fallback applied here)
    """
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(input_data.messages):
        content = getattr(message, "content", None)
        attachments = extract_attachments(content)
        messages.append(
            {
                "role": message.role,
                "content": extract_text(content),
                "attachments": attachments,
            }
        )
        _debug_log(
            "build_chat_payload.message",
            index=index,
            role=message.role,
            has_content=content is not None,
            attachments_count=len(attachments),
        )
    _debug_log(
        "build_chat_payload.done",
        messages_count=len(messages),
        model=requested_model or DEFAULT_MODEL_ID,
    )
    return {
        "messages": messages,
        "model": requested_model or DEFAULT_MODEL_ID,
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in input_data.tools
        ],
        "context": [
            {
                "description": context.description,
                "value": context.value,
            }
            for context in input_data.context
        ],
    }


async def agui_event_stream(
    payload: dict[str, Any],
    provider: ChatProvider | None = None,
) -> Iterable[str]:
    """
    Build an AG-UI-compliant server-sent event stream.

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
    llm_provider = provider or LangChainGeminiProvider()

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
        if _has_frontend_tool(input_data, "navigate_to_page"):
            nav_route = _resolve_navigation_target(latest_user_text)
            if nav_route:
                tool_call_id = f"tool_{run_id}_navigate_to_page"
                tool_args_json = json.dumps({"route": nav_route}, ensure_ascii=False)
                _debug_log(
                    "stream.fast_path.navigation",
                    run_id=run_id,
                    requested=latest_user_text[:160],
                    resolved_route=nav_route,
                )

                fast_path_events = [
                    ToolCallStartEvent(
                        type=EventType.TOOL_CALL_START,
                        tool_call_id=tool_call_id,
                        tool_call_name="navigate_to_page",
                        parent_message_id=message_id,
                    ),
                    ToolCallArgsEvent(
                        type=EventType.TOOL_CALL_ARGS,
                        tool_call_id=tool_call_id,
                        delta=tool_args_json,
                    ),
                    ToolCallEndEvent(
                        type=EventType.TOOL_CALL_END,
                        tool_call_id=tool_call_id,
                    ),
                    TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        message_id=message_id,
                        role="assistant",
                    ),
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=message_id,
                        delta=json.dumps(
                            {"message": f"Navigating to {nav_route}.", "chartSpec": None, "actions": []},
                            ensure_ascii=False,
                        ),
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
                for event in fast_path_events:
                    _debug_log(
                        "stream.emit",
                        run_id=run_id,
                        event_type=getattr(event, "type", None),
                    )
                    yield encoder.encode(event)
                return

        requested_model = payload.get("model")
        chat_payload = build_chat_payload(input_data, requested_model=requested_model)
        _debug_log(
            "provider.generate.start",
            run_id=run_id,
            thread_id=thread_id,
            provider=llm_provider.__class__.__name__,
        )
        raw_output = llm_provider.generate(chat_payload)
        result_text = _format_assistant_text(raw_output)
        parsed_payload: dict[str, Any] | None = None
        chart_spec: Any = None
        action_calls: list[dict[str, Any]] = []
        try:
            normalized_payload = json.loads(result_text)
            parsed_payload = normalized_payload if isinstance(normalized_payload, dict) else None
            chart_spec = parsed_payload.get("chartSpec") if parsed_payload else None
            action_calls = _normalize_action_calls(parsed_payload.get("actions") if parsed_payload else None)
            if isinstance(chart_spec, list):
                chart_count = len(chart_spec)
                chart_types = [
                    item.get("type")
                    for item in chart_spec
                    if isinstance(item, dict)
                ]
            elif isinstance(chart_spec, dict):
                chart_count = 1
                chart_types = [chart_spec.get("type")]
            else:
                chart_count = 0
                chart_types = []
            _debug_log(
                "provider.generate.normalized_payload",
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
                "provider.generate.normalized_payload_parse_failed",
                run_id=run_id,
                preview=result_text[:300],
            )
        _debug_log(
            "provider.generate.done",
            run_id=run_id,
            output_chars=len(result_text),
        )

        frontend_tool_events: list[Any] = []
        available_tool_names = {getattr(tool, "name", None) for tool in input_data.tools}

        for index, action in enumerate(action_calls):
            action_name = action["name"]
            action_args = action["args"]
            if action_name not in available_tool_names:
                _debug_log(
                    "stream.skip.unknown_action",
                    run_id=run_id,
                    action_name=action_name,
                )
                continue
            tool_call_id = f"tool_{run_id}_{index + 1}_{action_name}"
            action_args_json = json.dumps(action_args, ensure_ascii=False)
            frontend_tool_events.extend(
                [
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
            )
            _debug_log(
                "stream.emit.frontend_tool_call",
                run_id=run_id,
                tool_call_id=tool_call_id,
                tool_name=action_name,
                args_preview=action_args_json[:300],
            )

        emitted_add_chart_action = any(action.get("name") == "add_chart_spec" for action in action_calls)
        if (
            chart_spec is not None
            and _has_frontend_tool(input_data, "add_chart_spec")
            and not emitted_add_chart_action
        ):
            tool_call_id = f"tool_{run_id}_add_chart_spec"
            tool_args = {"chartSpecs": chart_spec} if isinstance(chart_spec, list) else {"chartSpec": chart_spec}
            tool_args_json = json.dumps(tool_args, ensure_ascii=False)
            frontend_tool_events.extend(
                [
                    ToolCallStartEvent(
                        type=EventType.TOOL_CALL_START,
                        tool_call_id=tool_call_id,
                        tool_call_name="add_chart_spec",
                        parent_message_id=message_id,
                    ),
                    ToolCallArgsEvent(
                        type=EventType.TOOL_CALL_ARGS,
                        tool_call_id=tool_call_id,
                        delta=tool_args_json,
                    ),
                    ToolCallEndEvent(
                        type=EventType.TOOL_CALL_END,
                        tool_call_id=tool_call_id,
                    ),
                ]
            )
            _debug_log(
                "stream.emit.frontend_tool_call",
                run_id=run_id,
                tool_call_id=tool_call_id,
                tool_name="add_chart_spec",
                args_preview=tool_args_json[:300],
            )

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
    provider: ChatProvider | None = None,
) -> StreamingResponse:
    """
    FastAPI response factory for AG-UI streams.

    Keeps route handlers simple:
    - `server.py` delegates route payload directly here.
    - feature-specific behavior remains in this module.
    """
    return StreamingResponse(
        agui_event_stream(payload, provider=provider),
        media_type="text/event-stream",
    )
