from __future__ import annotations

from typing import Any

from agents.coordinator import coordinator_agent
from application.chat.policies import (
    available_tool_names as _available_tool_names,
    extract_latest_user_text as _extract_latest_user_text,
    is_probable_ui_action_request,
    normalize_tool_name as _normalize_tool_name,
    resolve_message as _resolve_message,
)
from application.chat.service import (
    UnifiedActionPlan,
    UnifiedAssistantPayload,
    UnifiedChatMode,
    run_unified_action_plan as _run_unified_action_plan_impl,
    run_unified_chat as _run_unified_chat_impl,
)
from shared.agent_langchain import run_chat
from shared.chartspec import normalize_assistant_payload


def _is_probable_ui_action_request(message: str, tools: Any) -> bool:
    return is_probable_ui_action_request(message, tools)


def run_unified_action_plan(payload: dict[str, Any]) -> UnifiedActionPlan:
    return _run_unified_action_plan_impl(
        payload,
        run_chat_fn=run_chat,
        normalize_assistant_payload_fn=normalize_assistant_payload,
        resolve_message_fn=_resolve_message,
    )


def run_unified_chat(
    payload: dict[str, Any],
    *,
    force_provider: bool = False,
) -> tuple[UnifiedChatMode, UnifiedAssistantPayload]:
    return _run_unified_chat_impl(
        payload,
        force_provider=force_provider,
        run_chat_fn=run_chat,
        coordinator_agent_fn=coordinator_agent,
        normalize_assistant_payload_fn=normalize_assistant_payload,
        resolve_message_fn=_resolve_message,
        is_probable_ui_action_request_fn=is_probable_ui_action_request,
    )
