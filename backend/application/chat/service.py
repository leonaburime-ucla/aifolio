from __future__ import annotations

from typing import Any, Literal, TypedDict
from agents.coordinator import coordinator_agent
from shared.agent_langchain import run_chat
from shared.chartspec import normalize_assistant_payload
from application.chat.policies import (
    available_tool_names,
    extract_latest_user_text,
    is_probable_ui_action_request,
    normalize_tool_name,
    resolve_message,
)


class UnifiedAssistantPayload(TypedDict):
    message: str
    chartSpec: Any
    actions: list[dict[str, Any]]


UnifiedChatMode = Literal["coordinator", "provider"]


class UnifiedActionPlan(TypedDict):
    actions: list[dict[str, Any]]
    planner_message: str

def _normalize_action_plan_payload(normalized: dict[str, Any]) -> UnifiedActionPlan:
    actions = normalized.get("actions")
    return {
        "actions": actions if isinstance(actions, list) else [],
        "planner_message": str(normalized.get("message") or "").strip(),
    }


def run_unified_action_plan(
    payload: dict[str, Any],
    *,
    run_chat_fn=run_chat,
    normalize_assistant_payload_fn=normalize_assistant_payload,
    resolve_message_fn=resolve_message,
) -> UnifiedActionPlan:
    """
    Fast planning pass that asks the provider to return only frontend actions.

    This intentionally avoids coordinator execution so UI tool calls can be emitted
    early and serially before any expensive synthesis work.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return {"actions": [], "planner_message": ""}

    message = resolve_message_fn(payload)
    raw_output = run_chat_fn(
        {**payload, "message": message, "response_mode": "actions_only"}
    )
    normalized = normalize_assistant_payload_fn(raw_output)
    return _normalize_action_plan_payload(normalized)


def run_unified_chat(
    payload: dict[str, Any],
    *,
    force_provider: bool = False,
    run_chat_fn=run_chat,
    coordinator_agent_fn=coordinator_agent,
    normalize_assistant_payload_fn=normalize_assistant_payload,
    resolve_message_fn=resolve_message,
    is_probable_ui_action_request_fn=is_probable_ui_action_request,
) -> tuple[UnifiedChatMode, UnifiedAssistantPayload]:
    """
    Run the shared backend chat pipeline used by both regular chat routes and AG-UI.

    Routing policy:
    - If dataset-aware request (`dataset_id` + user text): use coordinator_agent
      (Data Scientist + Analyst stack).
    - Otherwise: use generic provider chat (`run_chat`) and normalize to assistant payload.
    """
    message = resolve_message_fn(payload)

    dataset_id = str(payload.get("dataset_id") or "").strip()
    tools = payload.get("tools")
    ui_action_request = is_probable_ui_action_request_fn(message, tools)
    force_coordinator = bool(payload.get("_force_coordinator"))

    if dataset_id and message and (force_coordinator or (not ui_action_request and not force_provider)):
        coordinator_payload = {
            **payload,
            "message": message,
            "dataset_id": dataset_id,
        }
        result = coordinator_agent_fn(coordinator_payload)
        return (
            "coordinator",
            {
                "message": str(result.get("message") or ""),
                "chartSpec": result.get("chartSpec"),
                "actions": [],
            },
        )

    raw_output = run_chat_fn({**payload, "message": message})
    normalized = normalize_assistant_payload_fn(raw_output)
    return ("provider", normalized)
