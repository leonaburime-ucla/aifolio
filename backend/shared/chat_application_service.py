from __future__ import annotations

from typing import Any, Literal, TypedDict
import re

from shared.agent_langchain import run_chat
from agents.coordinator import coordinator_agent
from shared.chartspec import normalize_assistant_payload


class UnifiedAssistantPayload(TypedDict):
    message: str
    chartSpec: Any
    actions: list[dict[str, Any]]


UnifiedChatMode = Literal["coordinator", "provider"]


class UnifiedActionPlan(TypedDict):
    actions: list[dict[str, Any]]
    planner_message: str


def _extract_latest_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str):
            return item["content"]
    return ""


def _normalize_tool_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _resolve_message(payload: dict[str, Any]) -> str:
    direct_message = str(payload.get("message") or payload.get("prompt") or "").strip()
    return direct_message or _extract_latest_user_text(payload.get("messages"))


def _available_tool_names(tools: Any) -> set[str]:
    if not isinstance(tools, list):
        return set()
    names = {_normalize_tool_name(tool.get("name", "")) for tool in tools if isinstance(tool, dict)}
    names.discard("")
    return names


def _normalize_action_plan_payload(normalized: dict[str, Any]) -> UnifiedActionPlan:
    actions = normalized.get("actions")
    return {
        "actions": actions if isinstance(actions, list) else [],
        "planner_message": str(normalized.get("message") or "").strip(),
    }


def is_probable_ui_action_request(message: str, tools: Any) -> bool:
    """
    Heuristic gate: when a prompt looks like a frontend action intent,
    keep the provider path so it can emit `actions` for AG-UI tool calls.
    """
    normalized_message = _normalize_tool_name(message)
    if not normalized_message:
        return False

    for tool_name in _available_tool_names(tools):
        if tool_name and tool_name in normalized_message:
            return True

    ui_action_patterns = [
        r"\b(navigate|go|open|take me)\b",
        r"\b(switch|change)\b.*\b(tab|page|workspace)\b",
        r"\b(switch|change|open|go)\b.*\b(agentic research|agentic-research|research|pytorch|tensorflow|charts?)\b",
        r"\bclear\b.*\b(chart|charts)\b",
        r"\b(add|create|show|plot|draw|render)\b.*\b(chart|graph)\b",
        r"\btrain\b.*\b(pytorch|tensorflow|model)\b",
        r"\b(start|run)\b.*\b(training|train|sweep|runs?)\b",
        r"\b(set|update|change|randomize)\b.*\b(hidden|layers?|dims?|dropout|learning rate|batch|epoch|target|columns?|sweep|auto distill|autodistill)\b",
        r"\b(set|select|choose|change|switch|use)\b.*\b(dataset)\b",
        r"\bdataset\b.*\b(to|as)\b",
        r"\b(remove|delete)\b.*\b(chart)\b",
        r"\breorder\b.*\b(chart|charts)\b",
    ]
    return any(re.search(pattern, normalized_message) for pattern in ui_action_patterns)


def _is_probable_ui_action_request(message: str, tools: Any) -> bool:
    """Backward-compatible wrapper for legacy tests/imports."""
    return is_probable_ui_action_request(message, tools)


def run_unified_action_plan(payload: dict[str, Any]) -> UnifiedActionPlan:
    """
    Fast planning pass that asks the provider to return only frontend actions.

    This intentionally avoids coordinator execution so UI tool calls can be emitted
    early and serially before any expensive synthesis work.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return {"actions": [], "planner_message": ""}

    message = _resolve_message(payload)
    raw_output = run_chat({**payload, "message": message, "response_mode": "actions_only"})
    normalized = normalize_assistant_payload(raw_output)
    return _normalize_action_plan_payload(normalized)


def run_unified_chat(
    payload: dict[str, Any],
    *,
    force_provider: bool = False,
) -> tuple[UnifiedChatMode, UnifiedAssistantPayload]:
    """
    Run the shared backend chat pipeline used by both regular chat routes and AG-UI.

    Routing policy:
    - If dataset-aware request (`dataset_id` + user text): use coordinator_agent
      (Data Scientist + Analyst stack).
    - Otherwise: use generic provider chat (`run_chat`) and normalize to assistant payload.
    """
    message = _resolve_message(payload)

    dataset_id = str(payload.get("dataset_id") or "").strip()
    tools = payload.get("tools")
    ui_action_request = is_probable_ui_action_request(message, tools)
    force_coordinator = bool(payload.get("_force_coordinator"))

    if dataset_id and message and (force_coordinator or (not ui_action_request and not force_provider)):
        coordinator_payload = {
            **payload,
            "message": message,
            "dataset_id": dataset_id,
        }
        result = coordinator_agent(coordinator_payload)
        return (
            "coordinator",
            {
                "message": str(result.get("message") or ""),
                "chartSpec": result.get("chartSpec"),
                "actions": [],
            },
        )

    raw_output = run_chat({**payload, "message": message})
    normalized = normalize_assistant_payload(raw_output)
    return ("provider", normalized)
