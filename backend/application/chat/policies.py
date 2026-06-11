from __future__ import annotations

import re
from typing import Any


def extract_latest_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str):
            return item["content"]
    return ""


def normalize_tool_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def resolve_message(payload: dict[str, Any]) -> str:
    direct_message = str(payload.get("message") or payload.get("prompt") or "").strip()
    return direct_message or extract_latest_user_text(payload.get("messages"))


def available_tool_names(tools: Any) -> set[str]:
    if not isinstance(tools, list):
        return set()
    names = {
        normalize_tool_name(tool.get("name", ""))
        for tool in tools
        if isinstance(tool, dict)
    }
    names.discard("")
    return names


def is_probable_ui_action_request(message: str, tools: Any) -> bool:
    normalized_message = normalize_tool_name(message)
    if not normalized_message:
        return False

    for tool_name in available_tool_names(tools):
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
