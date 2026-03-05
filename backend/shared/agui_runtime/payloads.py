from __future__ import annotations

"""Payload normalization helpers for AG-UI chat/runtime boundaries.

This module converts AG-UI message/tool/context transport structures into the
normalized payload shape consumed by backend chat providers.
"""

import json
from typing import Any, Callable


def extract_text(content: Any) -> str:
    """Normalize AG-UI message content into plain text.

    Args:
        content: AG-UI content field (string, list blocks, or null).

    Returns:
        Flattened string text representation.
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
    """Extract binary attachment-like items from AG-UI multimodal content blocks.

    Args:
        content: AG-UI content field, usually list of multimodal blocks.

    Returns:
        List of normalized attachment dictionaries.
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


def decode_context_value(raw_value: Any) -> str:
    """Decode Copilot-readable context values serialized via `JSON.stringify`.

    Args:
        raw_value: Raw readable-context value from transport.

    Returns:
        Decoded scalar string value suitable for routing/config lookups.
    """
    text = str(raw_value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except Exception:
        return text

    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, (int, float, bool)):
        return str(parsed)
    return text


def extract_context_map(context_items: list[Any]) -> dict[str, str]:
    """Build a normalized context map keyed by lowercase description.

    Args:
        context_items: AG-UI readable context records.

    Returns:
        Mapping of normalized description -> decoded value.
    """
    context_map: dict[str, str] = {}
    for item in context_items:
        description = str(getattr(item, "description", "") or "").strip().lower()
        value = decode_context_value(getattr(item, "value", ""))
        if not description:
            continue
        context_map[description] = value
    return context_map


def build_chat_payload(
    *,
    messages: list[Any],
    tools: list[Any],
    context: list[Any],
    requested_model: str,
    default_model_id: str,
    debug_log: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Translate AG-UI transport fields into provider payload schema.

    Args:
        messages: AG-UI message history items.
        tools: Frontend tool descriptors for this run.
        context: AG-UI readable context items.
        requested_model: Explicit model request from context/payload.
        default_model_id: Fallback model identifier.
        debug_log: Optional callback for debug instrumentation.

    Returns:
        Provider-compatible payload with normalized messages/tools/context.
    """
    normalized_messages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        content = getattr(message, "content", None)
        attachments = extract_attachments(content)
        normalized_messages.append(
            {
                "role": message.role,
                "content": extract_text(content),
                "attachments": attachments,
            }
        )
        if debug_log is not None:
            debug_log(
                "build_chat_payload.message",
                index=index,
                role=message.role,
                has_content=content is not None,
                attachments_count=len(attachments),
            )

    selected_model = requested_model or default_model_id
    if debug_log is not None:
        debug_log(
            "build_chat_payload.done",
            messages_count=len(normalized_messages),
            model=selected_model,
        )

    return {
        "messages": normalized_messages,
        "model": selected_model,
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ],
        "context": [
            {
                "description": item.description,
                "value": item.value,
            }
            for item in context
        ],
    }
