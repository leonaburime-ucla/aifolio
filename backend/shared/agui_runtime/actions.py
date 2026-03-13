from __future__ import annotations

"""Action normalization and policy helpers for AG-UI tool calls.

This module centralizes deterministic action filtering/enforcement so AG-UI
tool-call behavior remains consistent across planner/generator outputs.
"""

import json
import re
from typing import Any, Callable

from .intents import resolve_ag_ui_tab_target
from .ml_actions import normalize_ml_tab_actions

PYTORCH_EXECUTION_TOOL_NAMES: set[str] = {
    "start_pytorch_training_runs",
    "train_pytorch_model",
}


def normalize_action_calls(raw_actions: Any) -> list[dict[str, Any]]:
    """Normalize raw action payload into validated `{name, args}` entries.

    Args:
        raw_actions: Untrusted model/planner output.

    Returns:
        List of normalized action dictionaries. Invalid entries are skipped.
    """
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


def _has_phrase(text: str, pattern: str) -> bool:
    """Return whether `pattern` regex is present in `text`."""
    return re.search(pattern, text) is not None




def detect_pytorch_training_intent(text: str) -> tuple[bool, bool, bool]:
    """Detect train/sweep/autodistill intent flags from user text.

    Args:
        text: Latest user message text.

    Returns:
        Tuple of `(wants_train, wants_run_sweep, wants_auto_distill)`.
    """
    normalized_text = text.strip().lower()
    if not normalized_text:
        return (False, False, False)

    wants_train = any(
        _has_phrase(normalized_text, pattern)
        for pattern in [
            r"\btrain\b",
            r"\bstart\b.*\b(training|runs?)\b",
            r"\brun\b.*\b(training|runs?)\b",
        ]
    )
    wants_run_sweep = _has_phrase(normalized_text, r"\brun\s+sweep\b")
    wants_auto_distill = _has_phrase(normalized_text, r"\b(auto\s*-?\s*distill|autodistill)\b")
    return (wants_train, wants_run_sweep, wants_auto_distill)


def build_enforced_pytorch_actions(
    latest_user_text: str,
    has_tool_messages_after_latest_user: bool,
    action_calls: list[dict[str, Any]],
    available_tool_names: set[str],
    debug_log: Callable[..., None] | None = None,
) -> list[dict[str, Any]]:
    """Enforce deterministic ordered PyTorch action plans for compound intents.

    Args:
        latest_user_text: Latest user prompt text.
        has_tool_messages_after_latest_user: Whether transport already emitted
            tool messages after latest user turn.
        action_calls: Candidate actions from planner/model output.
        available_tool_names: Tools exposed by the frontend in current run.
        debug_log: Optional logger callback for policy traces.

    Returns:
        Ordered, filtered, de-duplicated action list safe for AG-UI emission.
    """
    if not latest_user_text.strip():
        return action_calls

    wants_train, wants_run_sweep, wants_auto_distill = detect_pytorch_training_intent(latest_user_text)

    if not (wants_train or wants_run_sweep or wants_auto_distill):
        filtered = [
            item
            for item in action_calls
            if str(item.get("name") or "").strip() not in PYTORCH_EXECUTION_TOOL_NAMES
        ]
        if len(filtered) != len(action_calls) and debug_log is not None:
            debug_log(
                "stream.actions.training_filtered",
                reason="no_training_intent",
                before=len(action_calls),
                after=len(filtered),
                removed=[item.get("name") for item in action_calls if item not in filtered],
            )
        return filtered

    if has_tool_messages_after_latest_user:
        return action_calls

    planned: list[dict[str, Any]] = []

    def _already_has(name: str) -> bool:
        return any(item.get("name") == name for item in planned)

    for item in action_calls:
        name = str(item.get("name") or "").strip()
        if not name or name not in available_tool_names:
            continue
        args = item.get("args")
        if not isinstance(args, dict):
            args = {}
        planned.append({"name": name, "args": args})

    tab_target = resolve_ag_ui_tab_target(latest_user_text)
    if tab_target and "switch_ag_ui_tab" in available_tool_names:
        has_switch_action = any(item.get("name") == "switch_ag_ui_tab" for item in planned)
        if not has_switch_action:
            planned.insert(0, {"name": "switch_ag_ui_tab", "args": {"tab": tab_target}})

    if (wants_run_sweep or wants_auto_distill) and "set_pytorch_form_fields" in available_tool_names:
        has_form_patch = any(item.get("name") == "set_pytorch_form_fields" for item in planned)
        if not has_form_patch:
            fields: dict[str, Any] = {}
            if wants_run_sweep:
                fields["run_sweep"] = True
            if wants_auto_distill:
                fields["auto_distill"] = True
            if fields:
                planned.insert(0, {"name": "set_pytorch_form_fields", "args": {"fields": fields}})

    if wants_train and "start_pytorch_training_runs" in available_tool_names and not _already_has("start_pytorch_training_runs"):
        planned.append({"name": "start_pytorch_training_runs", "args": {}})

    if wants_train and not wants_run_sweep and "set_pytorch_form_fields" in available_tool_names:
        configured = False
        for item in planned:
            if item.get("name") != "set_pytorch_form_fields":
                continue
            args = item.get("args")
            if not isinstance(args, dict):
                args = {}
                item["args"] = args
            fields = args.get("fields")
            if not isinstance(fields, dict):
                fields = {}
                args["fields"] = fields
            fields["run_sweep"] = False
            configured = True
            break
        if not configured:
            planned.insert(0, {"name": "set_pytorch_form_fields", "args": {"fields": {"run_sweep": False}}})

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in planned:
        key = json.dumps(
            {"name": item.get("name"), "args": item.get("args", {})},
            sort_keys=True,
            ensure_ascii=False,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped
