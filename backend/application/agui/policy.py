from __future__ import annotations

from typing import Any

from shared.agui_runtime.intents import resolve_ag_ui_tab_target
from shared.agui_runtime.ml_actions import normalize_dataset_id_value

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
    "agentic-research": BASE_AG_UI_TOOL_NAMES
    | GLOBAL_CHART_TOOL_NAMES
    | AGENTIC_RESEARCH_TOOL_NAMES,
    "pytorch": BASE_AG_UI_TOOL_NAMES | PYTORCH_TOOL_NAMES,
    "tensorflow": BASE_AG_UI_TOOL_NAMES | TENSORFLOW_TOOL_NAMES,
}
COORDINATOR_ENABLED_TABS: set[str] = {"agentic-research"}
DATASET_SWITCH_TOOL_NAMES: set[str] = {"ar-set_active_dataset", "set_active_dataset"}
ML_ACTION_TABS: set[str] = {"pytorch", "tensorflow"}


def _sanitize_runtime_tools(
    tools: list[dict[str, Any]],
    *,
    active_tab: str,
    latest_user_text: str,
) -> tuple[list[dict[str, Any]], list[str]]:
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


def _filter_messages_for_surface(messages: list[Any], *, active_tab: str) -> list[Any]:
    if active_tab in COORDINATOR_ENABLED_TABS:
        return messages

    for message in reversed(messages):
        if getattr(message, "role", None) == "user":
            return [message]
    return []


def _resolve_surface_dataset_id(context_map: dict[str, str], *, active_tab: str) -> str | None:
    if active_tab not in COORDINATOR_ENABLED_TABS:
        return None
    return context_map.get("agentic_research_selected_dataset_id", "").strip() or None


def _resolve_dataset_from_action_args(action_args: dict[str, Any]) -> str | None:
    for key in ("dataset_id", "datasetId", "dataset", "id"):
        value = action_args.get(key)
        if isinstance(value, str) and value.strip():
            normalized_value = normalize_dataset_id_value(value.strip())
            if isinstance(normalized_value, str) and normalized_value.strip():
                return normalized_value.strip()
    return None


def _resolve_planned_dataset_override(
    planned_actions: list[dict[str, Any]],
    *,
    active_tab: str,
) -> str | None:
    if active_tab not in COORDINATOR_ENABLED_TABS:
        return None
    for action in planned_actions:
        action_name = str(action.get("name") or "").strip()
        if action_name not in DATASET_SWITCH_TOOL_NAMES:
            continue
        action_args = action.get("args")
        if not isinstance(action_args, dict):
            continue
        if dataset_id := _resolve_dataset_from_action_args(action_args):
            return dataset_id
    return None


def _has_agentic_research_analysis_intent(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    analysis_terms = [
        " run ",
        " analyze ",
        " analysis",
        " regression",
        " classification",
        " transform",
        " decomposition",
        " pca",
        " nmf",
        " plsr",
        " lasso",
        " ridge",
        " forest",
        " svm",
        " cluster",
    ]
    padded = f" {normalized} "
    return any(term in padded for term in analysis_terms)


def _summarize_chart_spec(chart_spec: Any) -> tuple[int, list[Any]]:
    if isinstance(chart_spec, list):
        return len(chart_spec), [
            item.get("type") for item in chart_spec if isinstance(item, dict)
        ]
    if isinstance(chart_spec, dict):
        return 1, [chart_spec.get("type")]
    return 0, []


def _filter_non_planned_known_actions(
    action_calls: list[dict[str, Any]],
    planned_actions: list[dict[str, Any]],
    available_tool_names: set[str | None],
) -> list[dict[str, Any]]:
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
