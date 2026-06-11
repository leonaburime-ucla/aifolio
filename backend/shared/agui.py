"""Compatibility facade for AG-UI streaming orchestration.

The real AG-UI application service lives under `application.agui`. This module
keeps the legacy import path and monkeypatch surface stable for tests and
older callers.
"""

from __future__ import annotations

from application.agui.policy import (
    AGENTIC_RESEARCH_TOOL_NAMES,
    BASE_AG_UI_TOOL_NAMES,
    CHART_RENDER_TOOL_NAMES,
    CONTEXT_BOUND_TOOL_NAMES,
    COORDINATOR_ENABLED_TABS,
    DATASET_SWITCH_TOOL_NAMES,
    GLOBAL_CHART_TOOL_NAMES,
    ML_ACTION_TABS,
    PYTORCH_TOOL_NAMES,
    TAB_SCOPED_TOOL_NAMES,
    TENSORFLOW_TOOL_NAMES,
    _filter_messages_for_surface,
    _filter_non_planned_known_actions,
    _has_agentic_research_analysis_intent,
    _resolve_dataset_from_action_args,
    _resolve_planned_dataset_override,
    _resolve_surface_dataset_id,
    _sanitize_runtime_tools,
    _summarize_chart_spec,
)
from application.agui.service import (
    _build_enforced_pytorch_actions,
    _build_tool_call_events,
    _debug_log,
    _decode_context_value,
    _extract_context_map,
    _extract_latest_user_text,
    _format_assistant_text,
    _has_frontend_tool,
    _has_tool_messages_after_latest_user,
    _is_debug_enabled,
    _is_pure_tab_switch_intent,
    _normalize_action_calls,
    _normalize_ml_tab_actions,
    _resolve_ag_ui_tab_target,
    _resolve_navigation_target,
    agui_event_stream as _agui_event_stream_impl,
    build_chat_payload,
    create_agui_stream_response as _create_agui_stream_response_impl,
    extract_attachments,
    extract_text,
)
from shared.chat_application_service import (
    is_probable_ui_action_request,
    run_unified_action_plan,
    run_unified_chat,
)


def agui_event_stream(payload: dict):
    return _agui_event_stream_impl(
        payload,
        is_probable_ui_action_request_fn=is_probable_ui_action_request,
        run_unified_action_plan_fn=run_unified_action_plan,
        run_unified_chat_fn=run_unified_chat,
    )


def create_agui_stream_response(payload: dict):
    return _create_agui_stream_response_impl(
        payload,
        is_probable_ui_action_request_fn=is_probable_ui_action_request,
        run_unified_action_plan_fn=run_unified_action_plan,
        run_unified_chat_fn=run_unified_chat,
    )
