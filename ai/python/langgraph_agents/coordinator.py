"""
Coordinator flow for Agentic Research.

NOTE:
This project has a local package named `langgraph`, which shadows the external
`langgraph` library import path. Because of that namespace collision, importing
`langgraph.graph` from the external package fails in this environment.

This file implements the same node-style coordinator flow in plain Python:
Data Scientist -> Analyst -> Formatter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from google_gemini import DEFAULT_MODEL_ID
from backend.agents.analyst import interpret_analysis
from backend.agents.data_scientist import run_data_scientist_analysis

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATASETS_MANIFEST_PATH = SAMPLE_DATA_DIR / "datasets.json"


def _load_dataset_metadata(dataset_id: str) -> Dict[str, Any]:
    """
    Load dataset metadata (context + optional names file excerpt) for analyst prompts.
    """
    if not DATASETS_MANIFEST_PATH.exists():
        return {}
    try:
        entries = json.loads(DATASETS_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entry = next((item for item in entries if item.get("id") == dataset_id), None)
    if not entry:
        return {}
    metadata = entry.get("metadata", {}) or {}
    files = metadata.get("files", {}) or {}
    names_path = files.get("names")
    names_excerpt = ""
    if names_path:
        file_path = SAMPLE_DATA_DIR / names_path
        if file_path.exists():
            try:
                names_excerpt = file_path.read_text(encoding="utf-8", errors="ignore")[:2000].strip()
            except OSError:
                names_excerpt = ""
    return {
        "context": metadata.get("context", ""),
        "files": {"names": names_path} if names_path else {},
        "excerpts": {"names": names_excerpt} if names_excerpt else {},
    }


def _normalize_charts(charts: Any) -> list[dict[str, Any]]:
    """Normalize chart payloads to the list shape expected downstream."""
    if isinstance(charts, dict):
        return [charts]
    if isinstance(charts, list):
        return charts
    return []


def _resolve_dataset_label(dataset_id: str, ds_result: Dict[str, Any]) -> str:
    """Prefer the chart-level dataset label when the DS result exposes one."""
    charts = _normalize_charts(ds_result.get("chartSpec"))
    if charts:
        return charts[0].get("meta", {}).get("datasetLabel") or dataset_id
    return dataset_id


def _parse_conversation_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop empty chat messages and normalize role/content history records."""
    conversation_history = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            conversation_history.append({"role": role, "content": content})
    return conversation_history


def _build_langsmith_metadata() -> Dict[str, Any]:
    """Return the temporary disabled LangSmith payload used by the UI."""
    return {
        "enabled": False,
        "note": "Temporarily disabled for performance during Agentic UI iteration.",
    }


def _build_response_payload(
    state: CoordinatorState,
    ds_result: Dict[str, Any],
    analyst: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the frontend response contract from the coordinator state."""
    charts = _normalize_charts(ds_result.get("chartSpec"))
    findings = analyst.get("findings", [])
    langsmith = _build_langsmith_metadata()
    response_message = (
        f"[Data Scientist] {ds_result.get('message', '')}\n\n"
        f"[Analyst] {analyst.get('analyst_summary', '')}\n\n"
        "[LangSmith] disabled (temporarily for performance)."
    )
    return {
        "message": response_message,
        "chartSpec": charts,
        "data": {
            "dataset": state.get("dataset_label", state.get("dataset_id")),
            "tool_summary": ds_result.get("message", ""),
            "langsmith": langsmith,
        },
        "findings": findings if isinstance(findings, list) else [],
        "observability": {"langsmith": langsmith},
    }


class CoordinatorState(TypedDict, total=False):
    """Shared state passed between coordinator nodes."""

    # User prompt text.
    user_message: str
    # Active dataset id from the frontend dropdown.
    dataset_id: str
    # Selected model id (Gemini) used by both agents.
    model_id: str
    # Conversation history from prior messages (for follow-up questions).
    conversation_history: List[Dict[str, Any]]
    # Human-readable dataset label for UI copy.
    dataset_label: str
    # Raw Data Scientist output (charts + summary).
    data_scientist_result: Dict[str, Any]
    # Raw Analyst output (interpretation + findings).
    analyst_result: Dict[str, Any]
    # LangSmith observability payload (trace/run identifiers).
    langsmith: Dict[str, Any]
    # Final payload returned to FastAPI route.
    response: Dict[str, Any]


def _data_scientist_node(state: CoordinatorState) -> CoordinatorState:
    """
    Run Data Scientist analysis using the current user request and dataset.
    """
    # Pull message from state.
    message = state.get("user_message", "")
    # Pull dataset id from state.
    dataset_id = state.get("dataset_id", "")
    # Use explicit model id or default model.
    model_id = state.get("model_id", DEFAULT_MODEL_ID)
    # Pull conversation history for follow-up context.
    conversation_history = state.get("conversation_history", [])
    # Execute the Data Scientist pipeline (tool planning + tool execution + charts).
    ds_result = run_data_scientist_analysis(
        message=message,
        dataset_id=dataset_id,
        model_id=model_id,
        conversation_history=conversation_history,
    )
    # Resolve a display label for downstream messaging.
    dataset_label = _resolve_dataset_label(dataset_id, ds_result)
    # Return updated state snapshot.
    return {
        **state,
        "dataset_label": dataset_label,
        "data_scientist_result": ds_result,
    }


def _analyst_node(state: CoordinatorState) -> CoordinatorState:
    """
    Ask the Analyst agent to interpret Data Scientist outputs.
    """
    # Pull DS result.
    ds_result = state.get("data_scientist_result", {})
    # Normalize chart payload to a list.
    charts = _normalize_charts(ds_result.get("chartSpec"))
    # Run Analyst interpretation with conversation history for follow-ups.
    analyst = interpret_analysis(
        user_request=state.get("user_message", ""),
        dataset_label=state.get("dataset_label", state.get("dataset_id", "")),
        data_scientist_message=ds_result.get("message", ""),
        charts=charts,
        non_chart_response=ds_result.get("nonChartResponse"),
        dataset_metadata=_load_dataset_metadata(state.get("dataset_id", "")),
        conversation_history=state.get("conversation_history", []),
        model_id=state.get("model_id", DEFAULT_MODEL_ID),
    )
    # Return state with analyst output attached.
    return {
        **state,
        "analyst_result": analyst,
    }


def _format_response_node(state: CoordinatorState) -> CoordinatorState:
    """
    Build final frontend response with role-tagged message + charts + findings.
    """
    # Pull DS result.
    ds_result = state.get("data_scientist_result", {})
    # Pull analyst result.
    analyst = state.get("analyst_result", {})
    response = _build_response_payload(state, ds_result, analyst)
    # Return state with final response.
    return {**state, "response": response}


def _coordinator_pipeline(initial_state: CoordinatorState) -> CoordinatorState:
    # Execute deterministic node sequence (LangGraph-style orchestration).
    state_after_ds = _data_scientist_node(initial_state)
    state_after_analyst = _analyst_node(state_after_ds)
    state_with_observability: CoordinatorState = {
        **state_after_analyst,
        "langsmith": _build_langsmith_metadata(),
    }
    return _format_response_node(state_with_observability)


def coordinator_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the coordinator graph and return final structured response.

    Args:
        payload: FastAPI request payload containing user message, dataset id, model,
                 and optional messages array for conversation history.

    Returns:
        Dict with message, chartSpec, findings, and data metadata.
    """
    # Extract conversation history from payload (sent by client for follow-ups).
    messages = payload.get("messages", [])
    conversation_history = _parse_conversation_history(messages)

    # Initialize graph state from request payload.
    initial_state: CoordinatorState = {
        "user_message": payload.get("message", ""),
        "dataset_id": payload.get("dataset_id", ""),
        "model_id": payload.get("model") or DEFAULT_MODEL_ID,
        "conversation_history": conversation_history,
    }
    final_state = _coordinator_pipeline(initial_state)
    # Return only final response payload.
    return final_state.get("response", {})
