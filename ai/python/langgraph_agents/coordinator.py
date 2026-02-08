"""
Coordinator flow for Agentic Research.

NOTE:
This project has a local package named `langgraph`, which shadows the external
`langgraph` library import path. Because of that namespace collision, importing
`langgraph.graph` from the external package fails in this environment.

This file implements the same node-style coordinator flow in plain Python:
Data Scientist -> Researcher -> Formatter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from google_gemini import DEFAULT_MODEL_ID
from langgraph_agents.data_scientist import run_data_scientist_analysis

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"
DATASETS_MANIFEST_PATH = SAMPLE_DATA_DIR / "datasets.json"


def _load_dataset_metadata(dataset_id: str) -> Dict[str, Any]:
    """
    Load dataset metadata (context + optional names file excerpt) for researcher prompts.
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
from langgraph_agents.researcher import interpret_analysis


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
    # Raw Researcher output (interpretation + findings).
    researcher_result: Dict[str, Any]
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
    dataset_label = (
        ds_result.get("chartSpec", [{}])[0].get("meta", {}).get("datasetLabel")
        if ds_result.get("chartSpec")
        else dataset_id
    )
    # Return updated state snapshot.
    return {
        **state,
        "dataset_label": dataset_label,
        "data_scientist_result": ds_result,
    }


def _researcher_node(state: CoordinatorState) -> CoordinatorState:
    """
    Ask the Researcher agent to interpret Data Scientist outputs.
    """
    # Pull DS result.
    ds_result = state.get("data_scientist_result", {})
    # Normalize chart payload to a list.
    charts = ds_result.get("chartSpec") or []
    if isinstance(charts, dict):
        charts = [charts]
    # Run Researcher interpretation with conversation history for follow-ups.
    researcher = interpret_analysis(
        user_request=state.get("user_message", ""),
        dataset_label=state.get("dataset_label", state.get("dataset_id", "")),
        data_scientist_message=ds_result.get("message", ""),
        charts=charts,
        non_chart_response=ds_result.get("nonChartResponse"),
        dataset_metadata=_load_dataset_metadata(state.get("dataset_id", "")),
        conversation_history=state.get("conversation_history", []),
        model_id=state.get("model_id", DEFAULT_MODEL_ID),
    )
    # Return state with researcher output attached.
    return {
        **state,
        "researcher_result": researcher,
    }


def _format_response_node(state: CoordinatorState) -> CoordinatorState:
    """
    Build final frontend response with role-tagged message + charts + findings.
    """
    # Pull DS result.
    ds_result = state.get("data_scientist_result", {})
    # Pull researcher result.
    researcher = state.get("researcher_result", {})
    # Normalize charts to list.
    charts = ds_result.get("chartSpec")
    if isinstance(charts, dict):
        charts = [charts]
    # Pull researcher findings list.
    findings = researcher.get("findings", [])
    # Build chat text with explicit role sections for readability.
    response_message = (
        f"[Data Scientist] {ds_result.get('message', '')}\n\n"
        f"[Researcher] {researcher.get('researcher_summary', '')}\n\n"
        "[LangSmith] TBD"
    )
    # Build structured response contract consumed by frontend.
    response = {
        "message": response_message,
        "chartSpec": charts,
        "data": {
            "dataset": state.get("dataset_label", state.get("dataset_id")),
            "tool_summary": ds_result.get("message", ""),
        },
        "findings": findings,
    }
    # Return state with final response.
    return {**state, "response": response}


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
    conversation_history = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            conversation_history.append({"role": role, "content": content})

    # Initialize graph state from request payload.
    initial_state: CoordinatorState = {
        "user_message": payload.get("message", ""),
        "dataset_id": payload.get("dataset_id", ""),
        "model_id": payload.get("model") or DEFAULT_MODEL_ID,
        "conversation_history": conversation_history,
    }
    # Execute deterministic node sequence (LangGraph-style orchestration).
    state_after_ds = _data_scientist_node(initial_state)
    state_after_researcher = _researcher_node(state_after_ds)
    final_state = _format_response_node(state_after_researcher)
    # Return only final response payload.
    return final_state.get("response", {})
