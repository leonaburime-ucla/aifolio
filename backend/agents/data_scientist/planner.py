"""Planner helpers for data scientist tool selection and repair."""

from __future__ import annotations

import json
from typing import Any, Callable


PlannerInvoker = Callable[[str], Any]
Parser = Callable[[Any], dict[str, Any]]


def format_conversation_history(history: list[dict[str, Any]]) -> str:
    """Format recent conversation turns for planner context."""
    if not history:
        return ""
    lines = []
    for message in history[-4:]:
        role = message.get("role", "user").upper()
        content = str(message.get("content", ""))
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def build_plan_prompt(
    message: str,
    tools_schema: list[dict[str, Any]],
    dataset: dict[str, Any],
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    """Build the planning prompt for tool selection."""
    tool_names = [tool["name"] for tool in tools_schema]
    tools_catalog = [
        {
            "name": tool["name"],
            "doc": (tool.get("doc") or "")[:160],
            "params": [param["name"] for param in tool.get("params", [])],
        }
        for tool in tools_schema
    ]
    synonym_map = {
        "pca": "pca_transform",
        "principal component analysis": "pca_transform",
        "svd": "truncated_svd",
        "truncated svd": "truncated_svd",
        "ica": "fast_ica",
        "independent component analysis": "fast_ica",
        "nmf": "nmf_decomposition",
        "non-negative matrix factorization": "nmf_decomposition",
        "non negative matrix factorization": "nmf_decomposition",
        "plsr": "pls_regression",
        "pls": "pls_regression",
        "partial least squares": "pls_regression",
        "linear regression": "linear_regression",
        "ridge": "ridge_regression",
        "lasso": "lasso_regression",
        "elastic net": "elasticnet_regression",
        "random forest": "random_forest_regression",
        "gradient boosting": "gradient_boosting_regression",
        "logistic regression": "logistic_regression",
    }
    history_text = format_conversation_history(conversation_history or [])
    history_context = ""
    if history_text:
        history_context = (
            "Previous conversation:\n"
            f"{history_text}\n\n"
            "IMPORTANT: If the user is asking a follow-up question about a previous analysis "
            "(e.g., 'what does that mean?', 'explain the alcohol loading'), "
            "return an empty tool_calls array and set summary to describe the prior analysis context. "
            "The Analyst will handle interpretation questions.\n\n"
        )

    return (
        "You are a data scientist. Decide which sklearn tools to run based on the request.\n"
        "Return ONLY valid JSON with this shape:\n"
        "{\n"
        '  "summary": string,\n'
        '  "tool_calls": [\n'
        "    {\n"
        '      "tool_name": string,\n'
        '      "tool_args": object,\n'
        '      "chart_kind": "pca"|"plsr"|"regression"|"none"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"{history_context}"
        "Rules:\n"
        "- tool_name MUST be one of the available tool names listed below.\n"
        "- If the user uses a synonym (e.g., PCA/PLSR), map it to the exact tool name.\n"
        "- If multiple analyses are requested, return multiple tool_calls.\n"
        "- For follow-up/clarification questions, return empty tool_calls and let the Analyst handle it.\n\n"
        "Synonyms (map to tool_name):\n"
        f"{json.dumps(synonym_map, default=str)}\n\n"
        "Dataset info:\n"
        f"- Columns: {', '.join(dataset.get('columns', []))}\n"
        f"- Target column: {dataset.get('targetColumn')}\n"
        f"- Task: {dataset.get('task')}\n\n"
        "Available tools:\n"
        f"{json.dumps(tools_catalog, default=str)}\n\n"
        f"Tool names: {', '.join(tool_names)}\n\n"
        f"User request: {message}\n"
    )


def build_repair_prompt(
    raw_plan: dict[str, Any],
    message: str,
    tool_names: list[str],
) -> str:
    """Build the prompt used to repair invalid tool names."""
    return (
        "Your previous plan contained invalid tool names.\n"
        "Return ONLY valid JSON with the same shape and valid tool_name values.\n"
        f"Valid tool names: {', '.join(tool_names)}\n"
        f"Previous plan: {json.dumps(raw_plan, default=str)}\n"
        f"User request: {message}\n"
    )


def _planner_error(stage: str, raw: Any, previous_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    error = {"stage": stage, "raw": raw}
    if previous_plan is not None:
        error["previous_plan"] = previous_plan
    return error


def parse_planner_response(
    raw_content: Any,
    parser: Parser,
    *,
    failure_summary: str,
    stage: str,
    previous_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse planner output into a stable tool plan envelope."""
    parsed = parser(raw_content)
    if parsed and "tool_calls" in parsed:
        return parsed
    return {
        "summary": failure_summary,
        "tool_calls": [],
        "planner_error": _planner_error(stage, raw_content, previous_plan),
    }


def validate_tool_plan(plan: dict[str, Any], tool_names: list[str]) -> dict[str, Any]:
    """Drop tool calls whose names are not in the allowed list."""
    tool_calls = plan.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return {"summary": "Invalid tool plan.", "tool_calls": []}
    cleaned = []
    for call in tool_calls:
        if call.get("tool_name") in tool_names:
            cleaned.append(call)
    return {**plan, "tool_calls": cleaned}


def _plan_needs_repair(raw_plan: dict[str, Any], validated_plan: dict[str, Any]) -> bool:
    raw_calls = raw_plan.get("tool_calls")
    validated_calls = validated_plan.get("tool_calls")
    return isinstance(raw_calls, list) and isinstance(validated_calls, list) and len(validated_calls) < len(raw_calls)


def resolve_tool_plan(
    *,
    message: str,
    tools_schema: list[dict[str, Any]],
    dataset: dict[str, Any],
    conversation_history: list[dict[str, Any]] | None,
    invoke_prompt: PlannerInvoker,
    parser: Parser,
) -> dict[str, Any]:
    """Plan tool calls and repair invalid names when necessary."""
    tool_names = [tool["name"] for tool in tools_schema]
    raw_plan = parse_planner_response(
        invoke_prompt(build_plan_prompt(message, tools_schema, dataset, conversation_history)),
        parser,
        failure_summary="Failed to plan tools.",
        stage="plan",
    )
    validated_plan = validate_tool_plan(raw_plan, tool_names)
    if not _plan_needs_repair(raw_plan, validated_plan):
        return validated_plan

    repaired_plan = parse_planner_response(
        invoke_prompt(build_repair_prompt(raw_plan, message, tool_names)),
        parser,
        failure_summary="Failed to repair tool plan.",
        stage="repair",
        previous_plan=raw_plan,
    )
    repaired_validated = validate_tool_plan(repaired_plan, tool_names)
    if repaired_validated.get("tool_calls"):
        return repaired_validated
    if repaired_plan.get("planner_error"):
        return {**repaired_validated, "planner_error": repaired_plan["planner_error"]}
    return validated_plan
