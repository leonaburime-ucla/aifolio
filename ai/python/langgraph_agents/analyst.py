"""
Analyst Agent Module.

Receives Data Scientist output and produces human-readable interpretations.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from google_gemini import DEFAULT_MODEL_ID, get_model


def _safe_json_parse(raw: Any) -> Dict[str, Any]:
    """
    Parse possibly messy LLM output into a JSON object.

    This helper handles:
    - None values
    - Gemini list-style content payloads
    - Markdown code fences
    - Partial JSON embedded inside surrounding text

    Args:
        raw: Raw LLM response content (string, list, dict-like text, or None).

    Returns:
        A parsed JSON dictionary, or an empty dict when parsing fails.
    """
    # If nothing was returned, fail safely with an empty object.
    if raw is None:
        return {}
    # Gemini can return content as a list of parts; flatten into one string.
    if isinstance(raw, list):
        # Collect text fragments from each content part.
        parts = []
        # Iterate each content part.
        for item in raw:
            # If item is a dict-like part, prefer text/content fields.
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            # Otherwise coerce the part directly to string.
            else:
                parts.append(str(item))
        # Join all fragments into one parseable text blob.
        raw = "\n".join(parts)
    # Normalize to stripped string.
    text = str(raw).strip()
    # Remove code fence wrappers if present.
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    # First attempt: parse as plain JSON object.
    try:
        return json.loads(text)
    # Fallback: try extracting the first {...} region and parse that.
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        # If we cannot find a valid object range, fail safely.
        if start == -1 or end == -1 or end <= start:
            return {}
        # Parse extracted object slice.
        try:
            return json.loads(text[start : end + 1])
        # If all parsing attempts fail, return empty dict.
        except json.JSONDecodeError:
            return {}


def _format_conversation_history(history: List[Dict[str, Any]]) -> str:
    """
    Format conversation history for inclusion in prompt.
    """
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for msg in history[-6:]:  # Keep last 6 messages to avoid token bloat
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        # Truncate long messages
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def interpret_analysis(
    user_request: str,
    dataset_label: str,
    data_scientist_message: str,
    charts: List[Dict[str, Any]],
    non_chart_response: Dict[str, Any] | None = None,
    dataset_metadata: Dict[str, Any] | None = None,
    conversation_history: List[Dict[str, Any]] | None = None,
    model_id: str = DEFAULT_MODEL_ID,
) -> Dict[str, Any]:
    """
    Ask the Analyst agent to interpret analysis results for the user.

    Args:
        user_request: Current user question/request.
        dataset_label: Human-readable dataset name.
        data_scientist_message: Summary from Data Scientist agent.
        charts: Chart specs with data points.
        non_chart_response: Additional metrics/notes.
        dataset_metadata: Dataset context info.
        conversation_history: Prior messages for follow-up context.
        model_id: LLM model to use.

    Returns:
        Dict with analyst_summary and findings list.
    """
    # Format conversation history if present (for follow-up questions)
    history_text = _format_conversation_history(conversation_history or [])

    # Build the prompt with optional conversation context
    prompt_parts = [
        "You are a data science analyst. Interpret the analysis results below.\n",
        "Return ONLY valid JSON:\n"
        '{"analyst_summary": "...", "findings": ["...", "..."]}\n\n'
        "RULES:\n"
        "- Cite specific numeric values from the data (loadings, coefficients, R², etc.)\n"
        "- Explain what each number means practically for the target variable\n"
        "- For loadings/coefficients: positive = positive relationship, negative = inverse\n"
        "- Larger absolute values = stronger influence\n"
        "- NO generic statements like 'several features influence the outcome'\n"
    ]

    # Add conversation history for follow-up context
    if history_text:
        prompt_parts.append(
            "- If this is a follow-up question, reference the prior conversation context\n"
            "- Answer the user's specific question based on prior analysis results\n\n"
            f"{history_text}\n\n"
        )
    else:
        prompt_parts.append("\n")

    prompt_parts.extend([
        f"User request: {user_request}\n"
        f"Dataset: {dataset_label}\n"
        f"Data Scientist summary: {data_scientist_message}\n\n"
        f"Chart data:\n{json.dumps(charts, indent=2, default=str)}\n\n"
        f"Metrics: {json.dumps(non_chart_response, default=str)}\n"
        f"Dataset context: {json.dumps(dataset_metadata, default=str)}\n"
    ])

    prompt = "".join(prompt_parts)

    llm = get_model(model_id)
    result = llm.invoke([HumanMessage(content=prompt)])
    parsed = _safe_json_parse(getattr(result, "content", ""))

    return {
        "analyst_summary": parsed.get("analyst_summary", "Could not parse response."),
        "findings": parsed.get("findings", []),
    }
