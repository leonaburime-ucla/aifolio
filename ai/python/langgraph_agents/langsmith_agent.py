"""
LangSmith Observability Agent.

Interprets LangSmith trace summary data into user-facing observability narration.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from google_gemini import DEFAULT_MODEL_ID, get_model


def _safe_json_parse(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        raw = "\n".join(parts)
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}


def _deterministic_fallback(langsmith: Dict[str, Any]) -> str:
    if not langsmith.get("enabled"):
        return "[LangSmith] Tracing is disabled."

    summary = langsmith.get("summary") if isinstance(langsmith.get("summary"), dict) else {}
    if summary:
        status_text = str(summary.get("status") or "running")
        tokens = summary.get("total_tokens")
        model_calls = summary.get("model_call_count")
        tool_calls = summary.get("tool_call_count")
        error_count = summary.get("error_count")
        retry_count = summary.get("retry_count")
        latency_ms = summary.get("total_latency_ms")
        cost = summary.get("total_cost")

        parts = [f"This request is {status_text}."]
        if tokens is not None:
            parts.append(f"It used {tokens} tokens")
            if model_calls is not None:
                parts[-1] += f" across {model_calls} model call(s)"
            if tool_calls is not None:
                parts[-1] += f" and {tool_calls} tool call(s)"
            parts[-1] += "."
        elif model_calls is not None or tool_calls is not None:
            parts.append(f"It has {model_calls or 0} model call(s) and {tool_calls or 0} tool call(s).")

        if error_count is not None and retry_count is not None:
            if int(error_count) == 0 and int(retry_count) == 0:
                parts.append("There were no errors or retries.")
            else:
                retry_word = "retry" if int(retry_count) == 1 else "retries"
                parts.append(f"There were {error_count} error(s) and {retry_count} {retry_word}.")
        elif error_count is not None:
            parts.append(f"There were {error_count} error(s).")

        if latency_ms is not None:
            parts.append(f"Total latency was {latency_ms} ms.")
        if cost is not None:
            parts.append(f"Estimated cost was {cost}.")

        return "[LangSmith] " + " ".join(parts)

    report_status = langsmith.get("report_status")
    if report_status:
        message = f"[LangSmith] report_status={report_status}"
        report_error = langsmith.get("report_error")
        if report_error:
            message += f" error={report_error}"
        return message

    return "[LangSmith] Tracing is enabled."


def interpret_langsmith_observability(
    langsmith: Dict[str, Any],
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    """
    LLM-based LangSmith narrator. Returns one line:
    - [LangSmith] ...
    """
    summary = langsmith.get("summary") if isinstance(langsmith.get("summary"), dict) else {}
    if not summary:
        return _deterministic_fallback(langsmith)

    prompt = (
        "You are the LangSmith Observability Agent.\n"
        "Given trace summary data, return ONLY valid JSON:\n"
        '{"langsmith_message":"..."}\n'
        "Rules:\n"
        "- langsmith_message must start with '[LangSmith] '.\n"
        "- Keep langsmith_message to 1-2 concise sentences.\n"
        "- Mention status, tokens (if present), model/tool call counts, errors/retries, and latency.\n"
        "- Mention cost only if present.\n"
        "- For no errors/retries, explicitly say 'no errors or retries'.\n"
        "- Do not add a second line.\n\n"
        f"Trace summary:\n{json.dumps(summary, default=str)}\n"
    )

    try:
        llm = get_model(model_id)
        result = llm.invoke([HumanMessage(content=prompt)])
        parsed = _safe_json_parse(getattr(result, "content", ""))
        langsmith_message = str(parsed.get("langsmith_message", "")).strip()
        if not langsmith_message.startswith("[LangSmith]"):
            return _deterministic_fallback(langsmith)
        return langsmith_message
    except Exception:
        return _deterministic_fallback(langsmith)


def summarize_langsmith_observability(langsmith: Dict[str, Any]) -> str:
    """
    Backward-compatible wrapper for existing callers.
    """
    return interpret_langsmith_observability(langsmith=langsmith, model_id=DEFAULT_MODEL_ID)
