from __future__ import annotations

import json
import re
from typing import Any

ALLOWED_CHART_TYPES = {
    "line",
    "area",
    "bar",
    "scatter",
    "histogram",
    "density",
    "roc",
    "pr",
    "errorbar",
    "heatmap",
    "box",
    "violin",
    "biplot",
    "dendrogram",
    "surface",
}


def extract_text_from_llm_output(output: Any) -> str:
    """
    Normalize provider output into a plain text string.

    Handles common Gemini/LangChain shapes:
    - plain string
    - list of blocks: [{type: "text", text: "..."}]
    - fallback to `str(output)`
    """
    if isinstance(output, str):
        return output

    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)

    return str(output)


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_assistant_json(text: str) -> dict[str, Any] | None:
    """
    Parse expected assistant JSON payload:
    { "message": string, "chartSpec": ChartSpec | ChartSpec[] | null }
    """
    raw = strip_code_fences(text)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
            return parsed
    except Exception:
        pass

    # Fallback: try first JSON object span if model wrapped text around it.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
        if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
            return parsed
    except Exception:
        return None
    return None


def _coerce_chart_data_rows(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def normalize_chart_spec(raw: Any, index: int = 0) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    chart_type = raw.get("type")
    x_key = raw.get("xKey")
    y_keys = raw.get("yKeys")
    data = _coerce_chart_data_rows(raw.get("data"))

    if not isinstance(chart_type, str) or chart_type not in ALLOWED_CHART_TYPES:
        return None
    if not isinstance(x_key, str) or not x_key.strip():
        return None
    if not isinstance(y_keys, list):
        return None

    cleaned_y_keys = [k for k in y_keys if isinstance(k, str) and k.strip()]
    if not cleaned_y_keys:
        return None
    if not data:
        return None

    chart_id = raw.get("id")
    if not isinstance(chart_id, str) or not chart_id.strip():
        chart_id = f"chart_{index + 1}"

    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        title = f"{chart_type.title()} chart"

    normalized: dict[str, Any] = {
        "id": chart_id,
        "title": title,
        "type": chart_type,
        "xKey": x_key,
        "yKeys": cleaned_y_keys,
        "data": data,
    }

    optional_string_fields = [
        "description",
        "xLabel",
        "yLabel",
        "zKey",
        "colorKey",
        "unit",
        "currency",
    ]
    for key in optional_string_fields:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value

    error_keys = raw.get("errorKeys")
    if isinstance(error_keys, dict):
        cleaned_error_keys = {
            str(k): str(v)
            for k, v in error_keys.items()
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
        }
        if cleaned_error_keys:
            normalized["errorKeys"] = cleaned_error_keys

    timeframe = raw.get("timeframe")
    if isinstance(timeframe, dict):
        start = timeframe.get("start")
        end = timeframe.get("end")
        if isinstance(start, str) and isinstance(end, str) and start.strip() and end.strip():
            normalized["timeframe"] = {"start": start, "end": end}

    source = raw.get("source")
    if isinstance(source, dict):
        provider = source.get("provider")
        url = source.get("url")
        if isinstance(provider, str) and provider.strip():
            normalized["source"] = {"provider": provider}
            if isinstance(url, str) and url.strip():
                normalized["source"]["url"] = url

    meta = raw.get("meta")
    if isinstance(meta, dict):
        normalized_meta: dict[str, Any] = {}
        dataset_label = meta.get("datasetLabel")
        query_time_ms = meta.get("queryTimeMs")
        if isinstance(dataset_label, str) and dataset_label.strip():
            normalized_meta["datasetLabel"] = dataset_label
        if isinstance(query_time_ms, (int, float)):
            normalized_meta["queryTimeMs"] = query_time_ms
        if normalized_meta:
            normalized["meta"] = normalized_meta

    return normalized


def normalize_chart_spec_payload(raw: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
    if raw is None:
        return None

    if isinstance(raw, list):
        specs: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            normalized = normalize_chart_spec(item, index=index)
            if normalized:
                specs.append(normalized)
        return specs if specs else None

    normalized = normalize_chart_spec(raw, index=0)
    return normalized


def normalize_actions_payload(raw: Any) -> list[dict[str, Any]]:
    """
    Normalize optional frontend action intents returned by the model.

    Expected item shape:
    { "name": string, "args": object }
    """
    if not isinstance(raw, list):
        return []

    actions: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        args = item.get("args")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(args, dict):
            args = {}
        actions.append({"name": name.strip(), "args": args})
    return actions


def normalize_assistant_payload(raw_output: Any) -> dict[str, Any]:
    text = extract_text_from_llm_output(raw_output)
    parsed = parse_assistant_json(text)
    if not parsed:
        return {"message": text.strip(), "chartSpec": None, "actions": []}

    message = parsed.get("message")
    chart_spec = normalize_chart_spec_payload(parsed.get("chartSpec"))
    actions = normalize_actions_payload(parsed.get("actions"))
    if not isinstance(message, str):
        message = text.strip()
    return {"message": message.strip(), "chartSpec": chart_spec, "actions": actions}


def format_assistant_json_text(raw_output: Any) -> str:
    """
    Convert provider output into strict assistant JSON text.

    Output shape is always:
    { "message": string, "chartSpec": ChartSpec | ChartSpec[] | null }
    """
    normalized = normalize_assistant_payload(raw_output)
    return json.dumps(normalized, ensure_ascii=False)
