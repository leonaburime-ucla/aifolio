"""
LangSmith telemetry setup helpers.
"""

from __future__ import annotations

from datetime import datetime
import os
from typing import Any, Dict, Iterable, List

try:
    from langsmith import Client
except Exception:  # pragma: no cover
    Client = None  # type: ignore[assignment]


def configure_langsmith() -> bool:
    """
    Enable LangSmith tracing when credentials are available.
    Returns True if tracing was enabled.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return False

    # Prefer current LangSmith env vars.
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "AIfolio")

    # Backward compatibility for older LangChain tracing configs.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", os.environ.get("LANGSMITH_TRACING", "true"))
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ.get("LANGSMITH_PROJECT", "AIfolio"))
    return True


def _safe_serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_serialize(item) for item in value]
    return value


def _run_to_dict(run: Any) -> Dict[str, Any]:
    if hasattr(run, "model_dump"):
        return _safe_serialize(run.model_dump())
    if hasattr(run, "dict"):
        return _safe_serialize(run.dict())
    if isinstance(run, dict):
        return _safe_serialize(run)
    return _safe_serialize(getattr(run, "__dict__", {}))


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def _latency_ms(start_time: Any, end_time: Any) -> float | None:
    start_dt = _parse_timestamp(start_time)
    end_dt = _parse_timestamp(end_time)
    if start_dt is None or end_dt is None:
        return None
    return round((end_dt - start_dt).total_seconds() * 1000, 3)


def _extract_usage(run_dict: Dict[str, Any]) -> Dict[str, Any]:
    # The SDK/provider can place usage in different fields; preserve anything relevant.
    candidates = [
        run_dict.get("usage_metadata"),
        run_dict.get("token_usage"),
        run_dict.get("usage"),
    ]
    extra = run_dict.get("extra")
    if isinstance(extra, dict):
        candidates.extend(
            [
                extra.get("usage"),
                extra.get("token_usage"),
                extra.get("usage_metadata"),
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    # Backfill common token fields when explicit usage dict is absent.
    fallback = {
        "prompt_tokens": run_dict.get("prompt_tokens"),
        "completion_tokens": run_dict.get("completion_tokens"),
        "total_tokens": run_dict.get("total_tokens"),
    }
    if any(v is not None for v in fallback.values()):
        return {k: v for k, v in fallback.items() if v is not None}
    return {}


def _extract_cost(run_dict: Dict[str, Any]) -> Dict[str, Any]:
    cost: Dict[str, Any] = {}
    for key in (
        "total_cost",
        "prompt_cost",
        "completion_cost",
        "input_cost",
        "output_cost",
        "cost",
    ):
        value = run_dict.get(key)
        if value is not None:
            cost[key] = value
    return cost


def _status_from_run(run_dict: Dict[str, Any]) -> str:
    if run_dict.get("error"):
        return "error"
    if run_dict.get("end_time") is None:
        return "running"
    return "success"


def _collect_retry_events(run_dicts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    retry_events: List[Dict[str, Any]] = []
    for run_dict in run_dicts:
        events = run_dict.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            event_text = " ".join(
                str(event.get(key, "")) for key in ("name", "event", "type", "message")
            ).lower()
            if "retry" not in event_text:
                continue
            retry_events.append(
                {
                    "run_id": run_dict.get("id"),
                    "run_name": run_dict.get("name"),
                    "run_type": run_dict.get("run_type"),
                    "event": event,
                }
            )
    return retry_events


def get_trace_report(
    trace_id: str,
    project_name: str | None = None,
    include_raw: bool = True,
) -> Dict[str, Any]:
    if Client is None:
        return {
            "status": "error",
            "error": "langsmith SDK is not available in this environment.",
            "trace_id": trace_id,
        }
    if not trace_id:
        return {"status": "error", "error": "trace_id is required."}

    client = Client()
    project = project_name or os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    runs_iter = client.list_runs(
        trace_id=trace_id,
        project_name=project,
        limit=100,
    )
    runs = list(runs_iter)
    if not runs:
        return {
            "status": "error",
            "error": "Trace not found or inaccessible in the selected project.",
            "trace_id": trace_id,
            "project": project,
        }

    run_dicts = [_run_to_dict(run) for run in runs]
    root = next((r for r in run_dicts if r.get("parent_run_id") is None), run_dicts[0])
    root_id = root.get("id")
    root_run_obj = next((run for run in runs if str(getattr(run, "id", "")) == str(root_id)), runs[0])
    root_status = _status_from_run(root)
    root_latency_ms = _latency_ms(root.get("start_time"), root.get("end_time"))

    # Child runs sorted by start time for a stable timeline.
    def _sort_key(run_dict: Dict[str, Any]) -> str:
        return str(run_dict.get("start_time") or "")

    child_runs = [r for r in run_dicts if r.get("id") != root_id]
    child_runs.sort(key=_sort_key)

    step_timings: List[Dict[str, Any]] = []
    model_calls: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for run_dict in child_runs:
        status = _status_from_run(run_dict)
        latency = _latency_ms(run_dict.get("start_time"), run_dict.get("end_time"))
        step = {
            "id": run_dict.get("id"),
            "parent_run_id": run_dict.get("parent_run_id"),
            "name": run_dict.get("name"),
            "run_type": run_dict.get("run_type"),
            "status": status,
            "start_time": run_dict.get("start_time"),
            "end_time": run_dict.get("end_time"),
            "latency_ms": latency,
        }
        step_timings.append(step)

        usage = _extract_usage(run_dict)
        cost = _extract_cost(run_dict)
        run_type = str(run_dict.get("run_type") or "")
        if run_type in {"llm", "chat_model"} or "generative" in str(run_dict.get("name", "")).lower():
            model_calls.append(
                {
                    **step,
                    "model_name": run_dict.get("name"),
                    "usage": usage,
                    "cost": cost,
                    "error": run_dict.get("error"),
                    "inputs": run_dict.get("inputs"),
                    "outputs": run_dict.get("outputs"),
                }
            )
        if run_type == "tool":
            tool_calls.append(
                {
                    **step,
                    "tool_name": run_dict.get("name"),
                    "inputs": run_dict.get("inputs"),
                    "outputs": run_dict.get("outputs"),
                    "error": run_dict.get("error"),
                }
            )
        if run_dict.get("error"):
            errors.append(
                {
                    "id": run_dict.get("id"),
                    "name": run_dict.get("name"),
                    "run_type": run_dict.get("run_type"),
                    "error": run_dict.get("error"),
                }
            )

    retry_events = _collect_retry_events(run_dicts)

    # Aggregate tokens/cost where available.
    total_tokens = 0
    total_cost = 0.0
    cost_fields_seen = False
    for call in model_calls:
        usage = call.get("usage") or {}
        tokens_value = usage.get("total_tokens")
        if isinstance(tokens_value, (int, float)):
            total_tokens += int(tokens_value)
        call_cost = call.get("cost") or {}
        for key in ("total_cost", "cost", "output_cost", "completion_cost", "input_cost", "prompt_cost"):
            value = call_cost.get(key)
            if isinstance(value, (int, float)):
                total_cost += float(value)
                cost_fields_seen = True

    overall_cost = _extract_cost(root)
    if not cost_fields_seen and isinstance(overall_cost.get("total_cost"), (int, float)):
        total_cost = float(overall_cost["total_cost"])
        cost_fields_seen = True

    try:
        root_url = client.get_run_url(run=root_run_obj, project_name=project)
    except Exception:
        # SDK signatures vary; reconstruct from known IDs as a fallback.
        org_id = root.get("tenant_id")
        project_id = root.get("session_id")
        if org_id and project_id and root_id:
            root_url = f"https://smith.langchain.com/o/{org_id}/projects/p/{project_id}/r/{root_id}?poll=true"
        else:
            root_url = None

    report: Dict[str, Any] = {
        "status": "ok",
        "trace_id": trace_id,
        "project": project,
        "root": {
            "id": root_id,
            "name": root.get("name"),
            "run_type": root.get("run_type"),
            "status": root_status,
            "start_time": root.get("start_time"),
            "end_time": root.get("end_time"),
            "latency_ms": root_latency_ms,
            "error": root.get("error"),
            "usage": _extract_usage(root),
            "cost": _extract_cost(root),
            "inputs": root.get("inputs"),
            "outputs": root.get("outputs"),
            "url": root_url,
        },
        "summary": {
            "status": "error" if errors else root_status,
            "trace_url": root_url,
            "total_latency_ms": root_latency_ms,
            "step_count": len(step_timings),
            "model_call_count": len(model_calls),
            "tool_call_count": len(tool_calls),
            "error_count": len(errors),
            "retry_count": len(retry_events),
            "total_tokens": total_tokens if total_tokens > 0 else None,
            "total_cost": round(total_cost, 8) if cost_fields_seen else None,
            "step_summary": " -> ".join([str(step.get("name")) for step in step_timings if step.get("name")]),
        },
        "step_timings": step_timings,
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "errors": errors,
        "retries": retry_events,
    }
    if include_raw:
        report["raw"] = {
            "root_run": root,
            "all_runs": run_dicts,
        }
    return report
