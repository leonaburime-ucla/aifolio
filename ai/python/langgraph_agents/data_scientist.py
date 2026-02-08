"""
Data Scientist Agent Module.

Overall job:
- Load dataset metadata and rows from backend sample data storage.
- Plan sklearn tool execution from natural-language user requests.
- Execute selected tools and convert outputs into frontend-ready chart specs.
- Return structured analysis payloads for the coordinator and chat routes.
"""

from __future__ import annotations
import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, TypedDict

from pathlib import Path
import csv
import openpyxl
import xlrd

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from tools import sklearn_tools

from google_gemini import DEFAULT_MODEL_ID


# 1) Define State
class AgentState(TypedDict):
    messages: List[BaseMessage]


DEFAULT_PCA_MESSAGE = (
    "Run PCA on the provided dataset and return the transformed values, "
    "components, and explained variance ratio."
)

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def build_gemini_agent(
    model_id: str = DEFAULT_MODEL_ID,
    tools: List[Callable] | None = None,
) -> ChatGoogleGenerativeAI:
    """
    Create the Gemini-backed data scientist agent.
    """
    llm = ChatGoogleGenerativeAI(model=model_id, temperature=0.3)
    if tools:
        llm = llm.bind_tools(tools)
    return llm


def get_sklearn_tools() -> List[Callable]:
    """
    Expose a single router tool that can invoke any sklearn tool by name.
    """
    return [sklearn_tool_router]


def sklearn_tool_router(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route a tool call to any sklearn tool by name.

    Args:
        tool_name: The sklearn_tools function name to invoke.
        tool_args: Arguments to forward to the tool function.

    Returns:
        The tool's output dict, or an error payload if the tool fails.
    """
    available = set(sklearn_tools.list_available_tools())
    if tool_name not in available:
        return {"status": "error", "error": f"Unknown tool: {tool_name}"}

    tool_fn = getattr(sklearn_tools, tool_name, None)
    if tool_fn is None:
        return {"status": "error", "error": f"Tool not found: {tool_name}"}

    try:
        result = tool_fn(**tool_args)
        return {"status": "ok", "result": result}
    except NotImplementedError as exc:
        return {"status": "error", "error": str(exc), "tool": tool_name}
    except Exception as exc:  # noqa: BLE001 - surface tool errors to the caller
        return {"status": "error", "error": str(exc), "tool": tool_name}


def get_tools_catalog() -> Dict[str, Any]:
    """
    Provide a catalog of available sklearn tools for the LLM prompt.
    """
    return {
        "tool_router": "sklearn_tool_router",
        "available_tools": sklearn_tools.list_available_tools(),
        "tool_schemas": sklearn_tools.get_tools_schema(),
    }


_DS_AGENT_CACHE: dict[str, ChatGoogleGenerativeAI] = {}


def get_data_scientist_agent(model_id: str = DEFAULT_MODEL_ID) -> ChatGoogleGenerativeAI:
    """
    Build the DS agent once and reuse it across requests especially 
    when the model id changes.
    """
    global _DS_AGENT_CACHE
    if model_id not in _DS_AGENT_CACHE:
        _DS_AGENT_CACHE[model_id] = build_gemini_agent(
            model_id=model_id,
            tools=get_sklearn_tools(),
        )
    return _DS_AGENT_CACHE[model_id]


def data_scientist_agent():
    """
    Placeholder constructor for the Data Scientist agent.
    This is where sklearn / torch / tensorflow tools will be wired later.
    """
    raise NotImplementedError("Data scientist agent not implemented yet.")


def run_data_scientist(message: str, model_id: str = DEFAULT_MODEL_ID) -> str:
    """
    Minimal callable for the DS agent to test the route.
    """
    llm = get_data_scientist_agent(model_id=model_id)
    result = llm.invoke([HumanMessage(content=message)])
    return result.content


def get_demo_pca_payload() -> Dict[str, Any]:
    """
    Provide deterministic demo data for PCA testing via the API.
    """
    data = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 4.0, 3.0],
        [3.0, 4.0, 1.0, 2.0],
        [4.0, 3.0, 2.0, 1.0],
        [5.0, 6.0, 2.0, 3.0],
        [6.0, 5.0, 3.0, 2.0],
    ]
    feature_names = ["feature_a", "feature_b", "feature_c", "feature_d"]
    return {"message": DEFAULT_PCA_MESSAGE, "data": data, "feature_names": feature_names}


def run_demo_pca_transform(n_components: int | None = 2) -> Dict[str, Any]:
    """
    Run PCA against deterministic demo data using the LLM tool call.
    """
    payload = get_demo_pca_payload()
    tool_args = {
        "data": payload["data"],
        "n_components": n_components,
        "feature_names": payload["feature_names"],
    }
    return run_data_scientist_tool(
        message=payload["message"],
        tool_name="pca_transform",
        tool_args=tool_args,
    )


def _extract_tool_calls(result: Any) -> List[Dict[str, Any]]:
    """
    Normalize provider-specific tool call payloads.

    Args:
        result: LLM response object.

    Returns:
        List of tool call dictionaries, or an empty list.
    """
    # Preferred path: explicit tool_calls attribute.
    tool_calls = getattr(result, "tool_calls", None)
    if tool_calls:
        return tool_calls
    # Fallback path: nested additional_kwargs payload.
    additional = getattr(result, "additional_kwargs", {}) or {}
    return additional.get("tool_calls", []) or []


def _normalize_tool_args(raw_args: Any) -> Dict[str, Any]:
    """
    Normalize model-generated tool arguments into a dict.

    Args:
        raw_args: Raw args object from tool call payload.

    Returns:
        Dict of parsed args, or {'_raw': ...} when parsing fails.
    """
    # Already a dict: pass through.
    if isinstance(raw_args, dict):
        return raw_args
    # JSON string: parse if possible.
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {"_raw": raw_args}
    # Unknown type: preserve raw for debugging.
    return {"_raw": raw_args}


def run_data_scientist_tool(
    message: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    model_id: str = DEFAULT_MODEL_ID,
) -> Dict[str, Any]:
    """
    Ask the LLM to invoke a tool, then execute it locally and return the result.
    """
    llm = get_data_scientist_agent(model_id=model_id)
    tools_catalog = get_tools_catalog()
    prompt = (
        f"{message}\n\n"
        "You can use sklearn tools via the router.\n"
        f"Tool router: {tools_catalog['tool_router']}\n"
        f"Available tools: {', '.join(tools_catalog['available_tools'])}\n"
        "Tool schemas (name, params, doc):\n"
        f"{json.dumps(tools_catalog['tool_schemas'], default=str)}\n\n"
        "Use the tool `sklearn_tool_router` with the following JSON args exactly:\n"
        f"{json.dumps({'tool_name': tool_name, 'tool_args': tool_args})}\n"
        "Return only the tool call."
    )
    # Ask the model to emit a tool call (prefer the router so we can reach any tool).
    result = llm.invoke([HumanMessage(content=prompt)])
    # Pull tool calls from the model response payload (implementation varies by provider).
    tool_calls = _extract_tool_calls(result)

    # Find a call for the router or for the specific tool if the model ignored the router.
    selected = None
    for call in tool_calls:
        name = call.get("name") or call.get("tool") or ""
        if name in {"sklearn_tool_router", tool_name}:
            selected = call
            break

    # If the model didn't emit the expected call, return the raw response for debugging.
    if not selected:
        return {
            "message": message,
            "tool": tool_name,
            "tool_args": tool_args,
            "status": "error",
            "error": "LLM did not emit the expected tool call.",
            "raw_response": getattr(result, "content", None),
            "tool_calls": tool_calls,
        }

    # Normalize the tool arguments, which may arrive as JSON or a dict.
    raw_args = selected.get("args") or selected.get("arguments") or {}
    normalized_args = _normalize_tool_args(raw_args)

    # If the router was used, unpack its schema and route internally.
    if (selected.get("name") or selected.get("tool")) == "sklearn_tool_router":
        routed_name = normalized_args.get("tool_name", tool_name)
        routed_args = normalized_args.get("tool_args", tool_args)
        tool_output = sklearn_tool_router(routed_name, routed_args)
        return {
            "message": message,
            "tool": routed_name,
            "tool_args": routed_args,
            "status": tool_output.get("status", "ok"),
            "result": tool_output.get("result"),
            "error": tool_output.get("error"),
            "tool_calls": tool_calls,
        }

    # Otherwise, execute the named tool directly.
    tool_fn = getattr(sklearn_tools, tool_name)
    tool_result = tool_fn(**normalized_args)

    return {
        "message": message,
        "tool": tool_name,
        "tool_args": normalized_args,
        "status": "ok",
        "result": tool_result,
        "tool_calls": tool_calls,
    }


def _load_manifest() -> List[Dict[str, Any]]:
    """
    Load dataset manifest JSON from sample_data directory.

    Returns:
        List of dataset manifest entries.
    """
    manifest_path = SAMPLE_DATA_DIR / "datasets.json"
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_dataset_entry(dataset_id: str) -> Optional[Dict[str, Any]]:
    """
    Resolve a dataset manifest entry by id.

    Args:
        dataset_id: Dataset identifier from frontend selection.

    Returns:
        Dataset manifest entry or None when not found.
    """
    manifest = _load_manifest()
    for entry in manifest:
        if entry.get("id") == dataset_id:
            return entry
    return None


def list_sample_datasets() -> List[Dict[str, Any]]:
    """
    Return the dataset manifest used by the agentic research UI.
    """
    return _load_manifest()


def _detect_delimiter(handle) -> str:
    """
    Detect likely CSV delimiter using a small text sample.

    Args:
        handle: Open text file handle positioned at start.

    Returns:
        Delimiter character (comma, semicolon, tab, or comma fallback).
    """
    sample = handle.read(2048)
    handle.seek(0)
    for delimiter in [",", ";", "\t"]:
        if sample.count(delimiter) > 0:
            return delimiter
    return ","


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """
    Read CSV rows into dictionaries with normalized headers.

    Args:
        path: CSV file path.

    Returns:
        List of row dictionaries.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter=_detect_delimiter(handle))
        rows = []
        for row in reader:
            rows.append({k.strip("\ufeff").strip(): v for k, v in row.items()})
        return rows


def _read_xlsx_rows(path: Path) -> List[Dict[str, Any]]:
    """
    Read XLSX rows from first worksheet.

    Args:
        path: XLSX file path.

    Returns:
        List of row dictionaries.
    """
    workbook = openpyxl.load_workbook(path, read_only=True)
    sheet = workbook.active
    rows_iter = sheet.iter_rows(values_only=True)
    headers = next(rows_iter, None)
    if not headers:
        return []
    cleaned_headers = [str(header).strip("\ufeff").strip() for header in headers]
    rows = []
    for row in rows_iter:
        rows.append(
            {
                cleaned_headers[index]: value
                for index, value in enumerate(row)
                if cleaned_headers[index]
            }
        )
    return rows


def _read_xls_rows(path: Path) -> List[Dict[str, Any]]:
    """
    Read legacy XLS rows from first worksheet.

    Args:
        path: XLS file path.

    Returns:
        List of row dictionaries.
    """
    workbook = xlrd.open_workbook(path.as_posix())
    sheet = workbook.sheet_by_index(0)
    headers = [str(value).strip("\ufeff").strip() for value in sheet.row_values(0)]
    rows = []
    for row_index in range(1, sheet.nrows):
        row_values = sheet.row_values(row_index)
        rows.append(
            {
                headers[col_index]: row_values[col_index]
                for col_index in range(len(headers))
                if headers[col_index]
            }
        )
    return rows


def _load_dataset_rows(file_path: Path) -> List[Dict[str, Any]]:
    """
    Dispatch file loading by extension.

    Args:
        file_path: Dataset file path.

    Returns:
        List of row dictionaries.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_rows(file_path)
    if suffix == ".xlsx":
        return _read_xlsx_rows(file_path)
    if suffix == ".xls":
        return _read_xls_rows(file_path)
    raise ValueError("Unsupported file format.")


def load_sample_dataset(dataset_id: str) -> Dict[str, Any]:
    """
    Load a dataset entry and rows by ID.
    Returns dict with dataset, columns, and rows.
    """
    entry = _resolve_dataset_entry(dataset_id)
    if not entry:
        return {"status": "error", "error": "Dataset not found."}
    metadata_files = entry.get("metadata", {}).get("files", {})
    data_path = metadata_files.get("data") or entry.get("files", {}).get("data")
    if not data_path:
        return {"status": "error", "error": "Dataset file missing."}
    file_path = SAMPLE_DATA_DIR / data_path
    if not file_path.exists():
        return {"status": "error", "error": "Dataset file not found."}
    rows = _load_dataset_rows(file_path)
    columns = list(rows[0].keys()) if rows else []
    return {"status": "ok", "dataset": entry, "columns": columns, "rows": rows}


def _coerce_float(value: Any) -> Optional[float]:
    """
    Convert mixed value types to float when possible.

    Args:
        value: Raw cell value.

    Returns:
        Float value or None when conversion fails.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _build_numeric_matrix(
    rows: List[Dict[str, Any]],
    target_column: Optional[str] = None,
) -> tuple[list[list[float]], list[float], list[str]]:
    """
    Build numeric feature matrix and target vector from row dictionaries.

    Args:
        rows: Parsed dataset rows.
        target_column: Optional target column name.

    Returns:
        Tuple of (feature_matrix, targets, feature_names).
    """
    if not rows:
        return [], [], []
    keys = list(rows[0].keys())
    feature_names = [key for key in keys if key != target_column]
    matrix: list[list[float]] = []
    targets: list[float] = []
    raw_targets: list[Any] = []
    for row in rows:
        values: list[float] = []
        valid = True
        for key in feature_names:
            value = _coerce_float(row.get(key))
            if value is None:
                valid = False
                break
            values.append(value)
        if not valid:
            continue
        matrix.append(values)
        if target_column:
            raw_targets.append(row.get(target_column))
    if not target_column:
        return matrix, targets, feature_names

    numeric_targets = []
    non_numeric_targets = []
    for value in raw_targets:
        num = _coerce_float(value)
        if num is None:
            non_numeric_targets.append(str(value).strip())
            numeric_targets.append(None)
        else:
            numeric_targets.append(num)

    if non_numeric_targets:
        labels = sorted(set(non_numeric_targets))
        label_to_index = {label: float(index) for index, label in enumerate(labels)}
        for idx, value in enumerate(raw_targets):
            num = numeric_targets[idx]
            if num is not None:
                targets.append(num)
                continue
            text = str(value).strip()
            if text in label_to_index:
                targets.append(label_to_index[text])
        return matrix[: len(targets)], targets, feature_names

    targets = [float(value) for value in numeric_targets if value is not None]
    return matrix, targets, feature_names


def _build_loadings_chart(
    title: str,
    components: Sequence[Sequence[float]],
    feature_names: Sequence[str],
    description: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert component/loadings arrays into scatter chart spec.

    Args:
        title: Chart title.
        components: Per-feature component coordinates.
        feature_names: Feature labels.
        description: Optional chart subtitle.

    Returns:
        ChartSpec-like dict or None when inputs are empty.
    """
    if not components or not feature_names:
        return None
    points = []
    for idx, name in enumerate(feature_names):
        values = components[idx] if idx < len(components) else []
        points.append(
            {
                "id": f"loading-{idx + 1}",
                "feature": name,
                "pc1": values[0] if len(values) > 0 else 0,
                "pc2": values[1] if len(values) > 1 else 0,
            }
        )
    chart = {
        "id": f"agentic-research-{title.lower().replace(' ', '-')}",
        "title": title,
        "type": "scatter",
        "xKey": "pc1",
        "yKeys": ["pc2"],
        "xLabel": "PC1",
        "yLabel": "PC2",
        "data": points,
    }
    if description:
        chart["description"] = description
    return chart


def _build_regression_chart(predictions: Sequence[float], targets: Sequence[float], title: str):
    points = []
    for idx, pred in enumerate(predictions):
        actual = targets[idx] if idx < len(targets) else None
        if actual is None:
            continue
        points.append({"id": f"pred-{idx + 1}", "actual": actual, "predicted": pred})
    return {
        "id": "agentic-research-regression",
        "title": title,
        "type": "scatter",
        "xKey": "actual",
        "yKeys": ["predicted"],
        "xLabel": "Actual",
        "yLabel": "Predicted",
        "data": points,
    }


def _transpose_components(components: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Transpose component matrix so rows map to features.
    """
    if not components:
        return []
    return [list(col) for col in zip(*components)]


def _build_feature_importance_chart(
    title: str,
    feature_names: Sequence[str],
    values: Sequence[float],
    description: Optional[str] = None,
):
    """
    Build bar chart spec for coefficients/importances.
    """
    if not feature_names or not values:
        return None
    points = []
    for idx, name in enumerate(feature_names):
        value = values[idx] if idx < len(values) else 0
        points.append({"feature": name, "importance": value})
    chart = {
        "id": f"agentic-research-{title.lower().replace(' ', '-')}",
        "title": title,
        "type": "bar",
        "xKey": "feature",
        "yKeys": ["importance"],
        "xLabel": "Feature",
        "yLabel": "Importance",
        "data": points,
    }
    if description:
        chart["description"] = description
    return chart


def _build_embedding_chart(
    title: str,
    embedding: Sequence[Sequence[float]],
    labels: Optional[Sequence[int]] = None,
    description: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build scatter chart for 2D embeddings (t-SNE, etc.).
    """
    if not embedding:
        return None
    points = []
    for idx, coords in enumerate(embedding):
        point = {
            "id": f"sample-{idx + 1}",
            "x": coords[0] if len(coords) > 0 else 0,
            "y": coords[1] if len(coords) > 1 else 0,
        }
        if labels is not None and idx < len(labels):
            point["cluster"] = labels[idx]
        points.append(point)
    chart = {
        "id": f"agentic-research-{title.lower().replace(' ', '-')}",
        "title": title,
        "type": "scatter",
        "xKey": "x",
        "yKeys": ["y"],
        "xLabel": "Dimension 1",
        "yLabel": "Dimension 2",
        "data": points,
    }
    if description:
        chart["description"] = description
    return chart


def _build_cluster_distribution_chart(
    title: str,
    labels: Sequence[int],
    description: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build bar chart showing cluster size distribution.
    """
    if not labels:
        return None
    from collections import Counter
    counts = Counter(labels)
    points = []
    for cluster_id in sorted(counts.keys()):
        label = f"Cluster {cluster_id}" if cluster_id >= 0 else "Noise"
        points.append({"cluster": label, "count": counts[cluster_id]})
    chart = {
        "id": f"agentic-research-{title.lower().replace(' ', '-')}",
        "title": title,
        "type": "bar",
        "xKey": "cluster",
        "yKeys": ["count"],
        "xLabel": "Cluster",
        "yLabel": "Sample Count",
        "data": points,
    }
    if description:
        chart["description"] = description
    return chart


def _parse_json_response(raw: Any) -> Dict[str, Any]:
    """
    Parse LLM text/list output into JSON object.
    """
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
    cleaned = str(raw).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {}


def _format_ds_conversation_history(history: List[Dict[str, Any]]) -> str:
    """Format conversation history for Data Scientist planner context."""
    if not history:
        return ""
    lines = []
    for msg in history[-4:]:  # Keep last 4 messages
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _plan_tools_with_llm(
    message: str,
    tools_schema: List[Dict[str, Any]],
    dataset: Dict[str, Any],
    model_id: str,
    conversation_history: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Ask LLM to choose one or more sklearn tools for a user request.

    Args:
        message: User request text.
        tools_schema: Available sklearn tools + params.
        dataset: Dataset context (columns, target, task).
        model_id: Gemini model id.
        conversation_history: Prior messages for follow-up context.

    Returns:
        Planner payload with summary/tool_calls and optional planner_error.
    """
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

    # Format conversation history if present
    history_text = _format_ds_conversation_history(conversation_history or [])
    history_context = ""
    if history_text:
        history_context = (
            "Previous conversation:\n"
            f"{history_text}\n\n"
            "IMPORTANT: If the user is asking a follow-up question about a previous analysis "
            "(e.g., 'what does that mean?', 'explain the alcohol loading'), "
            "return an empty tool_calls array and set summary to describe the prior analysis context. "
            "The Researcher will handle interpretation questions.\n\n"
        )

    prompt = (
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
        "- For follow-up/clarification questions, return empty tool_calls and let the Researcher handle it.\n\n"
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
    llm = get_data_scientist_agent(model_id=model_id)
    result = llm.invoke([HumanMessage(content=prompt)])
    raw_content = getattr(result, "content", "")
    parsed = _parse_json_response(raw_content)
    if not parsed or "tool_calls" not in parsed:
        return {
            "summary": "Failed to plan tools.",
            "tool_calls": [],
            "planner_error": {
                "stage": "plan",
                "raw": raw_content,
            },
        }
    return parsed


def _validate_tool_plan(plan: Dict[str, Any], tool_names: List[str]) -> Dict[str, Any]:
    """
    Drop invalid tool calls not present in allowed tool list.
    """
    tool_calls = plan.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return {"summary": "Invalid tool plan.", "tool_calls": []}
    cleaned = []
    for call in tool_calls:
        name = call.get("tool_name")
        if name not in tool_names:
            continue
        cleaned.append(call)
    return {**plan, "tool_calls": cleaned}


def _repair_tool_plan(
    raw_plan: Dict[str, Any],
    message: str,
    tools_schema: List[Dict[str, Any]],
    model_id: str,
) -> Dict[str, Any]:
    """
    Ask LLM to repair an invalid tool plan to allowed tool names.
    """
    tool_names = [tool["name"] for tool in tools_schema]
    prompt = (
        "Your previous plan contained invalid tool names.\n"
        "Return ONLY valid JSON with the same shape and valid tool_name values.\n"
        f"Valid tool names: {', '.join(tool_names)}\n"
        f"Previous plan: {json.dumps(raw_plan, default=str)}\n"
        f"User request: {message}\n"
    )
    llm = get_data_scientist_agent(model_id=model_id)
    result = llm.invoke([HumanMessage(content=prompt)])
    raw_content = getattr(result, "content", "")
    parsed = _parse_json_response(raw_content)
    if not parsed or "tool_calls" not in parsed:
        return {
            "summary": "Failed to repair tool plan.",
            "tool_calls": [],
            "planner_error": {
                "stage": "repair",
                "raw": raw_content,
                "previous_plan": raw_plan,
            },
        }
    return parsed


def run_data_scientist_analysis(
    message: str,
    dataset_id: str,
    model_id: str = DEFAULT_MODEL_ID,
    conversation_history: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    End-to-end analysis pipeline for a dataset request.

    Steps:
    1) Load dataset + build numeric matrix.
    2) Plan tool calls with LLM (with conversation context for follow-ups).
    3) Validate/repair plan; fallback when needed.
    4) Execute tools and map outputs to chart specs.
    5) Attach chart metadata and return structured response.

    Args:
        message: User request text.
        dataset_id: Active dataset id selected in UI.
        model_id: Gemini model id.
        conversation_history: Prior messages for follow-up context.

    Returns:
        Dict with message and chartSpec (single or list).
    """
    started_at = time.perf_counter()
    print(f"[ds] analysis:start dataset_id={dataset_id} model={model_id}")
    entry = _resolve_dataset_entry(dataset_id)
    if not entry:
        return {"message": "Dataset not found.", "chartSpec": None}
    data_path = entry.get("files", {}).get("data")
    if not data_path:
        return {"message": "Dataset file missing.", "chartSpec": None}
    file_path = SAMPLE_DATA_DIR / data_path
    rows = _load_dataset_rows(file_path)
    columns = list(rows[0].keys()) if rows else []
    print(
        f"[ds] dataset:loaded dataset_id={dataset_id} rows={len(rows)} cols={len(columns)} "
        f"elapsed_ms={(time.perf_counter() - started_at) * 1000:.1f}"
    )
    target_column = entry.get("targetColumn")
    data, targets, feature_names = _build_numeric_matrix(
        rows, target_column=target_column
    )
    print(
        f"[ds] matrix:built dataset_id={dataset_id} matrix_rows={len(data)} "
        f"features={len(feature_names)} targets={len(targets)} "
        f"elapsed_ms={(time.perf_counter() - started_at) * 1000:.1f}"
    )

    tools_schema = sklearn_tools.get_tools_schema()
    plan_started = time.perf_counter()
    print(f"[ds] planner:start dataset_id={dataset_id}")
    plan = _plan_tools_with_llm(
        message,
        tools_schema,
        {
            "columns": columns,
            "targetColumn": target_column,
            "task": entry.get("task"),
        },
        model_id=model_id,
        conversation_history=conversation_history,
    )
    print(
        f"[ds] planner:end dataset_id={dataset_id} tool_calls={len(plan.get('tool_calls', []))} "
        f"elapsed_ms={(time.perf_counter() - plan_started) * 1000:.1f}"
    )
    plan = _validate_tool_plan(plan, [tool["name"] for tool in tools_schema])
    tool_calls = plan.get("tool_calls", [])
    if not tool_calls:
        fallback_map = {
            "pca": "pca_transform",
            "principal component": "pca_transform",
            "svd": "truncated_svd",
            "truncated svd": "truncated_svd",
            "ica": "fast_ica",
            "independent component": "fast_ica",
            "nmf": "nmf_decomposition",
            "non-negative matrix": "nmf_decomposition",
            "t-sne": "tsne_embedding",
            "tsne": "tsne_embedding",
            "pls": "pls_regression",
            "plsr": "pls_regression",
            "ridge": "ridge_regression",
            "lasso": "lasso_regression",
            "elastic": "elasticnet_regression",
            "svr": "svr_regression",
            "random forest": "random_forest_regression",
            "gradient boosting": "gradient_boosting_regression",
            "logistic": "logistic_regression",
            "regression": "linear_regression",
            "kmeans": "kmeans_clustering",
            "k-means": "kmeans_clustering",
            "dbscan": "dbscan_clustering",
            "agglomerative": "agglomerative_clustering",
            "hierarchical": "agglomerative_clustering",
            "gaussian mixture": "gaussian_mixture_clustering",
            "gmm": "gaussian_mixture_clustering",
            "knn": "knn_classification",
            "naive bayes": "naive_bayes_classification",
        }
        lower = message.lower()
        fallback_tools = []
        for key, tool in fallback_map.items():
            if key in lower:
                fallback_tools.append(tool)
        fallback_tools = list(dict.fromkeys(fallback_tools))
        if fallback_tools:
            tool_calls = []
            for fallback_tool in fallback_tools:
                chart_kind = "regression"
                if fallback_tool in {"pca_transform", "truncated_svd", "fast_ica", "nmf_decomposition"}:
                    chart_kind = "pca"
                elif fallback_tool == "pls_regression":
                    chart_kind = "plsr"
                elif fallback_tool == "tsne_embedding":
                    chart_kind = "embedding"
                elif "clustering" in fallback_tool:
                    chart_kind = "clustering"
                elif "classification" in fallback_tool:
                    chart_kind = "classification"
                tool_calls.append(
                    {
                        "tool_name": fallback_tool,
                        "tool_args": {},
                        "chart_kind": chart_kind,
                    }
                )
        else:
            return {
                "message": plan.get("summary", "Tell me which analysis to run."),
                "chartSpec": None,
                "debug": plan.get("planner_error"),
            }

    charts = []
    notes = []
    metrics: List[str] = []
    tool_chart_map = {
        "pca_transform": "loadings_scatter",
        "incremental_pca": "loadings_scatter",
        "truncated_svd": "loadings_scatter",
        "fast_ica": "loadings_scatter",
        "nmf_decomposition": "loadings_scatter",
        "tsne_embedding": "embedding_scatter",
        "pls_regression": "loadings_scatter",
        "linear_regression": "coefficients_bar",
        "ridge_regression": "coefficients_bar",
        "lasso_regression": "coefficients_bar",
        "elasticnet_regression": "coefficients_bar",
        "svr_regression": "actual_vs_predicted",
        "random_forest_regression": "feature_importance_bar",
        "gradient_boosting_regression": "feature_importance_bar",
        "logistic_regression": "coefficients_bar",
        "random_forest_classification": "feature_importance_bar",
        "gradient_boosting_classification": "feature_importance_bar",
        "knn_classification": "cluster_scatter",
        "naive_bayes_classification": "cluster_scatter",
        "kmeans_clustering": "cluster_distribution",
        "minibatch_kmeans_clustering": "cluster_distribution",
        "dbscan_clustering": "cluster_distribution",
        "agglomerative_clustering": "cluster_distribution",
        "spectral_clustering": "cluster_distribution",
        "gaussian_mixture_clustering": "cluster_distribution",
        "optics_clustering": "cluster_distribution",
    }
    for call in tool_calls:
        tool_name = call.get("tool_name")
        tool_args = call.get("tool_args") or {}
        chart_kind = call.get("chart_kind")
        alias_map = {
            "pca": "pca_transform",
            "plsr": "pls_regression",
            "pls": "pls_regression",
            "svd": "truncated_svd",
            "ica": "fast_ica",
            "nmf": "nmf_decomposition",
            "tsne": "tsne_embedding",
            "t-sne": "tsne_embedding",
            "kmeans": "kmeans_clustering",
            "k-means": "kmeans_clustering",
            "dbscan": "dbscan_clustering",
            "agglomerative": "agglomerative_clustering",
            "spectral": "spectral_clustering",
            "gmm": "gaussian_mixture_clustering",
            "gaussian mixture": "gaussian_mixture_clustering",
            "optics": "optics_clustering",
        }
        if tool_name in alias_map:
            tool_name = alias_map[tool_name]
        if not tool_name and chart_kind in {"pca", "plsr"}:
            tool_name = "pca_transform" if chart_kind == "pca" else "pls_regression"
        if not tool_name:
            continue
        tool_started = time.perf_counter()
        print(f"[ds] tool:start name={tool_name} chart_kind={chart_kind}")

        if tool_name in {"pca_transform", "incremental_pca", "truncated_svd"}:
            tool_args = {
                "data": data,
                "n_components": tool_args.get("n_components", 3),
                "feature_names": feature_names,
            }
            if tool_name == "truncated_svd":
                result = sklearn_tools.truncated_svd(**tool_args)
            elif tool_name == "incremental_pca":
                result = sklearn_tools.incremental_pca(**tool_args)
            else:
                result = sklearn_tools.pca_transform(**tool_args)
            components = result.get("components", [])
            variance = result.get("explained_variance_ratio") or []
            variance_text = None
            if len(variance) >= 2:
                variance_text = (
                    f"Explained variance: PC1 {variance[0] * 100:.1f}%, "
                    f"PC2 {variance[1] * 100:.1f}%"
                )
            loadings = _transpose_components(components)
            chart = _build_loadings_chart(
                "PCA Loadings" if tool_name == "pca_transform" else "SVD Loadings",
                loadings,
                feature_names,
                variance_text,
            )
            if "r2_score" in result:
                metrics.append(f"{tool_name} r2: {result.get('r2_score')}")
            if chart:
                chart["meta"] = {**(chart.get("meta") or {}), "chartKind": tool_chart_map.get(tool_name)}
                charts.append(chart)
            notes.append("PCA complete.")
            print(
                f"[ds] tool:end name={tool_name} status=ok charts={len(charts)} "
                f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
            )
            continue

        if tool_name == "pls_regression":
            if not targets:
                notes.append("Target column is missing for PLSR.")
                continue
            tool_args = {
                "data": data,
                "target": targets,
                "n_components": tool_args.get("n_components", 2),
                "feature_names": feature_names,
            }
            result = sklearn_tools.pls_regression(**tool_args)
            loadings = result.get("x_loadings") or []
            variance = result.get("explained_variance_ratio_x") or []
            variance_text = None
            if len(variance) >= 2:
                variance_text = (
                    f"Explained variance (X): PLS1 {variance[0] * 100:.1f}%, "
                    f"PLS2 {variance[1] * 100:.1f}%"
                )
            chart = _build_loadings_chart(
                "PLSR Loadings",
                loadings,
                feature_names,
                variance_text,
            )
            if chart:
                chart["meta"] = {**(chart.get("meta") or {}), "chartKind": tool_chart_map.get(tool_name)}
                charts.append(chart)
            notes.append("PLSR complete.")
            print(
                f"[ds] tool:end name={tool_name} status=ok charts={len(charts)} "
                f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
            )
            continue

        regression_tools = {
            "linear_regression": sklearn_tools.linear_regression,
            "ridge_regression": sklearn_tools.ridge_regression,
            "lasso_regression": sklearn_tools.lasso_regression,
            "elasticnet_regression": sklearn_tools.elasticnet_regression,
            "svr_regression": sklearn_tools.svr_regression,
            "random_forest_regression": sklearn_tools.random_forest_regression,
            "gradient_boosting_regression": sklearn_tools.gradient_boosting_regression,
        }
        if tool_name in regression_tools:
            if not targets:
                notes.append("Target column is missing for regression.")
                continue
            tool_args = {
                "data": data,
                "target": targets,
                "feature_names": feature_names,
                **tool_args,
            }
            result = regression_tools[tool_name](**tool_args)
            if "feature_importances" in result:
                chart = _build_feature_importance_chart(
                    "Feature Importance",
                    result.get("feature_names") or feature_names,
                    result.get("feature_importances", []),
                )
            elif "coefficients" in result:
                chart = _build_feature_importance_chart(
                    "Coefficient Magnitudes",
                    result.get("feature_names") or feature_names,
                    result.get("coefficients", []),
                )
            else:
                chart = _build_regression_chart(
                    result.get("predictions", []),
                    targets,
                    "Actual vs Predicted",
                )
            if chart:
                chart["meta"] = {**(chart.get("meta") or {}), "chartKind": tool_chart_map.get(tool_name)}
                charts.append(chart)
            notes.append(f"{tool_name.replace('_', ' ').title()} complete.")
            print(
                f"[ds] tool:end name={tool_name} status=ok charts={len(charts)} "
                f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
            )
            continue

        if tool_name == "logistic_regression":
            if not targets:
                notes.append("Target column is missing for classification.")
                continue
            tool_args = {
                "data": data,
                "target": targets,
                "feature_names": feature_names,
                **tool_args,
            }
            result = sklearn_tools.logistic_regression(**tool_args)
            if "coefficients" in result:
                chart = _build_feature_importance_chart(
                    "Logistic Regression Coefficients",
                    result.get("feature_names") or feature_names,
                    result.get("coefficients", [])[0] if result.get("coefficients") else [],
                )
                if chart:
                    chart["meta"] = {**(chart.get("meta") or {}), "chartKind": tool_chart_map.get(tool_name)}
                    charts.append(chart)
            if "accuracy" in result:
                metrics.append(f"logistic_regression accuracy: {result.get('accuracy')}")
            notes.append(f"Logistic regression accuracy: {result.get('accuracy')}")
            print(
                f"[ds] tool:end name={tool_name} status=ok charts={len(charts)} "
                f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
            )
            continue

        if tool_name == "random_forest_classification":
            if not targets:
                notes.append("Target column is missing for classification.")
                continue
            tool_args = {
                "data": data,
                "target": targets,
                "feature_names": feature_names,
                **tool_args,
            }
            result = sklearn_tools.random_forest_classification(**tool_args)
            chart = _build_feature_importance_chart(
                "Feature Importance",
                result.get("feature_names") or feature_names,
                result.get("feature_importances", []),
            )
            if chart:
                chart["meta"] = {**(chart.get("meta") or {}), "chartKind": tool_chart_map.get(tool_name)}
                charts.append(chart)
            if "accuracy" in result:
                metrics.append(f"random_forest_classification accuracy: {result.get('accuracy')}")
            notes.append("Random forest classification complete.")
            print(
                f"[ds] tool:end name={tool_name} status=ok charts={len(charts)} "
                f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
            )
            continue

        if tool_name == "fast_ica":
            tool_args = {
                "data": data,
                "n_components": tool_args.get("n_components", 3),
            }
            result = sklearn_tools.fast_ica(**tool_args)
            components = result.get("components", [])
            loadings = _transpose_components(components)
            chart = _build_loadings_chart(
                "ICA Loadings",
                loadings,
                feature_names,
            )
            if chart:
                chart["meta"] = {**(chart.get("meta") or {}), "chartKind": tool_chart_map.get(tool_name)}
                charts.append(chart)
            notes.append("ICA complete.")
            print(
                f"[ds] tool:end name={tool_name} status=ok charts={len(charts)} "
                f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
            )
            continue

        if tool_name == "nmf_decomposition":
            tool_args = {
                "data": data,
                "n_components": tool_args.get("n_components", 2),
            }
            result = sklearn_tools.nmf_decomposition(**tool_args)
            components = result.get("components", [])
            loadings = _transpose_components(components)
            desc = f"Reconstruction error: {result.get('reconstruction_err', 0):.4f}"
            if result.get("data_shifted"):
                desc += f" (data shifted by {result.get('shift_amount', 0):.2f})"
            chart = _build_loadings_chart(
                "NMF Components",
                loadings,
                feature_names,
                desc,
            )
            if chart:
                chart["meta"] = {**(chart.get("meta") or {}), "chartKind": tool_chart_map.get(tool_name)}
                charts.append(chart)
            notes.append("NMF decomposition complete.")
            print(
                f"[ds] tool:end name={tool_name} status=ok charts={len(charts)} "
                f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
            )
            continue

        if tool_name == "tsne_embedding":
            # Use a sample if dataset is large to avoid timeout
            sample_data = data[:500] if len(data) > 500 else data
            sample_targets = targets[:500] if targets and len(targets) > 500 else targets
            tool_args = {
                "data": sample_data,
                "n_components": 2,
                "perplexity": min(30.0, len(sample_data) - 1),
            }
            result = sklearn_tools.tsne_embedding(**tool_args)
            embedding = result.get("embedding", [])
            desc = f"Perplexity: {result.get('perplexity', 30):.1f}"
            if len(data) > 500:
                desc += f" (sampled {len(sample_data)} of {len(data)} rows)"
            chart = _build_embedding_chart(
                "t-SNE Embedding",
                embedding,
                labels=[int(t) for t in sample_targets] if sample_targets else None,
                description=desc,
            )
            if chart:
                chart["meta"] = {**(chart.get("meta") or {}), "chartKind": tool_chart_map.get(tool_name)}
                charts.append(chart)
            notes.append("t-SNE embedding complete.")
            print(
                f"[ds] tool:end name={tool_name} status=ok charts={len(charts)} "
                f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
            )
            continue

        # Clustering algorithms
        clustering_tools = {
            "kmeans_clustering": sklearn_tools.kmeans_clustering,
            "minibatch_kmeans_clustering": sklearn_tools.minibatch_kmeans_clustering,
            "dbscan_clustering": sklearn_tools.dbscan_clustering,
            "agglomerative_clustering": sklearn_tools.agglomerative_clustering,
            "spectral_clustering": sklearn_tools.spectral_clustering,
            "gaussian_mixture_clustering": sklearn_tools.gaussian_mixture_clustering,
            "optics_clustering": sklearn_tools.optics_clustering,
        }
        if tool_name in clustering_tools:
            cluster_args = {"data": data, **tool_args}
            # These algorithms don't use n_clusters parameter
            no_n_clusters = {"dbscan_clustering", "optics_clustering", "gaussian_mixture_clustering"}
            if "n_clusters" not in cluster_args and tool_name not in no_n_clusters:
                cluster_args["n_clusters"] = 3
            # GMM uses n_components instead of n_clusters
            if tool_name == "gaussian_mixture_clustering" and "n_components" not in cluster_args:
                cluster_args["n_components"] = 3
            result = clustering_tools[tool_name](**cluster_args)
            labels = result.get("labels", [])
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            desc = f"{n_clusters} clusters found"
            if "inertia" in result:
                desc += f", inertia: {result.get('inertia', 0):.2f}"
            chart = _build_cluster_distribution_chart(
                f"{tool_name.replace('_', ' ').title().replace('Clustering', 'Clusters')}",
                labels,
                desc,
            )
            if chart:
                chart["meta"] = {**(chart.get("meta") or {}), "chartKind": tool_chart_map.get(tool_name)}
                charts.append(chart)
            notes.append(f"{tool_name.replace('_', ' ').title()} complete.")
            print(
                f"[ds] tool:end name={tool_name} status=ok charts={len(charts)} "
                f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
            )
            continue

        # Gradient Boosting Classification
        if tool_name == "gradient_boosting_classification":
            if not targets:
                notes.append("Target column is missing for classification.")
                continue
            tool_args = {
                "data": data,
                "target": targets,
                "feature_names": feature_names,
                **tool_args,
            }
            result = sklearn_tools.gradient_boosting_classification(**tool_args)
            chart = _build_feature_importance_chart(
                "Feature Importance (GB Classifier)",
                result.get("feature_names") or feature_names,
                result.get("feature_importances", []),
            )
            if chart:
                chart["meta"] = {**(chart.get("meta") or {}), "chartKind": tool_chart_map.get(tool_name)}
                charts.append(chart)
            if "accuracy" in result:
                metrics.append(f"gradient_boosting_classification accuracy: {result.get('accuracy')}")
            notes.append(f"Gradient boosting classification accuracy: {result.get('accuracy', 'N/A')}")
            print(
                f"[ds] tool:end name={tool_name} status=ok charts={len(charts)} "
                f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
            )
            continue

        # KNN Classification
        if tool_name == "knn_classification":
            if not targets:
                notes.append("Target column is missing for classification.")
                continue
            knn_args = {"data": data, "target": targets, **tool_args}
            result = sklearn_tools.knn_classification(**knn_args)
            # KNN doesn't have feature importances, show predictions distribution
            predictions = result.get("predictions", [])
            if predictions:
                chart = _build_cluster_distribution_chart(
                    "KNN Prediction Distribution",
                    [int(p) for p in predictions],
                    f"Accuracy: {result.get('accuracy', 0):.2%}",
                )
                if chart:
                    chart["meta"] = {**(chart.get("meta") or {}), "chartKind": tool_chart_map.get(tool_name)}
                    charts.append(chart)
            if "accuracy" in result:
                metrics.append(f"knn_classification accuracy: {result.get('accuracy')}")
            notes.append(f"KNN classification accuracy: {result.get('accuracy', 'N/A')}")
            print(
                f"[ds] tool:end name={tool_name} status=ok charts={len(charts)} "
                f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
            )
            continue

        # Naive Bayes Classification
        if tool_name == "naive_bayes_classification":
            if not targets:
                notes.append("Target column is missing for classification.")
                continue
            nb_args = {"data": data, "target": targets, **tool_args}
            result = sklearn_tools.naive_bayes_classification(**nb_args)
            predictions = result.get("predictions", [])
            if predictions:
                chart = _build_cluster_distribution_chart(
                    "Naive Bayes Prediction Distribution",
                    [int(p) for p in predictions],
                    f"Accuracy: {result.get('accuracy', 0):.2%}",
                )
                if chart:
                    chart["meta"] = {**(chart.get("meta") or {}), "chartKind": tool_chart_map.get(tool_name)}
                    charts.append(chart)
            if "accuracy" in result:
                metrics.append(f"naive_bayes_classification accuracy: {result.get('accuracy')}")
            notes.append(f"Naive Bayes classification accuracy: {result.get('accuracy', 'N/A')}")
            print(
                f"[ds] tool:end name={tool_name} status=ok charts={len(charts)} "
                f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
            )
            continue

        notes.append(f"Tool {tool_name} is not wired yet.")
        print(
            f"[ds] tool:end name={tool_name} status=unwired "
            f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f}"
        )

    summary = plan.get("summary") or "Analysis complete."
    if notes:
        summary = f"{summary} " + " ".join(notes)
    total_elapsed_ms = (time.perf_counter() - started_at) * 1000
    dataset_label = entry.get("label") or dataset_id
    for chart in charts:
        chart["meta"] = {
            "datasetLabel": dataset_label,
            "queryTimeMs": total_elapsed_ms,
            **(chart.get("meta") or {}),
        }
    non_chart_response = {
        "notes": notes,
        "metrics": metrics,
    }
    if not charts:
        print(
            f"[ds] analysis:end dataset_id={dataset_id} charts=0 "
            f"elapsed_ms={(time.perf_counter() - started_at) * 1000:.1f}"
        )
        return {"message": summary, "chartSpec": None, "nonChartResponse": non_chart_response}
    print(
        f"[ds] analysis:end dataset_id={dataset_id} charts={len(charts)} "
        f"elapsed_ms={(time.perf_counter() - started_at) * 1000:.1f}"
    )
    return {"message": summary, "chartSpec": charts, "nonChartResponse": non_chart_response}
