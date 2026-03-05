import json
import time
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from backend.agents import configure_langsmith, record_run
from google_gemini import DEFAULT_MODEL_ID, ensure_google_api_key_in_env, get_model

configure_langsmith()

load_dotenv()  # This loads the variables from .env into the environment


## Human vs AI vs System messages

# - SystemMessage: Instructions that set behavior/role (e.g., “You are a crypto analyst”). It’s highest priority; use it for policy/role/format rules.
# - HumanMessage: The user’s input. What the user said or sent (text, attachments).
# - AIMessage: The model’s response. You append this after you call the model.


ensure_google_api_key_in_env()



class ChatState(TypedDict):
    messages: List[BaseMessage]


class ChatPayload(TypedDict, total=False):
    message: str
    attachments: list
    model: str
    messages: list
    tools: list
    context: list
    response_mode: str


def _format_tools_for_prompt(payload: ChatPayload) -> str:
    tools = payload.get("tools") or []
    if not isinstance(tools, list) or not tools:
        return "No frontend tools were provided for this run."
    lines: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name", "")).strip()
        if not name:
            continue
        description = str(tool.get("description", "")).strip()
        lines.append(f"- {name}: {description}")
    return "\n".join(lines) if lines else "No frontend tools were provided for this run."


def _format_context_for_prompt(payload: ChatPayload) -> str:
    context_items = payload.get("context") or []
    if not isinstance(context_items, list) or not context_items:
        return "No additional app context."
    lines: list[str] = []
    for item in context_items:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description", "")).strip()
        value = str(item.get("value", "")).strip()
        if not description and not value:
            continue
        lines.append(f"- {description}: {value}".strip(": "))
    return "\n".join(lines) if lines else "No additional app context."


def formatMessage(payload: ChatPayload) -> ChatState:
    """
    payload shape:
      {
        "message": "...",
        "attachments": [...]
      }
    """
    history = payload.get("messages") or []
    if history:
        messages: List[BaseMessage] = []
        for item in history:
            role = item.get("role")
            content = item.get("content", "")
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                attachments = item.get("attachments")
                messages.append(
                    HumanMessage(
                        content=content,
                        additional_kwargs={"attachments": attachments} if attachments else {},
                    )
                )
        return {"messages": messages}

    text = payload.get("message", "")
    attachments = payload.get("attachments")

    human_msg = HumanMessage(
        content=text,
        additional_kwargs={"attachments": attachments} if attachments else {},
    )

    return {"messages": [human_msg]}


def _build_system_prompt(payload: ChatPayload) -> str:
    """Assemble the model system prompt for the current chat request."""
    tools_text = _format_tools_for_prompt(payload)
    context_text = _format_context_for_prompt(payload)
    response_mode = str(payload.get("response_mode") or "full").strip().lower()
    base_prompt = (
        "You are a helpful assistant. Reply ONLY in valid JSON with this exact shape:\n"
        '{ "message": string, "chartSpec": ChartSpec | ChartSpec[] | null, "actions": { "name": string, "args": object }[] }\n'
        "Do not wrap JSON in markdown or code fences.\n"
        "Do not include any extra keys at the top level.\n"
        "If the user asks for charting, populate chartSpec with one or more entries.\n"
        "ChartSpec fields: { id, title, description?, type, xKey, yKeys, xLabel?, yLabel?, zKey?, colorKey?, errorKeys?, data, unit?, currency?, timeframe?, source?, meta? }.\n"
        "Allowed chart types: 'line'|'area'|'bar'|'scatter'|'histogram'|'density'|'roc'|'pr'|'errorbar'|'heatmap'|'box'|'biplot'|'dendrogram'.\n"
        "Prefer those chart types and avoid unsupported UI types.\n"
        "If multiple charts are needed, return chartSpec as an array.\n"
        "If no chart is needed, set chartSpec to null.\n"
        "Use `actions` for frontend tools when appropriate.\n"
        "For compound requests, include multiple actions in execution order.\n"
        "Example: for 'run sweep with autodistill and then train it', emit:\n"
        '[{"name":"set_pytorch_form_fields","args":{"fields":{"set_sweep_values":true,"auto_distill":true}}},{"name":"start_pytorch_training_runs","args":{}}]\n'
        "When a request contains 'then', treat it as ordered steps and do not collapse to a single action.\n"
        "When the user asks to start/run/train PyTorch from the current UI state and `start_pytorch_training_runs` is available, prefer:\n"
        '{"name":"start_pytorch_training_runs","args":{}}\n'
        "Do not enable `set_sweep_values` (or legacy `run_sweep`) unless the user explicitly asks for sweep/hyperparameter sweep.\n"
        "Use `train_pytorch_model` only when the user explicitly provides `dataset_id` and `target_column` values.\n"
        "If the user asks to navigate/open/go to a page and `navigate_to_page` is available, add an action like:\n"
        '{"name":"navigate_to_page","args":{"route":"/agentic-research"}}\n'
        "If no tool action is needed, return actions as an empty array.\n"
        "\nAvailable frontend tools:\n"
        f"{tools_text}\n"
        "\nAdditional app context:\n"
        f"{context_text}"
    )
    if response_mode == "actions_only":
        return (
            base_prompt
            + "\nPLANNING MODE:\n"
            "- Return ONLY tool actions needed to satisfy the user request.\n"
            "- Keep `message` short and operational (or empty string).\n"
            "- Always set `chartSpec` to null in this mode.\n"
            "- Do not include analysis prose in this mode.\n"
            "- If no tool action is needed, return actions as [].\n"
        )
    return base_prompt


def run_chat(payload: ChatPayload) -> str:
    """
    Minimal chat call for server usage.
    """
    state = formatMessage(payload)
    system_msg = SystemMessage(content=_build_system_prompt(payload))
    messages = [system_msg] + state["messages"]
    model_id = payload.get("model") or DEFAULT_MODEL_ID
    result = get_model(model_id).invoke(messages)
    return result.content


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    return cleaned


def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a model response expected to be a JSON object string.

    Example input:
        '{ "message": "Hello", "chartSpec": null }'

    Example output:
        {"message": "Hello", "chartSpec": None}
    """
    try:
        cleaned = _strip_json_fences(text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict) and "message" in parsed:
            return parsed
    except Exception:
        return None
    return None


def run_chat_response(payload: dict):
    """
    Wrapper that runs the chat model and returns a JSONResponse-ready payload.
    """
    start_time = time.time()
    model_id = payload.get("model") or DEFAULT_MODEL_ID
    try:
        result = run_chat(payload)
        parsed = _parse_llm_json(result) if isinstance(result, str) else None
        record_run(model_id=model_id, latency_ms=(time.time() - start_time) * 1000)
        return {"status": "ok", "result": parsed if parsed is not None else result, "model": model_id}
    except Exception as exc:
        msg = str(exc)
        print(f"LLM error: {msg}")
        record_run(model_id=model_id, latency_ms=(time.time() - start_time) * 1000, error=msg)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "error": "Quota exceeded. Check Gemini API limits/billing.",
                    "detail": msg,
                },
            )
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": "LLM call failed.",
                "detail": msg,
            },
        )
