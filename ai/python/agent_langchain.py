from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from google_gemini import DEFAULT_MODEL_ID, ensure_google_api_key_in_env, get_model
from langgraph_agents.langsmith import configure_langsmith
from langgraph_agents.status import record_run

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


def run_chat(payload: ChatPayload) -> str:
    """
    Minimal chat call for server usage.
    """
    state = formatMessage(payload)
    system_msg = SystemMessage(
        content=(
            "You are a helpful assistant. Reply ONLY in valid JSON with this shape:\n"
            '{ "message": string, "chartSpec": ChartSpec | ChartSpec[] | null }\n'
            "Do not wrap the JSON in markdown or code fences.\n"
            "If the user asks for a chart, set chartSpec with these fields:\n"
            "{ id, title, type ('line'|'area'|'bar'|'scatter'|'heatmap'|'box'|'dendrogram'), xKey, yKeys, data, unit?, currency?, timeframe?, source? }\n"
            "If multiple charts are needed, return an array of chartSpec entries.\n"
            "If no chart is needed, set chartSpec to null."
        )
    )
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

# Configure it with what the point of the project is 
# to better answer questions

# Import tools from scikit-learn so the agent can use it

# Install memory so the agent can remember the conversation 
# I dont have sessions so need to keep track of the user another
# way and send that info on each request
import time
import json
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional
