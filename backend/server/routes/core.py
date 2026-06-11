from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agents import get_status, get_trace_report
from agents.data_scientist import (
    list_sample_datasets,
    load_sample_dataset,
    run_data_scientist,
    run_data_scientist_tool,
    run_demo_pca_transform,
)
from application.agui.service import create_agui_stream_response
from application.chat.service import run_unified_chat
from ml.ml_data import list_ml_datasets, load_ml_dataset
from server.routes.deps import resolve_http_override
from shared.agent_langchain import DEFAULT_MODEL_ID, run_chat_response as langchain_chat_response
from shared.google_gemini import list_gemini_models, resolve_default_model_id
from shared.tools import sklearn_tools

router = APIRouter()


def _runtime(name: str, fallback):
    return resolve_http_override(name, fallback)


def _run_chat_research(payload: dict):
    """Run unified backend chat flow and normalize route response envelope."""

    mode, response = _runtime("run_unified_chat", run_unified_chat)(payload)
    return {
        "status": "ok",
        "mode": mode,
        "result": response,
        "message": str(response.get("message") or ""),
        "chartSpec": response.get("chartSpec"),
        "actions": response.get("actions") or [],
    }


@router.post("/chat")
def chat_post(payload: dict):
    return _run_chat_research(payload)


@router.post("/chat-research")
def chat_research_post(payload: dict):
    return _run_chat_research(payload)


@router.get("/chat")
def chat_get(message: str = "tell me a joke"):
    return _runtime("langchain_chat_response", langchain_chat_response)(
        {"message": message, "attachments": [], "model": DEFAULT_MODEL_ID}
    )


@router.post("/agui")
async def agui_stream(payload: dict):
    return _runtime("create_agui_stream_response", create_agui_stream_response)(payload)


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/llm/ping")
def ping_llm(payload: dict):
    return JSONResponse(
        {
            "message": "LLM endpoint placeholder",
            "received": payload,
        }
    )


@router.post("/llm/ds")
def data_scientist(payload: dict):
    dataset_id = payload.get("dataset_id")
    if dataset_id:
        _, response = _runtime("run_unified_chat", run_unified_chat)(payload)
        return {"status": "ok", "mode": "coordinator", "result": response}

    message = payload.get("message", "Run the requested sklearn tool.")
    model_id = payload.get("model") or DEFAULT_MODEL_ID
    tool_name = payload.get("tool_name")
    tool_args = payload.get("tool_args", {})
    if tool_name:
        result = _runtime("run_data_scientist_tool", run_data_scientist_tool)(
            message=message,
            tool_name=tool_name,
            tool_args=tool_args,
            model_id=model_id,
        )
        return {"status": "ok", "mode": "tool", **result}

    result = _runtime("run_data_scientist", run_data_scientist)(message, model_id=model_id)
    return {"status": "ok", "mode": "chat", "result": result}


@router.get("/llm/ds")
def data_scientist_get(n_components: int = 2):
    demo = _runtime("run_demo_pca_transform", run_demo_pca_transform)(n_components=n_components)
    return {"status": "ok", "mode": "pca-demo", **demo}


@router.get("/llm/gemini-models")
def gemini_models():
    models = _runtime("list_gemini_models", list_gemini_models)()
    return {
        "status": "ok",
        "currentModel": _runtime("resolve_default_model_id", resolve_default_model_id)(models),
        "models": models,
    }


@router.get("/llm/agent-status")
def agent_status():
    return {"status": "ok", "data": _runtime("get_status", get_status)()}


@router.get("/llm/langsmith/trace/{trace_id}")
def get_langsmith_trace_report(trace_id: str, project: str | None = None, include_raw: bool = True):
    try:
        report = _runtime("get_trace_report", get_trace_report)(
            trace_id=trace_id,
            project_name=project,
            include_raw=include_raw,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": "Failed to fetch LangSmith trace.",
                "detail": str(exc),
            },
        )

    if report.get("status") == "error":
        return JSONResponse(status_code=404, content=report)
    return report


@router.get("/sample-data")
def list_sample_data():
    return {"status": "ok", "datasets": _runtime("list_sample_datasets", list_sample_datasets)()}


@router.get("/sample-data/{dataset_id}")
def get_sample_data(dataset_id: str):
    result = _runtime("load_sample_dataset", load_sample_dataset)(dataset_id)
    if result.get("status") == "error":
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": result.get("error")},
        )
    return result


@router.get("/ml-data")
def list_ml_data():
    return {"status": "ok", "datasets": _runtime("list_ml_datasets", list_ml_datasets)()}


@router.get("/ml-data/{dataset_id}")
def get_ml_data(dataset_id: str, row_limit: int | None = None, sheet_name: str | None = None):
    result = _runtime("load_ml_dataset", load_ml_dataset)(
        dataset_id,
        row_limit=row_limit,
        sheet_name=sheet_name,
    )
    if result.get("status") == "error":
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": result.get("error")},
        )
    return result


@router.get("/sklearn-tools")
def list_sklearn_tools():
    return {
        "status": "ok",
        "tools": _runtime("sklearn_tools", sklearn_tools).list_available_tools(),
        "schemas": _runtime("sklearn_tools", sklearn_tools).get_tools_schema(),
    }
