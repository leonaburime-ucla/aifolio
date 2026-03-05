"""FastAPI backend entrypoint for chat, AG-UI, data, and ML framework routes.

This module registers HTTP endpoints and delegates business logic to specialized
service/handler modules while maintaining stable response envelopes.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from pathlib import Path

# Allow importing sibling package: ai/ml/*
AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from agent_langchain import DEFAULT_MODEL_ID, run_chat_response as langchain_chat_response
from agui import create_agui_stream_response
from backend.agents import get_status, get_trace_report
from backend.agents.data_scientist import (
    list_sample_datasets,
    load_sample_dataset,
    run_data_scientist,
    run_data_scientist_tool,
    run_demo_pca_transform,
)
from backend.ml import list_ml_datasets, load_ml_dataset, resolve_ml_dataset_path
from backend.server.ml import framework_status, run_predict_endpoint, run_training_or_distill_endpoint
from chat_application_service import run_unified_chat
from google_gemini import list_gemini_models, resolve_default_model_id
from tools import sklearn_tools

PYTORCH_IMPORT_ERROR: str | None = None
PYTORCH_HANDLER_IMPORT_ERROR: str | None = None
PYTORCH_TRAINER_IMPORT_ERROR: str | None = None
try:
    from ml.frameworks.pytorch.handlers import (  # noqa: E402
        handle_distill_request as handle_pytorch_distill_request,
        handle_train_request as handle_pytorch_train_request,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    PYTORCH_HANDLER_IMPORT_ERROR = str(exc)
    handle_pytorch_distill_request = None  # type: ignore[assignment]
    handle_pytorch_train_request = None  # type: ignore[assignment]

try:
    from ml.frameworks.pytorch.trainer import (  # noqa: E402
        load_bundle as load_pytorch_bundle,
        predict_rows as predict_pytorch_rows,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    PYTORCH_TRAINER_IMPORT_ERROR = str(exc)
    load_pytorch_bundle = None  # type: ignore[assignment]
    predict_pytorch_rows = None  # type: ignore[assignment]
PYTORCH_IMPORT_ERROR = PYTORCH_HANDLER_IMPORT_ERROR or PYTORCH_TRAINER_IMPORT_ERROR

TENSORFLOW_IMPORT_ERROR: str | None = None
TENSORFLOW_HANDLER_IMPORT_ERROR: str | None = None
TENSORFLOW_TRAINER_IMPORT_ERROR: str | None = None
try:
    from ml.frameworks.tensorflow.handlers import (  # noqa: E402
        handle_distill_request as handle_tensorflow_distill_request,
        handle_train_request as handle_tensorflow_train_request,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    TENSORFLOW_HANDLER_IMPORT_ERROR = str(exc)
    handle_tensorflow_distill_request = None  # type: ignore[assignment]
    handle_tensorflow_train_request = None  # type: ignore[assignment]

try:
    from ml.frameworks.tensorflow.trainer import (  # noqa: E402
        load_bundle as load_tensorflow_bundle,
        predict_rows as predict_tensorflow_rows,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    TENSORFLOW_TRAINER_IMPORT_ERROR = str(exc)
    load_tensorflow_bundle = None  # type: ignore[assignment]
    predict_tensorflow_rows = None  # type: ignore[assignment]
TENSORFLOW_IMPORT_ERROR = TENSORFLOW_HANDLER_IMPORT_ERROR or TENSORFLOW_TRAINER_IMPORT_ERROR

app = FastAPI(title="AI Portfolio", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PYTORCH_ARTIFACTS_DIR = AI_ROOT / "ml" / "artifacts"
TENSORFLOW_ARTIFACTS_DIR = AI_ROOT / "ml" / "tensorflow_artifacts"



def _run_chat_research(payload: dict):
    """Run unified backend chat flow and normalize route response envelope.

    Args:
        payload: Incoming chat request body.

    Returns:
        JSON-serializable response dictionary for `/chat` endpoints.
    """
    mode, response = run_unified_chat(payload)
    return {
        "status": "ok",
        "mode": mode,
        "result": response,
        "message": str(response.get("message") or ""),
        "chartSpec": response.get("chartSpec"),
        "actions": response.get("actions") or [],
    }


@app.post("/chat")
def chat_post(payload: dict):
    """Handle POST `/chat`.

    Args:
        payload: Chat request payload.

    Returns:
        Unified chat response payload.
    """
    return _run_chat_research(payload)


@app.post("/chat-research")
def chat_research_post(payload: dict):
    """Handle POST `/chat-research`.

    Args:
        payload: Chat request payload.

    Returns:
        Unified chat response payload.
    """
    return _run_chat_research(payload)

@app.get("/chat")
def chat_get(message: str = "tell me a joke"):
    """Handle GET `/chat` using direct provider chat wrapper.

    Args:
        message: User prompt text.

    Returns:
        Provider response payload from `run_chat_response`.
    """
    return langchain_chat_response({"message": message, "attachments": [], "model": DEFAULT_MODEL_ID})


@app.post("/agui")
async def agui_stream(payload: dict):
    """Handle POST `/agui` and start AG-UI SSE stream.

    Args:
        payload: AG-UI run payload.

    Returns:
        Streaming response containing AG-UI protocol events.
    """
    return create_agui_stream_response(payload)

@app.get("/health")
def health_check():
    """Handle GET `/health`.

    Returns:
        Service health payload.
    """
    return {"status": "ok"}


@app.post("/llm/ping")
def ping_llm(payload: dict):
    """Handle POST `/llm/ping`.

    Args:
        payload: Arbitrary request payload.

    Returns:
        Echo-style placeholder JSON response.
    """
    return JSONResponse(
        {
            "message": "LLM endpoint placeholder",
            "received": payload,
        }
    )


@app.post("/llm/ds")
def data_scientist(payload: dict):
    """Handle POST `/llm/ds` for data-scientist flows and direct tool mode.

    Args:
        payload: Data-scientist chat/tool request payload.

    Returns:
        JSON response for coordinator, tool, or chat mode execution.
    """
    dataset_id = payload.get("dataset_id")
    if dataset_id:
        mode, response = run_unified_chat(payload)
        return {"status": "ok", "mode": "coordinator", "result": response}

    message = payload.get("message", "Run the requested sklearn tool.")
    model_id = payload.get("model") or DEFAULT_MODEL_ID
    tool_name = payload.get("tool_name")
    tool_args = payload.get("tool_args", {})
    if tool_name:
        result = run_data_scientist_tool(
            message=message,
            tool_name=tool_name,
            tool_args=tool_args,
            model_id=model_id,
        )
        return {"status": "ok", "mode": "tool", **result}

    result = run_data_scientist(message, model_id=model_id)
    return {"status": "ok", "mode": "chat", "result": result}


@app.get("/llm/ds")
def data_scientist_get(n_components: int = 2):
    """Handle GET `/llm/ds` demo PCA transform.

    Args:
        n_components: PCA component count for demo transform.

    Returns:
        PCA demo response payload.
    """
    demo = run_demo_pca_transform(n_components=n_components)
    return {"status": "ok", "mode": "pca-demo", **demo}


@app.get("/llm/gemini-models")
def gemini_models():
    """Handle GET `/llm/gemini-models`.

    Returns:
        Curated/discovered Gemini model list with default selection.
    """
    models = list_gemini_models()
    return {
        "status": "ok",
        "currentModel": resolve_default_model_id(models),
        "models": models,
    }


@app.get("/llm/agent-status")
def agent_status():
    """Handle GET `/llm/agent-status`.

    Returns:
        Agent runtime status payload.
    """
    return {"status": "ok", "data": get_status()}


@app.get("/llm/langsmith/trace/{trace_id}")
def get_langsmith_trace_report(trace_id: str, project: str | None = None, include_raw: bool = True):
    """Handle GET `/llm/langsmith/trace/{trace_id}`.

    Args:
        trace_id: LangSmith trace identifier.
        project: Optional project name override.
        include_raw: Whether raw trace payload should be included.

    Returns:
        Trace report payload or error `JSONResponse`.
    """
    try:
        report = get_trace_report(trace_id=trace_id, project_name=project, include_raw=include_raw)
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


@app.get("/sample-data")
def list_sample_data():
    """Handle GET `/sample-data`.

    Returns:
        Sample dataset descriptors.
    """
    return {"status": "ok", "datasets": list_sample_datasets()}


@app.get("/sample-data/{dataset_id}")
def get_sample_data(dataset_id: str):
    """Handle GET `/sample-data/{dataset_id}`.

    Args:
        dataset_id: Sample dataset identifier.

    Returns:
        Sample dataset payload or 404 error response.
    """
    result = load_sample_dataset(dataset_id)
    if result.get("status") == "error":
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": result.get("error")},
        )
    return result


@app.get("/ml-data")
def list_ml_data():
    """Handle GET `/ml-data`.

    Returns:
        ML dataset descriptors.
    """
    return {"status": "ok", "datasets": list_ml_datasets()}


@app.get("/ml-data/{dataset_id}")
def get_ml_data(dataset_id: str, row_limit: int | None = None, sheet_name: str | None = None):
    """Handle GET `/ml-data/{dataset_id}`.

    Args:
        dataset_id: ML dataset identifier.
        row_limit: Optional max number of rows to return.
        sheet_name: Optional worksheet name for spreadsheet sources.

    Returns:
        ML dataset payload or 404 error response.
    """
    result = load_ml_dataset(dataset_id, row_limit=row_limit, sheet_name=sheet_name)
    if result.get("status") == "error":
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": result.get("error")},
        )
    return result


@app.get("/sklearn-tools")
def list_sklearn_tools():
    """Handle GET `/sklearn-tools`.

    Returns:
        Available sklearn tool metadata and schemas.
    """
    return {
        "status": "ok",
        "tools": sklearn_tools.list_available_tools(),
        "schemas": sklearn_tools.get_tools_schema(),
    }

@app.post("/ml/pytorch/train")
def pytorch_train(payload: dict):
    """Handle POST `/ml/pytorch/train`.

    Args:
        payload: PyTorch training request payload.

    Returns:
        Shared train endpoint response payload or error response.
    """
    return run_training_or_distill_endpoint(
        payload=payload,
        handler=handle_pytorch_train_request,
        framework="PyTorch",
        package="torch",
        import_error=PYTORCH_IMPORT_ERROR,
        resolve_dataset_path=resolve_ml_dataset_path,
        artifacts_dir=PYTORCH_ARTIFACTS_DIR,
    )


@app.post("/ml/pytorch/distill")
def pytorch_distill(payload: dict):
    """Handle POST `/ml/pytorch/distill`.

    Args:
        payload: PyTorch distillation request payload.

    Returns:
        Shared distill endpoint response payload or error response.
    """
    return run_training_or_distill_endpoint(
        payload=payload,
        handler=handle_pytorch_distill_request,
        framework="PyTorch",
        package="torch",
        import_error=PYTORCH_IMPORT_ERROR,
        resolve_dataset_path=resolve_ml_dataset_path,
        artifacts_dir=PYTORCH_ARTIFACTS_DIR,
    )


@app.post("/ml/pytorch/predict")
def pytorch_predict(payload: dict):
    """Handle POST `/ml/pytorch/predict`.

    Args:
        payload: PyTorch prediction request payload.

    Returns:
        Prediction response payload or error response.
    """
    return run_predict_endpoint(
        payload=payload,
        load_bundle=load_pytorch_bundle,
        predict_rows=predict_pytorch_rows,
        framework="PyTorch",
        package="torch",
        import_error=PYTORCH_IMPORT_ERROR,
        artifacts_dir=PYTORCH_ARTIFACTS_DIR,
        artifact_filename="model_bundle.pt",
    )


@app.get("/ml/pytorch/status")
def pytorch_status():
    """Handle GET `/ml/pytorch/status`.

    Returns:
        PyTorch runtime availability status payload.
    """
    return framework_status(import_error=PYTORCH_IMPORT_ERROR, package="torch")


@app.post("/ml/tensorflow/train")
def tensorflow_train(payload: dict):
    """Handle POST `/ml/tensorflow/train`.

    Args:
        payload: TensorFlow training request payload.

    Returns:
        Shared train endpoint response payload or error response.
    """
    return run_training_or_distill_endpoint(
        payload=payload,
        handler=handle_tensorflow_train_request,
        framework="TensorFlow",
        package="tensorflow",
        import_error=TENSORFLOW_IMPORT_ERROR,
        resolve_dataset_path=resolve_ml_dataset_path,
        artifacts_dir=TENSORFLOW_ARTIFACTS_DIR,
    )


@app.post("/ml/tensorflow/distill")
def tensorflow_distill(payload: dict):
    """Handle POST `/ml/tensorflow/distill`.

    Args:
        payload: TensorFlow distillation request payload.

    Returns:
        Shared distill endpoint response payload or error response.
    """
    return run_training_or_distill_endpoint(
        payload=payload,
        handler=handle_tensorflow_distill_request,
        framework="TensorFlow",
        package="tensorflow",
        import_error=TENSORFLOW_IMPORT_ERROR,
        resolve_dataset_path=resolve_ml_dataset_path,
        artifacts_dir=TENSORFLOW_ARTIFACTS_DIR,
    )


@app.post("/ml/tensorflow/predict")
def tensorflow_predict(payload: dict):
    """Handle POST `/ml/tensorflow/predict`.

    Args:
        payload: TensorFlow prediction request payload.

    Returns:
        Prediction response payload or error response.
    """
    return run_predict_endpoint(
        payload=payload,
        load_bundle=load_tensorflow_bundle,
        predict_rows=predict_tensorflow_rows,
        framework="TensorFlow",
        package="tensorflow",
        import_error=TENSORFLOW_IMPORT_ERROR,
        artifacts_dir=TENSORFLOW_ARTIFACTS_DIR,
        artifact_filename="model_bundle.keras",
    )


@app.get("/ml/tensorflow/status")
def tensorflow_status():
    """Handle GET `/ml/tensorflow/status`.

    Returns:
        TensorFlow runtime availability status payload.
    """
    return framework_status(import_error=TENSORFLOW_IMPORT_ERROR, package="tensorflow")


if __name__ == "__main__":
    # Allow `python server.py` for local dev convenience.
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
