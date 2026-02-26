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
from google_gemini import list_gemini_models, resolve_default_model_id
from langgraph_agents.data_scientist import (
    list_sample_datasets,
    load_sample_dataset,
    run_data_scientist,
    run_demo_pca_transform,
    run_data_scientist_tool,
)
from tools import sklearn_tools
from langgraph_agents.coordinator import coordinator_agent
from langgraph_agents.langsmith import get_trace_report
from langgraph_agents.status import get_status
from crypto_data import list_datasets, load_dataset
from ml_data import list_ml_datasets, load_ml_dataset, resolve_ml_dataset_path

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
    dataset_id = payload.get("dataset_id")
    message = payload.get("message") or ""
    if dataset_id and message:
        response = coordinator_agent(payload)
        if response:
            return {"status": "ok", "mode": "coordinator", "result": response}
    return langchain_chat_response(payload)


@app.post("/chat")
def chat_post(payload: dict):
    return _run_chat_research(payload)


@app.post("/chat-research")
def chat_research_post(payload: dict):
    return _run_chat_research(payload)

@app.get("/chat")
def chat_get(message: str = "tell me a joke"):
    return langchain_chat_response({"message": message, "attachments": [], "model": DEFAULT_MODEL_ID})


@app.post("/agui")
async def agui_stream(payload: dict):
    return create_agui_stream_response(payload)

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/llm/ping")
def ping_llm(payload: dict):
    return JSONResponse(
        {
            "message": "LLM endpoint placeholder",
            "received": payload,
        }
    )


@app.post("/llm/ds")
def data_scientist(payload: dict):
    dataset_id = payload.get("dataset_id")
    if dataset_id:
        response = coordinator_agent(payload)
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
    demo = run_demo_pca_transform(n_components=n_components)
    return {"status": "ok", "mode": "pca-demo", **demo}


@app.get("/llm/gemini-models")
def gemini_models():
    models = list_gemini_models()
    return {
        "status": "ok",
        "currentModel": resolve_default_model_id(models),
        "models": models,
    }


@app.get("/llm/agent-status")
def agent_status():
    return {"status": "ok", "data": get_status()}


@app.get("/llm/langsmith/trace/{trace_id}")
def get_langsmith_trace_report(trace_id: str, project: str | None = None, include_raw: bool = True):
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


@app.get("/crypto-data")
def list_crypto_data():
    return {"status": "ok", "datasets": list_datasets()}


@app.get("/crypto-data/{dataset_id}")
def get_crypto_data(dataset_id: str):
    found, content = load_dataset(dataset_id)
    if not found:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": "Dataset not found."},
        )
    return JSONResponse(content={"status": "ok", "data": content})


@app.get("/sample-data")
def list_sample_data():
    return {"status": "ok", "datasets": list_sample_datasets()}


@app.get("/sample-data/{dataset_id}")
def get_sample_data(dataset_id: str):
    result = load_sample_dataset(dataset_id)
    if result.get("status") == "error":
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": result.get("error")},
        )
    return result


@app.get("/ml-data")
def list_ml_data():
    return {"status": "ok", "datasets": list_ml_datasets()}


@app.get("/ml-data/{dataset_id}")
def get_ml_data(dataset_id: str, row_limit: int | None = None, sheet_name: str | None = None):
    result = load_ml_dataset(dataset_id, row_limit=row_limit, sheet_name=sheet_name)
    if result.get("status") == "error":
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": result.get("error")},
        )
    return result


@app.get("/sklearn-tools")
def list_sklearn_tools():
    return {
        "status": "ok",
        "tools": sklearn_tools.list_available_tools(),
        "schemas": sklearn_tools.get_tools_schema(),
    }

@app.post("/ml/pytorch/train")
def pytorch_train(payload: dict):
    if handle_pytorch_train_request is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": "PyTorch runtime is unavailable in this Python environment.",
                "details": PYTORCH_IMPORT_ERROR,
                "hint": "Activate ai/.venv or install torch in the interpreter running the server.",
            },
        )
    status_code, response = handle_pytorch_train_request(
        payload=payload,
        resolve_dataset_path=resolve_ml_dataset_path,
        artifacts_dir=PYTORCH_ARTIFACTS_DIR,
    )
    if status_code != 200:
        return JSONResponse(status_code=status_code, content=response)
    return response


@app.post("/ml/pytorch/distill")
def pytorch_distill(payload: dict):
    if handle_pytorch_distill_request is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": "PyTorch runtime is unavailable in this Python environment.",
                "details": PYTORCH_IMPORT_ERROR,
                "hint": "Activate ai/.venv or install torch in the interpreter running the server.",
            },
        )
    status_code, response = handle_pytorch_distill_request(
        payload=payload,
        resolve_dataset_path=resolve_ml_dataset_path,
        artifacts_dir=PYTORCH_ARTIFACTS_DIR,
    )
    if status_code != 200:
        return JSONResponse(status_code=status_code, content=response)
    return response


@app.post("/ml/pytorch/predict")
def pytorch_predict(payload: dict):
    if load_pytorch_bundle is None or predict_pytorch_rows is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": "PyTorch runtime is unavailable in this Python environment.",
                "details": PYTORCH_IMPORT_ERROR,
                "hint": "Activate ai/.venv or install torch in the interpreter running the server.",
            },
        )

    rows = payload.get("rows")
    if not isinstance(rows, list):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "rows must be an array of objects."},
        )

    model_path = payload.get("model_path")
    model_id = payload.get("model_id")
    if not model_path:
        if not model_id:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "model_path or model_id is required."},
            )
        model_path = str(PYTORCH_ARTIFACTS_DIR / model_id / "model_bundle.pt")

    try:
        bundle = load_pytorch_bundle(model_path)
        predictions = predict_pytorch_rows(bundle, rows, device=payload.get("device"))
        return {
            "status": "ok",
            "model_path": model_path,
            "count": len(predictions),
            "predictions": predictions,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": str(exc)},
        )


@app.get("/ml/pytorch/status")
def pytorch_status():
    available = PYTORCH_IMPORT_ERROR is None
    return {
        "status": "ok",
        "available": available,
        "error": PYTORCH_IMPORT_ERROR,
        "hint": None
        if available
        else "Activate ai/.venv or install torch in the interpreter running the server.",
    }


@app.post("/ml/tensorflow/train")
def tensorflow_train(payload: dict):
    if handle_tensorflow_train_request is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": "TensorFlow runtime is unavailable in this Python environment.",
                "details": TENSORFLOW_IMPORT_ERROR,
                "hint": "Activate ai/.venv or install tensorflow in the interpreter running the server.",
            },
        )
    status_code, response = handle_tensorflow_train_request(
        payload=payload,
        resolve_dataset_path=resolve_ml_dataset_path,
        artifacts_dir=TENSORFLOW_ARTIFACTS_DIR,
    )
    if status_code != 200:
        return JSONResponse(status_code=status_code, content=response)
    return response


@app.post("/ml/tensorflow/distill")
def tensorflow_distill(payload: dict):
    if handle_tensorflow_distill_request is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": "TensorFlow runtime is unavailable in this Python environment.",
                "details": TENSORFLOW_IMPORT_ERROR,
                "hint": "Activate ai/.venv or install tensorflow in the interpreter running the server.",
            },
        )
    status_code, response = handle_tensorflow_distill_request(
        payload=payload,
        resolve_dataset_path=resolve_ml_dataset_path,
        artifacts_dir=TENSORFLOW_ARTIFACTS_DIR,
    )
    if status_code != 200:
        return JSONResponse(status_code=status_code, content=response)
    return response


@app.post("/ml/tensorflow/predict")
def tensorflow_predict(payload: dict):
    if load_tensorflow_bundle is None or predict_tensorflow_rows is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": "TensorFlow runtime is unavailable in this Python environment.",
                "details": TENSORFLOW_IMPORT_ERROR,
                "hint": "Activate ai/.venv or install tensorflow in the interpreter running the server.",
            },
        )

    rows = payload.get("rows")
    if not isinstance(rows, list):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "rows must be an array of objects."},
        )

    model_path = payload.get("model_path")
    model_id = payload.get("model_id")
    if not model_path:
        if not model_id:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "model_path or model_id is required."},
            )
        model_path = str(TENSORFLOW_ARTIFACTS_DIR / model_id / "model_bundle.keras")

    try:
        bundle = load_tensorflow_bundle(model_path)
        predictions = predict_tensorflow_rows(bundle, rows, device=payload.get("device"))
        return {
            "status": "ok",
            "model_path": model_path,
            "count": len(predictions),
            "predictions": predictions,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": str(exc)},
        )


@app.get("/ml/tensorflow/status")
def tensorflow_status():
    available = TENSORFLOW_IMPORT_ERROR is None
    return {
        "status": "ok",
        "available": available,
        "error": TENSORFLOW_IMPORT_ERROR,
        "hint": None
        if available
        else "Activate ai/.venv or install tensorflow in the interpreter running the server.",
    }


if __name__ == "__main__":
    # Allow `python server.py` for local dev convenience.
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
