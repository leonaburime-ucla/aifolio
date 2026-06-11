"""Compatibility facade for the FastAPI backend entrypoint.

The real composition root lives in `server.app`; this module keeps the old
import path and monkeypatch surface stable while route logic moved into
`server.routes`.
"""

from __future__ import annotations

from server.app import app
from server.routes.core import (
    _run_chat_research,
    agent_status,
    agui_stream,
    chat_get,
    chat_post,
    chat_research_post,
    data_scientist,
    data_scientist_get,
    gemini_models,
    get_langsmith_trace_report,
    get_ml_data,
    get_sample_data,
    health_check,
    list_ml_data,
    list_sample_data,
    list_sklearn_tools,
    ping_llm,
)
from server.routes.ml_framework import (
    PYTORCH_IMPORT_ERROR,
    PYTORCH_ARTIFACTS_DIR,
    TENSORFLOW_IMPORT_ERROR,
    TENSORFLOW_ARTIFACTS_DIR,
    pytorch_distill,
    pytorch_predict,
    pytorch_status,
    pytorch_train,
    tensorflow_distill,
    tensorflow_predict,
    tensorflow_status,
    tensorflow_train,
)
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
from ml.frameworks.pytorch.handlers import (
    handle_distill_request as handle_pytorch_distill_request,
    handle_train_request as handle_pytorch_train_request,
)
from ml.frameworks.pytorch.trainer import (
    load_bundle as load_pytorch_bundle,
    predict_rows as predict_pytorch_rows,
)
from ml.frameworks.tensorflow.handlers import (
    handle_distill_request as handle_tensorflow_distill_request,
    handle_train_request as handle_tensorflow_train_request,
)
from ml.frameworks.tensorflow.trainer import (
    load_bundle as load_tensorflow_bundle,
    predict_rows as predict_tensorflow_rows,
)
from ml.ml_data import list_ml_datasets, load_ml_dataset, resolve_ml_dataset_path
from server.ml import framework_status, run_predict_endpoint, run_training_or_distill_endpoint
from shared.agent_langchain import DEFAULT_MODEL_ID, run_chat_response as langchain_chat_response
from shared.google_gemini import list_gemini_models, resolve_default_model_id
from shared.tools import sklearn_tools

__all__ = [
    "app",
    "_run_chat_research",
    "agent_status",
    "agui_stream",
    "chat_get",
    "chat_post",
    "chat_research_post",
    "data_scientist",
    "data_scientist_get",
    "DEFAULT_MODEL_ID",
    "framework_status",
    "gemini_models",
    "get_langsmith_trace_report",
    "get_ml_data",
    "get_sample_data",
    "get_status",
    "get_trace_report",
    "health_check",
    "handle_pytorch_distill_request",
    "handle_pytorch_train_request",
    "handle_tensorflow_distill_request",
    "handle_tensorflow_train_request",
    "langchain_chat_response",
    "list_gemini_models",
    "list_ml_data",
    "list_ml_datasets",
    "list_sample_data",
    "list_sample_datasets",
    "list_sklearn_tools",
    "load_ml_dataset",
    "load_pytorch_bundle",
    "load_sample_dataset",
    "load_tensorflow_bundle",
    "pytorch_distill",
    "pytorch_predict",
    "pytorch_status",
    "pytorch_train",
    "predict_pytorch_rows",
    "predict_tensorflow_rows",
    "resolve_default_model_id",
    "resolve_ml_dataset_path",
    "run_data_scientist",
    "run_data_scientist_tool",
    "run_demo_pca_transform",
    "run_predict_endpoint",
    "run_training_or_distill_endpoint",
    "run_unified_chat",
    "sklearn_tools",
    "tensorflow_distill",
    "tensorflow_predict",
    "tensorflow_status",
    "tensorflow_train",
    "PYTORCH_IMPORT_ERROR",
    "PYTORCH_ARTIFACTS_DIR",
    "TENSORFLOW_IMPORT_ERROR",
    "TENSORFLOW_ARTIFACTS_DIR",
]
