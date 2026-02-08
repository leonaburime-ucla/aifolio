from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
from agent_langchain import DEFAULT_MODEL_ID, run_chat, run_chat_response as langchain_chat_response
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
from langgraph_agents.status import get_status
from crypto_data import list_datasets, load_dataset
from ag_ui.core import (
    EventType,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder

app = FastAPI(title="AI Orchestrator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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


def _extract_text(content):
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _build_chat_payload(input: RunAgentInput):
    messages = []
    for message in input.messages:
        role = message.role
        messages.append(
            {
                "role": role,
                "content": _extract_text(message.content),
                "attachments": message.attachments or [],
            }
        )
    return {
        "messages": messages,
        "model": input.model or DEFAULT_MODEL_ID,
    }


@app.post("/agui")
async def agui_stream(payload: dict):
    input_data = RunAgentInput.model_validate(payload)
    run_id = input_data.run_id or str(uuid.uuid4())
    thread_id = input_data.thread_id or str(uuid.uuid4())
    input_data = input_data.model_copy(update={"run_id": run_id, "thread_id": thread_id})
    encoder = EventEncoder()

    async def event_stream():
        try:
            yield encoder.encode(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    run_id=run_id,
                    thread_id=thread_id,
                    input=input_data,
                )
            )
            message_id = f"msg_{run_id}"
            yield encoder.encode(
                TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START,
                    message_id=message_id,
                    role="assistant",
                )
            )

            chat_payload = _build_chat_payload(input_data)
            result_text = run_chat(chat_payload)
            yield encoder.encode(
                TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=result_text,
                )
            )
            yield encoder.encode(
                TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                )
            )
            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    run_id=run_id,
                    thread_id=thread_id,
                )
            )
        except Exception as exc:
            error_message = str(exc)
            yield encoder.encode(
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    run_id=run_id,
                    thread_id=thread_id,
                    message=error_message,
                )
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")

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


@app.get("/sklearn-tools")
def list_sklearn_tools():
    return {
        "status": "ok",
        "tools": sklearn_tools.list_available_tools(),
        "schemas": sklearn_tools.get_tools_schema(),
    }


if __name__ == "__main__":
    # Allow `python server.py` for local dev convenience.
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
