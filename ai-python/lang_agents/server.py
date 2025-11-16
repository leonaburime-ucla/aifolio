from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Orchestrator", version="0.1.0")


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
