# Backend Workspace

This folder hosts the Python backend stack for FastAPI routes, agent orchestration helpers, and local ML tooling (PyTorch, TensorFlow, scikit-learn). It follows Orc-BASH + Domain-Driven Design boundaries so shared contracts remain consistent across runtimes.

## Structure

- `server/` – FastAPI app entrypoints and route wiring.
- `agents/` – agent orchestration and status/trace helpers.
- `shared/` – provider wrappers, AG-UI runtime helpers, and chat application services.
- `ml/` – local ML runtime implementations and framework adapters.
- `data/` – bundled sample and ML datasets.
- `__tests__/` – backend test suite.
- `.venv/` – project virtualenv containing FastAPI, LangChain, LangGraph, Torch, TensorFlow, pytest, etc.

### Orc-BASH & DDD expectations
- **Orchestrators (`agents/*`, `shared/chat_application_service.py`)** coordinate intent handling, domain/tool fan-out, and response assembly.
- **Business Logic (`ml/core/*`, `agents/data_scientist/*`)** implements deterministic transforms and aggregate logic reused in tests.
- **API Layer (`server/http.py`, `shared/*`)** exposes route adapters and external model/provider integration with stable envelopes.
- **State/Hooks equivalents** are enforced via FastAPI response models that match the shared `AgentEnvelope` contract so UI hooks can consume them without translation.

## Setup

```bash
python3.12 -m venv .venv        # already created, rerun only if you need a fresh env
source .venv/bin/activate
pip install -r requirements.txt
```

(Current env already has fastapi, uvicorn[standard], langchain, langgraph, torch, tensorflow, scikit-learn, pytest.)

## Running the FastAPI server

```bash
source .venv/bin/activate
uvicorn server:app --reload
```

Alternatively: `python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload`.

## Tests

```bash
source .venv/bin/activate
pytest __tests__/test_server.py
```

## Notes

- FastAPI app import target is `server:app` (`server/__init__.py` re-exports `app` from `server/http.py`).
- For full regression, run `pytest` from this folder to include route, AG-UI, and ML tests.
