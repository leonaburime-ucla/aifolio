# AI Python Workspace

This folder hosts the Python stack for agents that rely on FastAPI, LangChain/ LangGraph, and local ML tooling (PyTorch, TensorFlow, scikit-learn).

## Structure

- `lang_agents/` – FastAPI app + LangGraph orchestration.
- `machine-learning/` – notebooks or scripts that experiment with raw models.
- `.venv/` – shared virtualenv containing FastAPI, LangChain, LangGraph, Torch, TensorFlow, pytest, etc.

## Setup

```bash
cd ai-python
python3.12 -m venv .venv        # already created, rerun only if you need a fresh env
source .venv/bin/activate
pip install -r requirements.txt # optional placeholder; run pip install manually if no requirements file yet
```

(Current env already has fastapi, uvicorn[standard], langchain, langgraph, torch, tensorflow, scikit-learn, pytest.)

## Running the FastAPI server

```bash
cd ai-python
source .venv/bin/activate
cd lang_agents
uvicorn server:app --reload       # dev server
```

Alternatively: `fastapi dev server.py` or `python -m uvicorn server:app --host 0.0.0.0 --port 8000`.

## Tests

```bash
cd ai-python
source .venv/bin/activate
pytest lang_agents/__tests__/test_server.py
```

## Notes

- Keep LangGraph graphs + tool definitions inside `lang_agents/` so FastAPI routes can load them directly.
- When the TypeScript agents are ready, mirror the specs defined here to maintain parity between the Next.js and FastAPI runtimes.
