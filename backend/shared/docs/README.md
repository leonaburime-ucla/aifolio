# Backend Python Workspace

This repo’s Python backend stack lives in `backend/` and hosts FastAPI routes, agent helpers, and ML runtimes.

## Structure (relative to `backend/`)

- `server/` – FastAPI app routes and HTTP integration layer.
- `agents/` – agent orchestration/status logic.
- `shared/` – provider and AG-UI support code.
- `ml/` – framework runtimes and shared training/predict helpers.
- `.venv/` – shared virtualenv for all Python tooling.

## Usage

```bash
source .venv/bin/activate
python -c "import torch, tensorflow as tf; print(torch.__version__, tf.__version__)"
```

The shared virtual environment lives at `backend/.venv` so all backend modules (`server`, `agents`, `ml`) can reuse the same packages without duplication.

## Quick reminder (activate the venv)

```bash
source .venv/bin/activate
```

## Run FastAPI server (exact commands)

```bash
source .venv/bin/activate
uvicorn server:app --reload
```
