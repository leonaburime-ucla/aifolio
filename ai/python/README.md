# AI Python Workspace

This repo’s Python stack lives under `ai/` and hosts FastAPI + LangGraph agents and ML experiments.

## Structure (relative to repo root)

- `ai/lang_agents/` – FastAPI app + LangGraph orchestration.
- `ai/machine-learning/` – notebooks or scripts for model experiments.
- `ai/.venv/` – shared virtualenv for all Python tooling.

## Usage

```bash
cd ai
source .venv/bin/activate
cd machine-learning
python -c "import torch, tensorflow as tf; print(torch.__version__, tf.__version__)"
```

The shared virtual environment lives at `ai/.venv` so both `ai/machine-learning` and `ai/lang_agents` can reuse the same packages without duplicating installs.

## Quick reminder (activate the venv)

```bash
source ai/.venv/bin/activate
```
