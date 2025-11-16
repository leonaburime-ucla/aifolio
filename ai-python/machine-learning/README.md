# Machine Learning Workspace

This folder isolates Python experiments for agents that need heavier ML workloads.

## Usage

```bash
cd ai-python
source .venv/bin/activate
cd machine-learning
python -c "import torch, tensorflow as tf; print(torch.__version__, tf.__version__)"
```

The shared virtual environment now lives at `ai-python/.venv` so both `machine-learning` notebooks and `lang_agents` can reuse the same packages (PyTorch, TensorFlow, FastAPI, etc.) without duplicating installs.
