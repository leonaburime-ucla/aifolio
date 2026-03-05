# ML Framework Adapters

Framework-specific adapter layers that provide stable import paths while
legacy monolith implementations are incrementally extracted.

## Packages
- `pytorch/`
  - `handlers.py`: API-request-facing train/distill entrypoints
  - `trainer.py`: concrete train/predict runtime implementation + module exports
  - `models.py`: PyTorch model builders and model metadata helpers
  - `data.py`: PyTorch dataset vectorization/scaling preparation
  - `distill.py`: PyTorch distillation runtime
  - `serialization.py`: PyTorch bundle save/load
- `tensorflow/`
  - `handlers.py`: API-request-facing train/distill entrypoints
  - `trainer.py`: concrete train/predict runtime implementation + module exports
  - `models.py`: TensorFlow model builders and model metadata helpers
  - `data.py`: TensorFlow dataset vectorization/scaling preparation
  - `distill.py`: TensorFlow distillation runtime
  - `serialization.py`: TensorFlow bundle save/load

## Current State
Framework trainer modules now own runtime loops and serialization paths.
Legacy entrypoint modules (`ai/ml/pytorch.py`, `ai/ml/tensorflow.py`) are
kept for API compatibility and delegated imports.

## Migration Target
Keep framework modules flat and cohesive while preserving legacy files as
compatibility shims.
