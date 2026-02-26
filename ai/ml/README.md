# AI ML Runtime (`ai/ml`)

This package hosts local ML runtime capabilities for:
- PyTorch tabular training/distillation/prediction
- TensorFlow tabular training/distillation/prediction
- Shared preprocessing and payload contract helpers

It is being migrated from monolithic framework files into modular framework/core layers.

## Current Structure

### Legacy runtime modules (still active)
- `pytorch.py`: compatibility entrypoint; delegates runtime operations to framework modules.
- `tensorflow.py`: compatibility entrypoint; delegates runtime operations to framework modules.

### Shared core modules (new)
- `core/contracts.py`: payload parsing + common bounds validators.
- `core/preprocessing.py`: non-finite value imputation helpers.
- `core/artifacts.py`: artifact metadata helpers (file size lookup).
- `core/types.py`: shared runtime dataclasses.
- `core/request_helpers.py`: shared request parsing helpers.

### Framework adapters (new)
- `frameworks/pytorch/handlers.py`: API handler adapter surface.
- `frameworks/pytorch/trainer.py`: concrete train/predict runtime and exports.
- `frameworks/pytorch/models.py`, `data.py`, `distill.py`, `serialization.py`: flat PyTorch runtime slices.
- `frameworks/tensorflow/handlers.py`: API handler adapter surface.
- `frameworks/tensorflow/trainer.py`: concrete train/predict runtime and exports.
- `frameworks/tensorflow/models.py`, `data.py`, `distill.py`, `serialization.py`: flat TensorFlow runtime slices.

## Runtime Data

- Datasets: `ai/ml/data/`
- PyTorch artifacts: `ai/ml/artifacts/`
- TensorFlow artifacts: `ai/ml/tensorflow_artifacts/`

## API Integration

`ai/python/server.py` imports framework adapters from:
- `ml.frameworks.pytorch.*`
- `ml.frameworks.tensorflow.*`

This keeps server wiring stable while internals are migrated out of legacy modules.

## Refactor Status

Completed:
- Shared payload/list parsing extraction
- Shared bounds validation extraction
- Shared imputation extraction
- Shared artifact size helper extraction
- Framework adapter package + server wiring
- PyTorch train/distill/predict/save/load extraction into `frameworks/pytorch/trainer.py`
- TensorFlow train/distill/predict/save/load extraction into `frameworks/tensorflow/trainer.py`

Planned next:
- Add tests for framework handlers and trainer modules (fast first, matrix last).

## Testing Policy (when enabled)

- Fast checks: `unit`, `contract`, `smoke`
- Heavy checks: `slow_matrix` (dataset x algorithm matrix), run last/manual/nightly

See:
- `ai/pytest.ini`
- `ai/ml/TODO_modularization.md`
