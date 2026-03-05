# ML Core Helpers

Shared framework-agnostic helpers used by both PyTorch and TensorFlow runtimes.

## Modules
- `contracts.py`
  - Payload list parsing (`exclude_columns`, `date_columns`)
  - `dataset_id`/`data_path` resolution
  - Shared train/distill numeric bound checks
- `preprocessing.py`
  - Non-finite value imputation for train/test and inference arrays
- `artifacts.py`
  - Artifact metadata utilities (currently file-size lookup)
- `types.py`
  - Shared `TrainingConfig`, `Metrics`, and `ModelBundle` dataclasses
- `request_helpers.py`
  - Shared request parsing for data path/target, column lists, and teacher model path resolution
- `mode_catalog.py`
  - Framework training-mode allowlists and distillation support catalog
- `handler_utils.py`
  - Shared numeric payload parsing, runtime error shaping, and distillation stats helpers

## Design Rule
No framework imports (`torch`, `tensorflow`) in this package.
