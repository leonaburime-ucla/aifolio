# ai/ml/frameworks/pytorch/serialization.py

Spec ID:      PT-SERIAL-001
Version:      0.5
Last Edited:  2026-02-25T00:00:00Z
Hash:         sha256:31453773f60b7bc05ce63e0164005d8456788f96ffb75b031a17d02d392895ab

## Goals
- Define the save/load contract for PyTorch model bundles, ensuring lossless round-trip serialization.

## Scope
- **In scope**: `save_bundle` and `load_bundle` functions; on-disk file format.
- **Out of scope**: In-memory registry (see `handlers/handlers.feature.spec.md`); model architecture details (see `models/models.feature.spec.md`).

---

## On-Disk Format

### File: `model_bundle.pt` (torch.save)

The saved payload is a Python dict with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `state_dict` | `OrderedDict` | Model weight tensors |
| `task` | `str` | `"classification"` or `"regression"` |
| `target_column` | `str` | Name of the target column |
| `input_dim` | `int` | Number of input features |
| `output_dim` | `int` | Number of output classes, or 1 for regression |
| `class_names` | `list[str] \| None` | Label strings for classification; `None` for regression |
| `vectorizer` | `DictVectorizer` | Fitted sklearn vectorizer (pickle-serialized) |
| `scaler` | `StandardScaler` | Fitted feature scaler (pickle-serialized) |
| `feature_medians` | `np.ndarray \| None` | Imputation reference medians; may be absent in legacy files |
| `label_encoder` | `LabelEncoder \| None` | Fitted encoder (classification only) |
| `target_scaler` | `StandardScaler \| None` | Fitted scaler (regression only) |
| `model_config` | `dict` | Architecture hyperparameters for reconstruction |

### `model_config` sub-dict

| Key | Type | Source |
|-----|------|--------|
| `training_mode` | `str` | `model_training_mode(model)` |
| `hidden_dim` | `int` | `model_hidden_dim(model)` |
| `num_hidden_layers` | `int` | `model_num_hidden_layers(model)` |
| `dropout` | `float` | `model_dropout(model)` |

### File: `metrics.json` (optional, JSON)

Written only when the `metrics` argument is not `None`. Content: `dataclasses.asdict(metrics)` with `indent=2` and UTF-8 encoding.

---

## Function Contracts

### `save_bundle(bundle, output_dir, metrics?) → Path`

Serializes a `ModelBundle` — including all sklearn artifacts, the model's `state_dict`, and architecture metadata — to a directory on disk. The saved file can be loaded back with `load_bundle` to reproduce identical predictions on the same input.

- **REQ-S01**: Creates `output_dir` and any missing parent directories (`mkdir(parents=True, exist_ok=True)`).
- **REQ-S02**: Saves the bundle payload dict to `output_dir/model_bundle.pt` via `torch.save`.
- **REQ-S03**: If `metrics` is not `None`, writes `metrics.json` alongside `model_bundle.pt` with `dataclasses.asdict(metrics)`.
- **REQ-S04**: Returns the `Path` object pointing to the saved `model_bundle.pt`.
- **REQ-S05**: `model_config` is extracted from the live model instance via introspection helpers at save time — not from the config originally used to construct the model.

---

### `load_bundle(path, map_location?) → ModelBundle`

Deserializes a `ModelBundle` from a `.pt` file previously written by `save_bundle`, reconstructing the model architecture from the saved `model_config` and restoring all sklearn preprocessing artifacts. Supports CPU-only loading of GPU-trained models via `map_location`.

- **REQ-S06**: Loads the payload via `torch.load(path, map_location=map_location)`.
- **REQ-S07**: Reconstructs the model via `build_model` using `model_config` from the payload.
- **REQ-S08**: Loads the `state_dict` into the reconstructed model.
- **REQ-S09**: Falls back to safe defaults if `model_config` keys are missing: `training_mode="mlp_dense"`, `hidden_dim=128`, `num_hidden_layers=2`, `dropout=0.0`.
- **REQ-S10**: Returns a fully populated `ModelBundle` dataclass (optional fields may be `None` if absent in the file).

---

## Acceptance Criteria

- **AC-S01 (REQ-S01) [P1]**: Given a non-existent `output_dir` path, when `save_bundle` is called, then the directory and all parents are created before writing.
- **AC-S02 (REQ-S02) [P1]**: Given a valid bundle and output directory, when `save_bundle` is called, then a file named `model_bundle.pt` exists in `output_dir` afterwards.
- **AC-S03 (REQ-S03) [P2]**: Given `metrics` is not `None`, when `save_bundle` is called, then a file named `metrics.json` is written alongside `model_bundle.pt`.
- **AC-S03b (REQ-S03) [P2]**: Given `metrics is None`, when `save_bundle` is called, then no `metrics.json` file is written.
- **AC-S04 (REQ-S04) [P1]**: Given a successful save, when `save_bundle` returns, then the returned `Path` points to the `model_bundle.pt` file and the file exists.
- **AC-S05 (REQ-S05) [P2]**: Given a model whose introspected `hidden_dim` differs from the config used to construct it, when `save_bundle` is called, then `model_config.hidden_dim` reflects the introspected (live) value.
- **AC-S06 (REQ-S06) [P1]**: Given a valid `.pt` path, when `load_bundle` is called, then the payload is loaded via `torch.load` without error.
- **AC-S07 (REQ-S07) [P1]**: Given a saved bundle with `model_config.training_mode="tabresnet"`, when `load_bundle` is called, then the reconstructed model is a `TabResNet` instance.
- **AC-S08 (REQ-S08) [P1]**: Given a saved and loaded bundle, when `model.state_dict()` is compared, then it matches the weights from the original saved file.
- **AC-S09 (REQ-S09) [P1]**: Given a legacy `.pt` file with no `model_config` key, when `load_bundle` is called, then it defaults to `training_mode="mlp_dense"`, `hidden_dim=128`, `num_hidden_layers=2`, `dropout=0.0` and completes without error.
- **AC-S10 (REQ-S10) [P1]**: Given a valid `.pt` file, when `load_bundle` returns, then the result is a `ModelBundle` dataclass instance with `model`, `vectorizer`, and `scaler` all non-None.

---

## Invariants
- **INV-S01**: Round-trip guarantee — `load_bundle(path_from_save_bundle(b, d))` produces a `ModelBundle` whose model generates identical predictions for the same input as the original `b`.
- **INV-S02**: All sklearn artifacts (`vectorizer`, `scaler`, `label_encoder`, `target_scaler`) are pickle-serialized inside the `.pt` file.
- **INV-S03**: `model_config` is always present in newly saved files; may be absent in legacy files (fallback defaults apply).

---

## Edge Cases

- **EC-S01**: Legacy `.pt` file missing the `model_config` key.
  Expected behavior: `load_bundle` uses fallback defaults (`mlp_dense`, `hidden_dim=128`, `num_hidden_layers=2`, `dropout=0.0`) and completes successfully.

- **EC-S02**: Legacy `.pt` file missing the `feature_medians` key.
  Expected behavior: `load_bundle` sets `bundle.feature_medians = None`; no error raised; prediction falls back to zero-fill imputation.

- **EC-S03**: `map_location="cpu"` used to load a bundle trained on GPU.
  Expected behavior: All model tensors are mapped to CPU; returned model is on CPU; predictions work correctly on CPU-only machines.

---

## Dependencies

| Dependency | What It Provides | Failure Mode | Fallback |
|---|---|---|---|
| `torch` | `torch.save`, `torch.load` | `ImportError` at module load | None — module unusable |
| `pathlib.Path` | Directory creation, path handling | N/A (stdlib) | N/A |
| `json` | `metrics.json` serialization | N/A (stdlib) | N/A |
| `dataclasses.asdict` | Serializing `Metrics` dataclass | `TypeError` if `Metrics` contains non-serializable fields | None — propagates to caller |
| `ai.ml.frameworks.pytorch.models.build_model` | Model reconstruction from `model_config` | Raises on unknown `training_mode` | Defaults from REQ-S09 cover missing keys |
| `ai.ml.frameworks.pytorch.models.model_training_mode`, `model_hidden_dim`, `model_num_hidden_layers`, `model_dropout` | Introspecting live model at save time | Returns fallback defaults on unknown model | Defaults used |
| `ai.ml.core.types.Metrics`, `ModelBundle` | Return and param types | `AttributeError` on malformed bundle | None |

---

## Open Questions

None at this time.

---

## Traceability
- Requirement and acceptance trace matrix: [pytorch-backend-traceability.spec.md](../../../../traceability/pytorch-backend-traceability.spec.md#pt-serial-001)
