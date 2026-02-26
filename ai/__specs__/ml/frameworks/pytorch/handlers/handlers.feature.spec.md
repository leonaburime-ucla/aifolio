# ai/ml/frameworks/pytorch/handlers.py

Spec ID:      PT-HANDLERS-001
Version:      0.6
Last Edited:  2026-02-25T00:00:00Z
Hash:         sha256:140b08fcc708c745aa2ed9325865420a98492ed3aeff5f23f2a162fe8cfbc55c

## Goals
- Define the HTTP-facing handler contracts for PyTorch train and distill requests.
- Specify validation order, error responses, and success response shapes.

## Scope
- **In scope**: `handle_train_request`, `handle_distill_request`; in-memory bundle registry; internal helpers `_runtime_trainer`, `_parameter_count`, `_serialized_model_size_bytes`.
- **Out of scope**: Runtime training logic (see `trainer/trainer.feature.spec.md`); FastAPI route definitions in `ai/python/server.py`; distillation algorithm (see `distill/distill.feature.spec.md`).

---

## In-Memory Bundle Registry

Provides TTL-bounded in-memory storage for trained model bundles so that distillation and prediction calls can reference a recent training result by run ID without a disk round-trip. See `behavior.spec.md` for eviction and TTL behavior.

- **REQ-H01**: `_BUNDLE_REGISTRY` is an `InMemoryBundleRegistry[ModelBundle]` initialized with `ttl_seconds=900` and `max_items=128`.
- **REQ-H02**: `_store_in_memory_bundle(bundle) → str` stores the bundle in the registry and returns a string run ID.
- **REQ-H03**: `_load_in_memory_bundle(run_id) → ModelBundle | None` returns the bundle for the given run ID, or `None` if it has expired or been evicted.

---

## `handle_train_request(payload, resolve_dataset_path, artifacts_dir) → (int, dict)`

Validates an incoming HTTP training request, constructs the training configuration, invokes the PyTorch training runtime, stores the result in the in-memory registry, and returns a JSON-serializable response containing metrics and a run ID for subsequent distillation or prediction calls.

### Train Validation Chain (in order)

1. **REQ-H04**: Runtime availability — if PyTorch is not importable, return `(503, {"status": "error", "error": "..."})` via `runtime_unavailable_response("PyTorch", …)`.
2. **REQ-H05**: Data resolution — `resolve_data_target(payload, resolve_dataset_path)` must succeed; error returns `400`.
3. **REQ-H06**: `save_model` flag parsed from payload; `model_id` is taken from payload or auto-generated as UUID.
4. **REQ-H07**: Feature column parsing via `parse_feature_columns(payload)` — returns `(exclude_columns, date_columns, error)`.
5. **REQ-H08**: Numeric parameter parsing via `parse_train_numeric(payload)` — returns `(numeric_dict, error)`.
6. **REQ-H09**: Bounds validation via `validate_common_train_bounds(test_size, epochs, batch_size, learning_rate)`.
7. **REQ-H10**: Training mode normalized via `normalize_training_mode`.
8. **REQ-H11**: Mode must be in `PYTORCH_ALLOWED_TRAINING_MODES`; otherwise returns `400`.
9. **REQ-H12**: Hidden config validation via `hidden_config_error(training_mode, hidden_dim, num_hidden_layers, dropout)`.

### Execution

- **REQ-H13**: Constructs `TrainingConfig` from validated parameters and calls `runtime_trainer.train_model_from_file(…)`.
- **REQ-H14**: Stores the result bundle in the in-memory registry; if `save_model=true`, also persists to disk via `save_bundle`.

### Success Response `(200, dict)`

```json
{
  "status": "ok",
  "run_id": "<uuid>",
  "model_id": "<uuid> | null",
  "model_path": "<path> | null",
  "metrics": {
    "task": "classification | regression",
    "train_loss": float,
    "test_loss": float,
    "test_metric_name": "accuracy | rmse",
    "test_metric_value": float
  }
}
```

### Error Response `(400 | 503, dict)`

```json
{ "status": "error", "error": "<message>" }
```

- **REQ-H15**: Runtime exceptions raised during training are caught and returned as `(400, {"status": "error", "error": str(exc)})` — never re-raised.

---

## `handle_distill_request(payload, resolve_dataset_path, artifacts_dir) → (int, dict)`

Validates an incoming HTTP distillation request using its own validation chain (distinct from the train chain — it does NOT call `hidden_config_error` and checks the teacher source earlier), then delegates to the distillation runtime and returns a response that includes model compression statistics alongside standard metrics.

### Distill Validation Chain (in order)

1. **REQ-H04**: Runtime availability — same check as train.
2. **REQ-H05**: Data resolution — same check as train.
3. **REQ-H16**: Teacher source check — one of `teacher_run_id`, `teacher_model_path`, or `teacher_model_id` must be present in the payload; otherwise returns `400`. This check occurs before all other parameter parsing.
4. **REQ-H06**: `save_model` flag + `model_id` — same as train.
5. **REQ-H07**: Feature column parsing — same as train.
6. **REQ-H24**: Numeric parsing via `parse_distill_numeric(payload, default_epochs=60)` — distillation defaults to 60 epochs when `epochs` is absent.
7. **REQ-H17**: Bounds validation via `validate_common_distill_bounds` (includes train bounds plus `hidden_dim`, `num_hidden_layers`, `temperature`, `alpha`).
8. **REQ-H10**: Training mode normalized via `normalize_training_mode`.
9. **REQ-H18**: Training mode must not be in `PYTORCH_UNSUPPORTED_DISTILL_MODES`; otherwise returns `400`.
10. **REQ-H19**: If `teacher_run_id` is present but not found in registry → returns `(404, {"status": "error", "error": "Teacher run not found or expired."})`.

### Success Response `(200, dict)`

All fields from the train success response, plus:

```json
{
  "teacher_input_dim": int,
  "teacher_output_dim": int,
  "student_input_dim": int,
  "student_output_dim": int,
  "teacher_model_size_bytes": "int | null",
  "student_model_size_bytes": "int | null",
  "size_saved_bytes": "int | null",
  "size_saved_percent": "float | null",
  "teacher_param_count": int,
  "student_param_count": int,
  "param_saved_count": int,
  "param_saved_percent": "float | null"
}
```

- **REQ-H20**: Distill success response always includes the fields above, populated via `compute_distill_stats`.

---

## Internal Helpers

### `_runtime_trainer() → (module | None, error_str | None)`

Lazily imports the `.trainer` module so that PyTorch import errors are deferred to request time rather than server startup, allowing the server to boot and serve a `503` for PyTorch-dependent routes instead of crashing.

- **REQ-H21**: Attempts lazy import of `.trainer`; returns `(None, str(exc))` on `ModuleNotFoundError` or any import error.

### `_parameter_count(model) → int`

Returns the total number of trainable parameters in a model. Used in distillation responses to report compression ratios.

- **REQ-H22**: Returns `sum(p.numel() for p in model.parameters())`.

### `_serialized_model_size_bytes(model) → int | None`

Estimates the serialized disk size of a model by writing its `state_dict` to an in-memory buffer. Used to compute `size_saved_bytes` and `size_saved_percent` in distillation responses.

- **REQ-H23**: Serializes `model.state_dict()` to an in-memory `BytesIO` buffer via `torch.save`; returns the byte count. Returns `None` on any exception.

---

## Acceptance Criteria

- **AC-H01 (REQ-H01) [P2]**: Given the registry is initialized, when inspected, then `_BUNDLE_REGISTRY` has `ttl_seconds=900` and `max_items=128`.
- **AC-H02 (REQ-H02) [P1]**: Given a valid bundle, when `_store_in_memory_bundle` is called, then it returns a non-empty string run ID and the bundle is retrievable by that ID immediately after.
- **AC-H03 (REQ-H03) [P1]**: Given a run ID that has expired or was never stored, when `_load_in_memory_bundle` is called, then it returns `None`.
- **AC-H04 (REQ-H04) [P1]**: Given PyTorch is not installed, when `handle_train_request` is called, then it returns `(503, {"status": "error", …})` without attempting training.
- **AC-H05 (REQ-H05) [P1]**: Given an invalid dataset path in the payload, when `handle_train_request` is called, then data resolution fails and a `400` error is returned before training begins.
- **AC-H06 (REQ-H06) [P2]**: Given `save_model=true` and no `model_id` in the payload, when `handle_train_request` is called, then a UUID is auto-generated for `model_id`.
- **AC-H07 (REQ-H07) [P1]**: Given an invalid feature column specification in the payload, when `handle_train_request` is called, then feature column parsing returns an error and `400` is returned.
- **AC-H08 (REQ-H08) [P1]**: Given a non-numeric `epochs` value in the payload, when `handle_train_request` is called, then numeric parsing returns an error and `400` is returned.
- **AC-H09 (REQ-H09) [P1]**: Given `test_size=1.5` in the payload, when `handle_train_request` is called, then bounds validation returns an error and `400` is returned.
- **AC-H10 (REQ-H10) [P2]**: Given a training mode string with inconsistent casing, when `handle_train_request` is called, then it is normalized before the allowlist check.
- **AC-H11 (REQ-H11) [P1]**: Given `training_mode="unknown_mode"`, when `handle_train_request` is called, then it returns `400`.
- **AC-H12 (REQ-H12) [P1]**: Given an invalid hidden config (e.g., `num_hidden_layers=-1`), when `handle_train_request` is called, then `hidden_config_error` catches it and `400` is returned.
- **AC-H13 (REQ-H13) [P1]**: Given a valid payload, when training completes, then the resulting bundle is stored in the in-memory registry and `run_id` in the response references it.
- **AC-H14 (REQ-H14) [P1]**: Given `save_model=true`, when `handle_train_request` succeeds, then the bundle is written to disk and `model_path` is non-null in the response.
- **AC-H15 (REQ-H15) [P1]**: Given the training runtime raises an exception, when `handle_train_request` handles it, then it returns `(400, {"status": "error", …})` without re-raising.
- **AC-H16 (REQ-H16) [P1]**: Given no teacher source fields in the distill payload, when `handle_distill_request` is called, then it returns `400` before parsing any other parameters (teacher check at step 3).
- **AC-H17 (REQ-H17) [P1]**: Given `temperature=0` in the distill payload, when `handle_distill_request` is called, then bounds validation returns `400`.
- **AC-H18 (REQ-H18) [P1]**: Given `training_mode` in `PYTORCH_UNSUPPORTED_DISTILL_MODES`, when `handle_distill_request` is called, then it returns `400`.
- **AC-H19 (REQ-H19) [P1]**: Given a `teacher_run_id` that has expired from the registry, when `handle_distill_request` is called, then it returns `(404, {"status": "error", "error": "Teacher run not found or expired."})`.
- **AC-H20 (REQ-H20) [P1]**: Given a successful distillation, when the response is returned, then it includes `teacher_param_count`, `student_param_count`, `param_saved_count`, and `size_saved_percent`.
- **AC-H21 (REQ-H21) [P1]**: Given PyTorch raises `ModuleNotFoundError` on import, when `_runtime_trainer` is called, then it returns `(None, <error_string>)` without raising.
- **AC-H22 (REQ-H22) [P2]**: Given a model with exactly 1000 parameters, when `_parameter_count` is called, then it returns `1000`.
- **AC-H23 (REQ-H23) [P2]**: Given a model whose `state_dict` serialization fails, when `_serialized_model_size_bytes` is called, then it returns `None` without raising.
- **AC-H24 (REQ-H24) [P1]**: Given a distill payload with no `epochs` field, when numeric parsing runs, then `epochs` defaults to `60`.

---

## Invariants
- **INV-H01**: Validation errors are always returned before any runtime execution (training or distillation) is invoked.
- **INV-H02**: Every success response from either handler includes a `run_id` that references a live bundle in the in-memory registry.
- **INV-H03**: No exception from the training or distillation runtime propagates as an unhandled 500 — all exceptions are caught and returned as `400`.

---

## Edge Cases

- **EC-H01**: PyTorch is not installed on the server.
  Expected behavior: Every call to `handle_train_request` or `handle_distill_request` returns `(503, {"status": "error", "error": "..."})` without attempting any training.

- **EC-H02**: `save_model=false` in the payload.
  Expected behavior: No disk write occurs; `model_path` is `null` in the response; bundle is still stored in the in-memory registry.

- **EC-H03**: Teacher run ID was valid but TTL (900 seconds) expired between the train and distill calls.
  Expected behavior: `_load_in_memory_bundle` returns `None`; handler returns `(404, {"status": "error", "error": "Teacher run not found or expired."})`.

- **EC-H04**: Registry has reached capacity (128 bundles) and a new bundle is stored.
  Expected behavior: The oldest entry is evicted automatically; the new bundle is stored; no error is raised; `run_id` for the evicted bundle will return `None` on subsequent loads.

---

## Dependencies

| Dependency | What It Provides | Failure Mode | Fallback |
|---|---|---|---|
| `ai.ml.core.contracts` | `normalize_training_mode`, `validate_common_train_bounds`, `validate_common_distill_bounds` | `ImportError` | None — module unusable |
| `ai.ml.core.handler_utils` | `parse_train_numeric`, `parse_distill_numeric`, `compute_distill_stats`, `runtime_unavailable_response`, `hidden_config_error` | `ImportError` | None |
| `ai.ml.core.mode_catalog` | `PYTORCH_ALLOWED_TRAINING_MODES`, `PYTORCH_UNSUPPORTED_DISTILL_MODES` | `ImportError` | None |
| `ai.ml.core.request_helpers` | `resolve_data_target`, `parse_feature_columns`, `resolve_teacher_model_path` | `ImportError` | None |
| `ai.ml.core.artifacts` | `safe_file_size` | `ImportError` | None |
| `ai.ml.distill.InMemoryBundleRegistry` | TTL-bounded registry implementation | `ImportError` | None |
| `ai.ml.core.types.TrainingConfig`, `ModelBundle` | Config and bundle types | `AttributeError` on malformed payload | None — returns 400 |
| `.trainer` (lazy import) | Training runtime | `ModuleNotFoundError` → 503 | `_runtime_trainer()` returns `(None, err)` |

---

## Open Questions

None at this time.

---

## Traceability
- Requirement and acceptance trace matrix: [pytorch-backend-traceability.spec.md](../../../../traceability/pytorch-backend-traceability.spec.md#pt-handlers-001)
- Validation flow detail: [behavior.spec.md](./handlers.behavior.spec.md)
