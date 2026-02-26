# Errors Spec — ai/ml/frameworks/pytorch/handlers.py

Spec ID:      PT-HANDLERS-ERRORS-001
Version:      1.0
Last Edited:  2026-02-25T00:00:00Z
Hash:         sha256:20173cbb16e6d54a2b3acc9a2ca4f5c2fceede1db819e4bf82938589ed6791c8

## Purpose

Documents every error condition produced by `handle_train_request` and `handle_distill_request` — the HTTP status code, the response shape, the triggering condition, and which handler(s) produce it.

---

## Response Envelope

All error responses from both handlers share one shape:

```python
(status_code: int, {"status": "error", "error": "<message string>"})
```

There is no other error format. No nested objects, no error codes, no field-level detail.

---

## Error Registry

### 503 — Runtime Unavailable

| Field | Value |
|---|---|
| Status | `503` |
| Condition | PyTorch is not importable at request time (lazy import in `_runtime_trainer()` fails) |
| Message | Provided by `runtime_unavailable_response("PyTorch", str(exc))` |
| Handlers | `handle_train_request`, `handle_distill_request` |
| Validation step | Step 1 (first check in both chains) |

---

### 400 — Data Resolution Failure

| Field | Value |
|---|---|
| Status | `400` |
| Condition | `resolve_data_target(payload, resolve_dataset_path)` returns an error string |
| Message | Error string from `resolve_data_target` |
| Handlers | `handle_train_request`, `handle_distill_request` |
| Validation step | Step 2 |

---

### 400 — Teacher Source Missing *(distill only)*

| Field | Value |
|---|---|
| Status | `400` |
| Condition | None of `teacher_run_id`, `teacher_model_path`, `teacher_model_id` present in payload |
| Message | Fixed string from handler implementation |
| Handlers | `handle_distill_request` only |
| Validation step | Step 3 (distill chain) — fires before any parameter parsing |

---

### 400 — Feature Column Parse Error

| Field | Value |
|---|---|
| Status | `400` |
| Condition | `parse_feature_columns(payload)` returns an error |
| Message | Error string from `parse_feature_columns` |
| Handlers | `handle_train_request`, `handle_distill_request` |
| Validation step | Step 4 (train), Step 5 (distill) |

---

### 400 — Numeric Parse Error

| Field | Value |
|---|---|
| Status | `400` |
| Condition | `parse_train_numeric` or `parse_distill_numeric` returns an error (e.g., non-numeric `epochs`, `test_size`, `batch_size`, `learning_rate`) |
| Message | Error string from parser |
| Handlers | `handle_train_request` (`parse_train_numeric`), `handle_distill_request` (`parse_distill_numeric`) |
| Validation step | Step 5 (train), Step 6 (distill) |
| Note | Distill defaults `epochs=60` if omitted — no error for absent `epochs` |

---

### 400 — Bounds Validation Failure

| Field | Value |
|---|---|
| Status | `400` |
| Condition | Numeric parameters outside acceptable ranges (e.g., `test_size=1.5`, `epochs=0`, `temperature=0`, `alpha=1.5`) |
| Message | Error string from `validate_common_train_bounds` or `validate_common_distill_bounds` |
| Handlers | `handle_train_request` (`validate_common_train_bounds`), `handle_distill_request` (`validate_common_distill_bounds`) |
| Validation step | Step 6 (train), Step 7 (distill) |

---

### 400 — Training Mode Not Allowed

| Field | Value |
|---|---|
| Status | `400` |
| Condition | `training_mode` (after normalization) not in `PYTORCH_ALLOWED_TRAINING_MODES` (train) or in `PYTORCH_UNSUPPORTED_DISTILL_MODES` (distill) |
| Message | Error string from mode check |
| Handlers | `handle_train_request` (step 8), `handle_distill_request` (step 9) |
| Validation step | Step 8 (train), Step 9 (distill) |

---

### 400 — Hidden Config Invalid *(train only)*

| Field | Value |
|---|---|
| Status | `400` |
| Condition | `hidden_config_error(training_mode, hidden_dim, num_hidden_layers, dropout)` returns an error (e.g., invalid layer/dim combinations) |
| Message | Error string from `hidden_config_error` |
| Handlers | `handle_train_request` only |
| Validation step | Step 9 (train) — NOT called in distill chain |

---

### 404 — Teacher Run Not Found *(distill only)*

| Field | Value |
|---|---|
| Status | `404` |
| Condition | `teacher_run_id` was provided but `_load_in_memory_bundle(teacher_run_id)` returns `None` (expired TTL or evicted) |
| Message | `"Teacher run not found or expired."` (fixed string) |
| Handlers | `handle_distill_request` only |
| Validation step | Step 10 (distill) |

---

### 400 — Runtime Training/Distillation Exception

| Field | Value |
|---|---|
| Status | `400` |
| Condition | Any exception raised inside `train_model_from_file(...)` or `distill_model_from_file(...)` during execution |
| Message | `str(exc)` |
| Handlers | `handle_train_request`, `handle_distill_request` |
| Note | All runtime exceptions are caught — none propagate as unhandled 500 |

---

## Status Code Summary

| Code | Meaning in this context |
|------|------------------------|
| `200` | Request fully processed; bundle stored in registry |
| `400` | Validation failure or runtime exception during training/distillation |
| `404` | Teacher run ID not found or expired in registry |
| `503` | PyTorch runtime not available on this server |

`500` is never returned by these handlers — all exceptions are caught and converted to `400`.
