# Behavior Spec — ai/ml/frameworks/pytorch/handlers.py

Spec ID:      PT-HANDLERS-BEHAVIOR-001
Version:      1.0
Last Edited:  2026-02-25T00:00:00Z
Hash:         sha256:ef6cb912de971f5650401997390ff58b16aa4820b8f6c9fd8013bcc1fa1b2845

## Purpose

Documents the ordered, branching execution behavior of `handle_train_request` and `handle_distill_request`. These handlers have non-trivial validation chains where the step order matters: a later step must never execute if an earlier step fails. This file is the authoritative reference for writing ordered integration tests.

For requirement contracts (what each step checks), see `handlers.feature.spec.md`.

---

## Train Validation Chain

```
handle_train_request(payload)
│
├─ [1] PyTorch importable?
│   NO  → return (503, error)
│   YES → continue
│
├─ [2] resolve_data_target(payload)
│   FAIL → return (400, error)
│   OK   → data_path
│
├─ [3] parse save_model flag + model_id
│   (always succeeds; model_id auto-generated if absent)
│
├─ [4] parse_feature_columns(payload)
│   FAIL → return (400, error)
│   OK   → exclude_columns, date_columns
│
├─ [5] parse_train_numeric(payload)
│   FAIL → return (400, error)
│   OK   → test_size, epochs, batch_size, learning_rate, hidden_dim, num_hidden_layers, dropout
│
├─ [6] validate_common_train_bounds(test_size, epochs, batch_size, learning_rate)
│   FAIL → return (400, error)
│   OK   → continue
│
├─ [7] normalize_training_mode(training_mode)
│   (always succeeds)
│
├─ [8] training_mode in PYTORCH_ALLOWED_TRAINING_MODES?
│   NO  → return (400, error)
│   YES → continue
│
├─ [9] hidden_config_error(training_mode, hidden_dim, num_hidden_layers, dropout)
│   FAIL → return (400, error)
│   OK   → continue
│
├─ [10] build TrainingConfig + call train_model_from_file(...)
│   RAISES → return (400, error)   ← all runtime exceptions caught here
│   OK     → (bundle, metrics)
│
└─ [11] store bundle in registry + optional disk save
         return (200, success_response)
```

---

## Distill Validation Chain

```
handle_distill_request(payload)
│
├─ [1] PyTorch importable?
│   NO  → return (503, error)
│   YES → continue
│
├─ [2] resolve_data_target(payload)
│   FAIL → return (400, error)
│   OK   → data_path
│
├─ [3] teacher_run_id OR teacher_model_path OR teacher_model_id present?
│   NO  → return (400, error)          ← teacher check is step 3, NOT step 10
│   YES → continue
│
├─ [4] parse save_model flag + model_id
│   (always succeeds)
│
├─ [5] parse_feature_columns(payload)
│   FAIL → return (400, error)
│   OK   → exclude_columns, date_columns
│
├─ [6] parse_distill_numeric(payload, default_epochs=60)
│   FAIL → return (400, error)
│   OK   → epochs=60 if absent, plus temperature, alpha, student dims
│
├─ [7] validate_common_distill_bounds(...)
│   FAIL → return (400, error)
│   OK   → continue
│
├─ [8] normalize_training_mode(training_mode)
│   (always succeeds)
│
├─ [9] training_mode NOT in PYTORCH_UNSUPPORTED_DISTILL_MODES?
│   IN LIST → return (400, error)
│   OK      → continue
│
│   NOTE: hidden_config_error is NOT called in the distill chain.
│
├─ [10] teacher_run_id provided → _load_in_memory_bundle(teacher_run_id)
│   NOT FOUND/EXPIRED → return (404, error)
│   FOUND             → teacher_bundle
│
│   teacher_model_path/teacher_model_id → resolved via resolve_teacher_model_path(...)
│
├─ [11] call distill_model_from_file(...)
│   RAISES → return (400, error)
│   OK     → (student_bundle, metrics)
│
└─ [12] compute_distill_stats + store bundle + optional disk save
         return (200, distill_success_response)
```

---

## Key Behavioral Differences: Train vs Distill

| Aspect | Train | Distill |
|--------|-------|---------|
| Teacher source check | N/A | Step 3 (before parameter parsing) |
| Numeric parsing | `parse_train_numeric` | `parse_distill_numeric` (default_epochs=60) |
| Bounds validation | `validate_common_train_bounds` | `validate_common_distill_bounds` |
| `hidden_config_error` | Called (step 9) | NOT called |
| Mode allowlist | `PYTORCH_ALLOWED_TRAINING_MODES` | NOT in `PYTORCH_UNSUPPORTED_DISTILL_MODES` |
| Registry lookup | Not applicable | Step 10 (404 if teacher_run_id expired) |
| Response fields | Standard | Standard + compression stats |

---

## Registry Behavior

```
_BUNDLE_REGISTRY: InMemoryBundleRegistry[ModelBundle]
  ttl_seconds = 900    (15 minutes)
  max_items   = 128

Store:
  _store_in_memory_bundle(bundle) → run_id (uuid string)
  Eviction: when max_items reached, oldest entry is evicted first (FIFO)

Load:
  _load_in_memory_bundle(run_id) → ModelBundle | None
  Returns None if:
    - run_id was never stored
    - run_id TTL has expired (>900s since store)
    - run_id was evicted due to capacity
```

---

## Error Response Shape (all error paths)

All error responses from both handlers share the same shape:

```python
(status_code: int, {"status": "error", "error": "<message string>"})
```

- `503` — PyTorch not importable
- `404` — teacher_run_id not found or expired
- `400` — all other validation failures and runtime exceptions

No other status codes are returned by these handlers.

---

## Success Response Shape

### Train (200)

```python
{
  "status": "ok",
  "run_id": str,           # references stored bundle in registry
  "model_id": str | None,  # None if save_model=false
  "model_path": str | None,
  "metrics": {
    "task": str,
    "train_loss": float,
    "test_loss": float,
    "test_metric_name": str,   # "accuracy" or "rmse"
    "test_metric_value": float
  }
}
```

### Distill (200)

All train response fields, plus:

```python
{
  "teacher_input_dim": int,
  "teacher_output_dim": int,
  "student_input_dim": int,
  "student_output_dim": int,
  "teacher_model_size_bytes": int | None,
  "student_model_size_bytes": int | None,
  "size_saved_bytes": int | None,
  "size_saved_percent": float | None,
  "teacher_param_count": int,
  "student_param_count": int,
  "param_saved_count": int,
  "param_saved_percent": float | None
}
```
