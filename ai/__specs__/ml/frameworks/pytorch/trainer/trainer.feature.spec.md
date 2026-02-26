# ai/ml/frameworks/pytorch/trainer.py

Spec ID:      PT-TRAINER-001
Version:      0.6
Last Edited:  2026-02-25T00:00:00Z
Hash:         sha256:244e22f93a592875155f0e8d82c1e6fb83b6a7d56a5428c3d0607c09e6b1fd93

## Goals
- Specify the end-to-end training and prediction contracts exposed by the trainer module.
- Document the tree-teacher distillation training-loop variant.

## Scope
- **In scope**: `train_model_from_file`, `predict_rows`, `load_bundle`; training loop behavior including per-mode criterion selection and tree-teacher distillation.
- **Out of scope**: Model architecture details (see `models/models.feature.spec.md`); preprocessing pipeline (see `data/data.feature.spec.md`); knowledge-distillation from a teacher bundle (see `distill/distill.feature.spec.md`).

---

## Public API

### `train_model_from_file(data_path, cfg, sheet_name?, exclude_columns?, date_columns?, device?) → (ModelBundle, Metrics)`

Loads tabular data from disk, runs the full preprocessing and training pipeline, and returns the fitted model together with evaluation metrics. This is the primary entry point for all supervised training workflows supported by the PyTorch backend.

#### Requirements

- **REQ-T01**: Seeds `random`, `numpy`, and `torch` using `cfg.random_seed` before any stochastic operation.
- **REQ-T02**: Loads data via `load_tabular_file`, expands date columns via `expand_date_columns`, and applies column exclusions.
- **REQ-T03**: Raises `ValueError("target_column cannot be excluded")` if `cfg.target_column` appears in `exclude_columns`.
- **REQ-T04**: Task is inferred via `infer_task(y_raw)` when `cfg.task == "auto"`, otherwise uses the explicit value.
- **REQ-T05**: Raises `ValueError` if `training_mode` is `"imbalance_aware"` or `"calibrated_classifier"` and the resolved task is not `"classification"`.
- **REQ-T06**: Delegates preprocessing to `prepare_tensors`, passing `task`, `cfg`, and the loaded rows.
- **REQ-T07**: Device is taken from the `device` argument if provided; otherwise auto-detects CUDA (`torch.cuda.is_available()`), falling back to CPU.
- **REQ-T08**: Builds model via `build_model` and moves it to the selected device.

#### Loss Criteria by Mode

| Task | Training Mode | Criterion |
|------|--------------|-----------|
| classification | `imbalance_aware` | `CrossEntropyLoss(weight=compute_class_weights(…))` |
| classification | `calibrated_classifier` | `CrossEntropyLoss(label_smoothing=0.05)` |
| classification | all others | `CrossEntropyLoss()` |
| regression | any | `MSELoss()` |

- **REQ-T09**: For `training_mode="tree_teacher_distillation"`:
  - Classification: RF classifier (`n_estimators=120, max_depth=8`) trained as tree teacher; soft loss = KL-divergence with `temperature=2.0`; combined batch loss = `0.5 × hard + 0.5 × soft`.
  - Regression: RF regressor (`n_estimators=120, max_depth=10`); soft loss = `MSELoss(student, tree_preds)`; combined batch loss = `0.5 × hard + 0.5 × soft`.
- **REQ-T10**: Optimizer is `Adam(lr=cfg.learning_rate, weight_decay=1e-4)`.
- **REQ-T11**: Iterates `cfg.epochs` epochs; each epoch processes mini-batches of size `cfg.batch_size`.
- **REQ-T12**: Batches with fewer than 2 samples are skipped (no weight update performed).
- **REQ-T13**: Raises `ValueError` if `update_steps == 0` after all epochs (no valid batches processed).
- **REQ-T14**: After training, model is set to `eval()` mode; returned model has `model.training == False`.
- **REQ-T15**: Classification metric: `accuracy` (fraction of correct predictions on test set). Regression metric: `rmse` in original target scale (inverse-transformed via `target_scaler`).
- **REQ-T16**: Returns `(ModelBundle, Metrics)`. `ModelBundle` contains the trained model plus all fitted preprocessing artifacts from `prepare_tensors`.

---

### `predict_rows(bundle, rows, device?) → list[Any]`

Applies a fitted `ModelBundle` to new rows of raw feature data and returns decoded predictions. For classification returns class label strings; for regression returns scalar floats in the original (unscaled) target units.

- **REQ-T17**: Returns `[]` immediately for empty `rows` input.
- **REQ-T18**: For each row, applies `coerce_value` to each cell value, then `vectorizer.transform`, then non-finite imputation using `bundle.feature_medians` (falls back to zero-fill if shape mismatch), then `scaler.transform`.
- **REQ-T19**: Model is moved to device and set to `eval()` before inference; no gradients computed.
- **REQ-T20**: Classification: returns `label_encoder.inverse_transform(argmax(logits, dim=1)).tolist()`.
- **REQ-T21**: Regression: returns `target_scaler.inverse_transform(preds).squeeze(1).tolist()`. If `target_scaler` is `None`, returns raw model outputs as a list.

---

### `load_bundle(path, map_location?) → ModelBundle`

Deserializes a previously saved `ModelBundle` from disk and returns it ready for prediction or further distillation.

- **REQ-T22**: Thin wrapper around `serialization.load_bundle(path, map_location)`.
- **REQ-T23**: Re-exports `save_bundle` from `serialization` unchanged (available as `trainer.save_bundle`).

---

## Module Exports (`__all__`)

```
Metrics, ModelBundle, TrainingConfig,
distill_model_from_file, load_bundle, predict_rows, save_bundle, train_model_from_file
```

---

## Acceptance Criteria

- **AC-T01 (REQ-T01) [P1]**: Given the same `cfg.random_seed`, when `train_model_from_file` is called twice on identical data, then both runs produce identical trained weights.
- **AC-T02 (REQ-T02) [P2]**: Given a CSV path and a list of date columns, when `train_model_from_file` is called, then the loaded DataFrame contains expanded date features in place of the original date columns.
- **AC-T03 (REQ-T03) [P1]**: Given `cfg.target_column` listed in `exclude_columns`, when `train_model_from_file` is called, then it raises `ValueError("target_column cannot be excluded")`.
- **AC-T04 (REQ-T04) [P1]**: Given `cfg.task="auto"` and a binary target column, when `train_model_from_file` is called, then the inferred task is `"classification"`.
- **AC-T05 (REQ-T05) [P1]**: Given `training_mode="imbalance_aware"` and a regression dataset, when `train_model_from_file` is called, then it raises `ValueError`.
- **AC-T06 (REQ-T06) [P1]**: Given any valid configuration, when `train_model_from_file` is called, then the returned bundle contains a fitted vectorizer and scaler from `prepare_tensors`.
- **AC-T07 (REQ-T07) [P2]**: Given no `device` argument and no CUDA available, when `train_model_from_file` is called, then training proceeds on CPU without error.
- **AC-T08 (REQ-T08) [P2]**: Given a valid configuration, when `train_model_from_file` returns, then `bundle.model` is on the selected device.
- **AC-T09 (REQ-T09) [P1]**: Given `training_mode="tree_teacher_distillation"` and a classification task, when a training batch runs, then batch loss equals `0.5 × hard_loss + 0.5 × KL_soft_loss`.
- **AC-T10 (REQ-T10) [P1]**: Given any configuration, when the optimizer is created, then it is `Adam` with `weight_decay=1e-4`.
- **AC-T11 (REQ-T11) [P2]**: Given `cfg.epochs=5` and a dataset larger than `cfg.batch_size`, when training runs, then 5 epochs are completed.
- **AC-T12 (REQ-T12) [P1]**: Given a batch with only 1 sample, when the training loop processes it, then that batch is skipped and no weight update occurs.
- **AC-T13 (REQ-T13) [P1]**: Given a dataset where all batches have <2 samples, when training completes, then `ValueError` is raised because `update_steps == 0`.
- **AC-T14 (REQ-T14) [P1]**: Given a successfully trained model, when `train_model_from_file` returns, then `bundle.model.training == False`.
- **AC-T15 (REQ-T15) [P1]**: Given a regression task, when metrics are returned, then `metrics.test_metric_name == "rmse"` and the value is in the original target scale (not scaled).
- **AC-T16 (REQ-T16) [P1]**: Given a successful training run, when the function returns, then the result is a `(ModelBundle, Metrics)` tuple with a non-None model in the bundle.
- **AC-T17 (REQ-T17) [P1]**: Given an empty list as `rows`, when `predict_rows` is called, then it returns `[]` without error.
- **AC-T18 (REQ-T18) [P1]**: Given rows with NaN values, when `predict_rows` is called, then the imputed values use `bundle.feature_medians` (not zero-fill) when shapes match.
- **AC-T19 (REQ-T19) [P1]**: Given a bundle and valid rows, when `predict_rows` is called, then the model is in `eval()` mode during inference.
- **AC-T20 (REQ-T20) [P1]**: Given a classification bundle and a valid input row, when `predict_rows` is called, then it returns the decoded class label string (not an integer index).
- **AC-T21 (REQ-T21) [P1]**: Given a regression bundle and a valid input row, when `predict_rows` is called, then it returns a float in the original target scale.
- **AC-T22 (REQ-T22) [P2]**: Given a path to a saved bundle, when `trainer.load_bundle(path)` is called, then the result is identical to calling `serialization.load_bundle(path)` directly.
- **AC-T23 (REQ-T23) [P2]**: Given the trainer module, when `trainer.save_bundle` is accessed, then it is the same callable object as `serialization.save_bundle`.

---

## Invariants
- **INV-T01**: Deterministic seeding (`random`, `numpy`, `torch`) is applied before any stochastic operation.
- **INV-T02**: Model is always in `eval()` mode when returned from `train_model_from_file`.
- **INV-T03**: `metrics.task` matches the actual task used for training (not `"auto"`).
- **INV-T04**: `predict_rows` never mutates the input `rows` list or its contents.

---

## Edge Cases

- **EC-T01**: Dataset has fewer rows than `cfg.batch_size` after train/test split.
  Expected behavior: Training proceeds with a single batch per epoch (or zero batches if <2 rows); if zero valid batches, `ValueError` raised per REQ-T13.

- **EC-T02**: Single-class classification dataset (all labels identical).
  Expected behavior: Stratification disabled (single unique label); training proceeds; accuracy = 1.0 trivially on test set.

- **EC-T03**: `bundle.feature_medians` shape does not match the vectorized input shape during prediction.
  Expected behavior: Imputation falls back to zero-fill for the entire row; prediction continues without error.

---

## Dependencies

| Dependency | What It Provides | Failure Mode | Fallback |
|---|---|---|---|
| `torch` | Tensor ops, CUDA detection, `no_grad` | `ImportError` | None — module unusable |
| `numpy` | Array operations | `ImportError` | None |
| `sklearn.ensemble.RandomForestClassifier` | Tree teacher for classification | `ImportError` | None — tree_teacher_distillation mode fails |
| `sklearn.ensemble.RandomForestRegressor` | Tree teacher for regression | `ImportError` | None — tree_teacher_distillation mode fails |
| `ai.ml.frameworks.pytorch.data.prepare_tensors` | Preprocessing pipeline | Raises on invalid input | None — propagates to caller |
| `ai.ml.frameworks.pytorch.models.build_model` | Model construction | Raises on invalid mode | None |
| `ai.ml.frameworks.pytorch.serialization.save_bundle`, `load_bundle` | Disk persistence | `IOError` on disk write | None — `save_model=false` avoids disk write |
| `ai.ml.frameworks.pytorch.distill.distill_model_from_file` | Distillation entry point (re-exported) | Raises on invalid config | None |
| `ai.ml.core.types.TrainingConfig`, `Metrics`, `ModelBundle` | Config and return types | `AttributeError` on invalid config | None |
| `ai.ml.file_util` | `load_tabular_file`, `expand_date_columns` | `FileNotFoundError`, `ValueError` on bad file | None — propagates to caller |
| `ai.ml.ml_util` | `infer_task`, `coerce_value` | Raises on unsupported input | None |

---

## Open Questions

None at this time.

---

## Traceability
- Requirement and acceptance trace matrix: [pytorch-backend-traceability.spec.md](../../../../traceability/pytorch-backend-traceability.spec.md#pt-trainer-001)
