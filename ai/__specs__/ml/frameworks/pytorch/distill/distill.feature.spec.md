# ai/ml/frameworks/pytorch/distill.py

Spec ID:      PT-DISTILL-001
Version:      0.6
Last Edited:  2026-02-25T00:00:00Z
Hash:         sha256:98cf3524512c710789d2e6a61bc4b94d250dafbb3555799a0d22f77091a88703

## Goals
- Specify the knowledge-distillation pipeline that trains a smaller student model using a pre-trained teacher model.

## Scope
- **In scope**: `distill_model_from_file` function — teacher resolution, student construction, distillation loss, and output contract.
- **Out of scope**: Model architecture details (see `models/models.feature.spec.md`); handler-level validation and teacher registry lookup (see `handlers/handlers.feature.spec.md`).

---

## Function Contract

### `distill_model_from_file(data_path, cfg, teacher_path?, teacher_bundle?, sheet_name?, exclude_columns?, date_columns?, device?, temperature?, alpha?, student_hidden_dim?, student_num_hidden_layers?, student_dropout?) → (ModelBundle, Metrics)`

Trains a smaller student neural network by minimizing a combination of the hard label loss (against ground-truth labels) and a soft knowledge-distillation loss (derived from a pre-trained teacher's output distribution). The caller supplies the teacher either as a loaded bundle object or a file path. The student's architecture defaults are automatically derived from the teacher's size. Returns the student's `ModelBundle` — with all preprocessing artifacts inherited from the teacher — and evaluation metrics.

#### Execution Order

The actual execution order within this function is:
1. Seed `random`, `numpy`, `torch`
2. Validate `temperature` and `alpha`
3. Resolve teacher (`teacher_bundle` or `teacher_path`)
4. Resolve and validate task
5. Validate `exclude_columns`
6. Load and prepare data using teacher's artifacts
7. Build student model
8. Run distillation loop
9. Return `(ModelBundle, Metrics)`

#### Teacher Resolution

- **REQ-KD01**: If `teacher_bundle` is provided, use it directly (no disk load).
- **REQ-KD02**: Else if `teacher_path` is provided, load via `serialization.load_bundle`.
- **REQ-KD03**: If neither is provided, raise `ValueError("teacher_path or teacher_bundle is required.")`.

#### Validation

- **REQ-KD04**: Raises `ValueError("temperature must be > 0")` if `temperature <= 0`.
- **REQ-KD05**: Raises `ValueError("alpha must be between 0 and 1")` if `alpha` is outside `[0, 1]`.
- **REQ-KD06**: When `cfg.task == "auto"`, task is derived from the teacher bundle's `.task` field.
- **REQ-KD06a**: `infer_task(...)` is never called in distillation — teacher task is the only source when `cfg.task == "auto"`.
- **REQ-KD07**: Raises `ValueError("Requested task does not match teacher task.")` if explicit `cfg.task` conflicts with teacher's task.
- **REQ-KD08**: Raises `ValueError("target_column cannot be excluded")` if `cfg.target_column` is in `exclude_columns`.

#### Data Pipeline

- **REQ-KD09**: Seeds `random`, `numpy`, `torch` from `cfg.random_seed` before validation and teacher resolution.
- **REQ-KD10**: Data is loaded via `load_tabular_file`, date-expanded, and column-excluded identically to `trainer.py`.
- **REQ-KD11**: Uses the **teacher's** fitted `vectorizer` and `scaler` for all feature transformation (no re-fitting).
- **REQ-KD12**: Uses the **teacher's** `label_encoder` for classification label decoding.
  - Raises `ValueError("Teacher model is missing a label encoder.")` if it is `None` for a classification task.
- **REQ-KD13**: Uses the **teacher's** `target_scaler` for regression targets. If `None`, targets are left unscaled.
- **REQ-KD14**: Classification train/test split is stratified when there are >1 unique labels.

#### Student Architecture Defaults

| Parameter | Default when not specified |
|-----------|--------------------------|
| `student_hidden_dim` | `max(16, teacher_hidden_dim // 2)` |
| `student_num_hidden_layers` | `max(1, teacher_num_hidden_layers - 1)` |
| `student_dropout` | `min(0.5, teacher_dropout + 0.05)` |

- **REQ-KD15**: Student is built via `build_model` using `cfg.training_mode` and the resolved student hyperparameters.

#### Distillation Loop

- **REQ-KD16**: Teacher is set to `eval()` mode and its outputs are computed inside `torch.no_grad()`.
- **REQ-KD17**: Classification distillation loss:
  - `hard_loss = CrossEntropyLoss(student_logits, y_labels)`
  - `soft_loss = KLDivLoss(log_softmax(student_logits / T, dim=1), softmax(teacher_logits / T, dim=1)) × T²`
- **REQ-KD18**: Regression distillation loss:
  - `hard_loss = MSELoss(student_out, y_targets)`
  - `soft_loss = MSELoss(student_out, teacher_out)`
- **REQ-KD19**: Combined loss: `(alpha × hard_loss) + ((1 - alpha) × soft_loss)`.
- **REQ-KD20**: Optimizer is `Adam(lr=cfg.learning_rate, weight_decay=1e-4)`.
- **REQ-KD21**: Batches with fewer than 2 samples are skipped (no weight update).
- **REQ-KD22**: Raises `ValueError` if `update_steps == 0` after all epochs complete (no valid batches).

#### Output

- **REQ-KD23**: Evaluation metrics: `accuracy` for classification, `rmse` in original target scale for regression.
- **REQ-KD24**: Returns `(ModelBundle, Metrics)`.
- **REQ-KD25**: The student `ModelBundle` inherits `vectorizer`, `scaler`, `label_encoder`, `target_scaler`, and `class_names` directly from the teacher bundle.
- **REQ-KD26**: `student_bundle.feature_medians` is set to the teacher's `feature_medians` if available, otherwise the newly computed `col_medians` from the current dataset.

---

## Acceptance Criteria

- **AC-KD01 (REQ-KD01) [P1]**: Given a pre-loaded `teacher_bundle`, when `distill_model_from_file` is called, then the teacher bundle is used directly without a disk load.
- **AC-KD02 (REQ-KD02) [P1]**: Given a valid `teacher_path` and no `teacher_bundle`, when `distill_model_from_file` is called, then the teacher is loaded from that path via `serialization.load_bundle`.
- **AC-KD03 (REQ-KD03) [P1]**: Given neither `teacher_path` nor `teacher_bundle`, when `distill_model_from_file` is called, then it raises `ValueError("teacher_path or teacher_bundle is required.")`.
- **AC-KD04 (REQ-KD04) [P1]**: Given `temperature=0`, when `distill_model_from_file` is called, then it raises `ValueError("temperature must be > 0")`.
- **AC-KD05 (REQ-KD05) [P1]**: Given `alpha=1.5`, when `distill_model_from_file` is called, then it raises `ValueError("alpha must be between 0 and 1")`.
- **AC-KD06 (REQ-KD06) [P1]**: Given `cfg.task="auto"` and a teacher trained on `"classification"`, when `distill_model_from_file` is called, then the student trains as a classification model.
- **AC-KD06a (REQ-KD06a) [P1]**: Given `cfg.task="auto"`, when `distill_model_from_file` is called, then `infer_task` is never invoked; task is taken from the teacher bundle only.
- **AC-KD07 (REQ-KD07) [P1]**: Given `cfg.task="regression"` and a teacher trained on `"classification"`, when `distill_model_from_file` is called, then it raises `ValueError("Requested task does not match teacher task.")`.
- **AC-KD08 (REQ-KD08) [P1]**: Given the target column listed in `exclude_columns`, when `distill_model_from_file` is called, then it raises `ValueError("target_column cannot be excluded")`.
- **AC-KD09 (REQ-KD09) [P1]**: Given the same `cfg.random_seed`, when `distill_model_from_file` is called twice on identical data, then both runs produce identical student weights.
- **AC-KD11 (REQ-KD11) [P1]**: Given a teacher with a fitted vectorizer, when the student trains, then the teacher's vectorizer is used for all feature transforms and no new vectorizer is fitted.
- **AC-KD12 (REQ-KD12) [P1]**: Given a classification task and a teacher bundle where `label_encoder is None`, when `distill_model_from_file` is called, then it raises `ValueError("Teacher model is missing a label encoder.")`.
- **AC-KD14 (REQ-KD14) [P1]**: Given a classification dataset with ≥2 unique labels, when the distillation data pipeline runs, then the train/test split is stratified.
- **AC-KD15 (REQ-KD15) [P1]**: Given no explicit student hyperparameters and a teacher with `hidden_dim=128`, when student defaults are resolved, then `student_hidden_dim == max(16, 64) == 64`.
- **AC-KD16 (REQ-KD16) [P1]**: Given any distillation run, when teacher outputs are computed, then they are computed inside `torch.no_grad()` and the teacher remains in `eval()` throughout.
- **AC-KD17 (REQ-KD17) [P1]**: Given a classification task with `temperature=2.0` and `alpha=0.5`, when a batch loss is computed, then `soft_loss = KLDiv(log_softmax(student/2), softmax(teacher/2)) × 4`.
- **AC-KD19 (REQ-KD19) [P1]**: Given `alpha=0.3`, when combined loss is computed for any batch, then `loss == 0.3 × hard_loss + 0.7 × soft_loss`.
- **AC-KD21 (REQ-KD21) [P1]**: Given a batch with only 1 sample, when the distillation loop processes it, then that batch is skipped and no weight update occurs.
- **AC-KD22 (REQ-KD22) [P1]**: Given a dataset where no valid batches (≥2 samples) exist, when distillation completes all epochs, then `ValueError` is raised because `update_steps == 0`.
- **AC-KD24 (REQ-KD24) [P1]**: Given a successful distillation run, when the function returns, then the result is a `(ModelBundle, Metrics)` tuple.
- **AC-KD25 (REQ-KD25) [P1]**: Given a teacher with a fitted scaler, when the student bundle is returned, then `student_bundle.scaler` is the same object as `teacher_bundle.scaler`.
- **AC-KD26 (REQ-KD26) [P2]**: Given a teacher with non-None `feature_medians`, when the student bundle is returned, then `student_bundle.feature_medians` equals the teacher's `feature_medians`.

---

## Invariants
- **INV-KD01**: Teacher model is never modified (weights, mode, or attributes) during distillation.
- **INV-KD02**: Student model is always in `eval()` mode when returned.
- **INV-KD03**: Preprocessing artifacts (`vectorizer`, `scaler`, `label_encoder`, `target_scaler`) in the returned bundle always come from the teacher, ensuring prediction compatibility with the original training schema.
- **INV-KD04**: Deterministic seeding (`random`, `numpy`, `torch`) is applied before any stochastic operation, including before temperature/alpha validation and teacher resolution.

---

## Edge Cases

- **EC-KD01**: `alpha=0.0` (pure soft loss).
  Expected behavior: Combined loss = `0 × hard_loss + 1 × soft_loss`; training is entirely driven by mimicking the teacher distribution; valid run with no error.

- **EC-KD02**: `alpha=1.0` (pure hard loss).
  Expected behavior: Combined loss = `1 × hard_loss + 0 × soft_loss`; distillation degrades to standard supervised training; valid run with no error.

- **EC-KD03**: Student and teacher have the same architecture dimensions.
  Expected behavior: Distillation proceeds normally; no compression occurs; `param_saved_count=0` in handler response.

- **EC-KD04**: Teacher was trained on a different feature schema than the current dataset rows.
  Expected behavior: Teacher vectorizer is applied as-is (authoritative); feature mismatch may cause dimension errors or degraded quality; `distill_model_from_file` does not validate schema compatibility — that is the caller's responsibility.

---

## Dependencies

| Dependency | What It Provides | Failure Mode | Fallback |
|---|---|---|---|
| `torch` | Tensor ops, `no_grad`, Adam optimizer | `ImportError` | None — module unusable |
| `numpy` | Array operations | `ImportError` | None |
| `sklearn.model_selection.train_test_split` | Stratified splitting | `ImportError` | None |
| `ai.ml.frameworks.pytorch.models.build_model` | Student model construction | Raises on invalid mode | None |
| `ai.ml.frameworks.pytorch.models.model_hidden_dim`, `model_num_hidden_layers`, `model_dropout` | Teacher hyperparameter introspection for student defaults | Returns defaults on unknown model | Defaults used for student |
| `ai.ml.frameworks.pytorch.serialization.load_bundle` | Loading teacher from disk | `FileNotFoundError`, `RuntimeError` | None — propagates to caller |
| `ai.ml.core.types.TrainingConfig`, `Metrics`, `ModelBundle` | Config and return types | `AttributeError` on invalid config | None |
| `ai.ml.core.preprocessing.impute_train_test_non_finite` | NaN/Inf imputation | Propagates exception | None |
| `ai.ml.file_util` | `load_tabular_file`, `expand_date_columns` | `FileNotFoundError`, `ValueError` | None — propagates to caller |
| `ai.ml.ml_util` | `coerce_value`, task utilities | Raises on unsupported input | None |

---

## Open Questions

None at this time.

---

## Traceability
- Requirement and acceptance trace matrix: [pytorch-backend-traceability.spec.md](../../../../traceability/pytorch-backend-traceability.spec.md#pt-distill-001)
