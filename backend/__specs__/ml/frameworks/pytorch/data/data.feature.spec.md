# ai/ml/frameworks/pytorch/data.py

Spec ID:      PT-DATA-001
Version:      0.5
Last Edited:  2026-02-25T00:00:00Z
Hash:         sha256:91809e01e92adfc329ad4b757ff2e7f197d9f4f24e8187881cceea5a09511309

## Goals
- Define the contract for `prepare_tensors`, the sole public function responsible for converting raw feature dicts and labels into train/test `torch.Tensor` pairs with fitted preprocessing artifacts.

## Scope
- **In scope**: `prepare_tensors` function, its return tuple, and preprocessing pipeline.
- **Out of scope**: File-loading logic (see `file_util`); model construction (see `models/models.feature.spec.md`).

---

## Function Contract

### `prepare_tensors(x_rows, y_raw, task, cfg) → Tuple[12]`

Converts raw feature dicts and label values into train/test PyTorch tensor pairs, fitting all preprocessing artifacts (vectorizer, scaler, encoders) on training data only. Returns the tensors alongside the fitted artifacts needed for later prediction and serialization.

#### Signature
```python
def prepare_tensors(
    x_rows: list[dict[str, Any]],
    y_raw: list[Any],
    task: TaskType,          # "classification" | "regression"
    cfg: TrainingConfig,
) -> tuple[
    Tensor, Tensor,          # x_train, x_test
    Tensor, Tensor,          # y_train, y_test
    DictVectorizer,          # fitted vectorizer
    StandardScaler,          # fitted feature scaler
    np.ndarray,              # col_medians from imputation
    LabelEncoder | None,     # fitted encoder (classification only)
    StandardScaler | None,   # target_scaler (regression only)
    int,                     # input_dim
    int,                     # output_dim
    list[str] | None,        # class_names (classification) or None
]
```

#### Requirements

- **REQ-D01**: Data is split via `train_test_split` with `test_size=cfg.test_size` and `random_state=cfg.random_seed`.
- **REQ-D02**: For classification, stratified splitting is used when there are >1 unique label values.
- **REQ-D03**: Features are vectorized using `DictVectorizer(sparse=False)` fit on train rows only; test rows use `transform`.
- **REQ-D04**: Non-finite values (NaN, Inf) are imputed via `impute_train_test_non_finite` after vectorization.
- **REQ-D05**: Features are scaled using `StandardScaler` fit on train set only; test set uses `transform`.
- **REQ-D06**: For `task="classification"`:
  - Labels encoded via `LabelEncoder` fit on train labels; applied to test labels.
  - `output_dim = len(encoder.classes_)`.
  - `class_names = [str(c) for c in encoder.classes_]`.
  - y tensors have `dtype=torch.long`.
- **REQ-D07**: For `task="regression"`:
  - Labels cast to `float32`, reshaped to `(-1, 1)`, scaled via `StandardScaler`, then flattened before tensor creation.
  - `output_dim = 1`, `class_names = None`, `label_encoder = None`.
  - y tensors have `dtype=torch.float32` and shape `(N, 1)`.
- **REQ-D08**: `input_dim` is set to `x_train.shape[1]`.
- **REQ-D09**: x tensors are always `dtype=torch.float32`.

---

## Acceptance Criteria

- **AC-D01 (REQ-D01) [P1]**: Given a dataset of N rows with `test_size=0.2`, when `prepare_tensors` is called, then `x_train` has ≈0.8×N rows and `x_test` has ≈0.2×N rows.
- **AC-D02 (REQ-D02) [P1]**: Given a classification dataset with ≥2 unique labels and `test_size=0.2`, when `prepare_tensors` is called, then the label distribution in the test split is proportionally similar to the full dataset (stratified split).
- **AC-D03 (REQ-D03) [P1]**: Given feature rows processed by `prepare_tensors`, when the returned vectorizer is called with `.transform` on the same rows, then it produces identical output without refitting.
- **AC-D04 (REQ-D04) [P1]**: Given a feature matrix containing NaN values, when `prepare_tensors` is called, then the returned x tensors contain no NaN or Inf values.
- **AC-D05 (REQ-D05) [P1]**: Given a training split processed by `prepare_tensors`, when the returned scaler is inspected, then `scaler.mean_` and `scaler.var_` match training-set statistics only (not test-set statistics).
- **AC-D06 (REQ-D06) [P1]**: Given a classification task with K unique classes, when `prepare_tensors` is called, then `output_dim == K`, `len(class_names) == K`, and `y_train.dtype == torch.long`.
- **AC-D07 (REQ-D07) [P1]**: Given a regression task, when `prepare_tensors` is called, then `output_dim == 1`, `class_names is None`, `label_encoder is None`, `y_train.shape == (N_train, 1)`, and `y_train.dtype == torch.float32`.
- **AC-D08 (REQ-D08) [P2]**: Given any dataset, when `prepare_tensors` is called, then `input_dim == x_train.shape[1]`.
- **AC-D09 (REQ-D09) [P2]**: Given any dataset, when `prepare_tensors` is called, then both `x_train.dtype` and `x_test.dtype` equal `torch.float32`.

---

## Invariants
- **INV-D01**: Vectorizer and scaler are fit exclusively on training data — never on test data.
- **INV-D02**: All returned tensors are dense (no sparse tensors).
- **INV-D03**: `input_dim == x_train.shape[1] == x_test.shape[1]`.
- **INV-D04**: `len(y_train) + len(y_test) == len(y_raw)`.

---

## Edge Cases

- **EC-D01**: Single unique label in classification dataset.
  Expected behavior: `stratify=None` — split proceeds without stratification; no error raised.

- **EC-D02**: Feature column that is all-NaN after vectorization.
  Expected behavior: `impute_train_test_non_finite` fills with the column median (or 0 if median is also non-finite); tensor has no NaN values.

- **EC-D03**: Empty `x_rows` list passed to `prepare_tensors`.
  Expected behavior: `train_test_split` raises `ValueError`; `prepare_tensors` does not suppress it.

---

## Dependencies

| Dependency | What It Provides | Failure Mode | Fallback |
|---|---|---|---|
| `torch` | Tensor creation and dtype constants | `ImportError` at module load | None — module unusable |
| `numpy` | Array operations, dtype casting | `ImportError` at module load | None — module unusable |
| `sklearn.feature_extraction.DictVectorizer` | Feature vectorization from dicts | `ImportError` at module load | None |
| `sklearn.preprocessing.LabelEncoder` | Integer encoding of class labels | `ImportError` at module load | None |
| `sklearn.preprocessing.StandardScaler` | Zero-mean, unit-variance scaling | `ImportError` at module load | None |
| `sklearn.model_selection.train_test_split` | Stratified train/test splitting | `ImportError` at module load | None |
| `ai.ml.core.preprocessing.impute_train_test_non_finite` | NaN/Inf imputation with column medians | Propagates exception upward | None — caller must ensure clean data |
| `ai.ml.core.types.TrainingConfig` | `test_size`, `random_seed` config fields | Invalid config raises `AttributeError` | None |
| `ai.ml.ml_util.TaskType` | Task string type annotation | N/A (annotation only) | N/A |

---

## Open Questions

None at this time.

---

## Traceability
- Requirement and acceptance trace matrix: [pytorch-backend-traceability.spec.md](../../../../traceability/pytorch-backend-traceability.spec.md#pt-data-001)
