# ai/ml/frameworks/pytorch/models.py

Spec ID:      PT-MODELS-001
Version:      0.5
Last Edited:  2026-02-25T00:00:00Z
Hash:         sha256:0c407dde13a077127ac35a1a7d0ccb72d75460822862f0e860722dbe91ad74e8

## Goals
- Define the behavioral contract for every neural-network architecture exposed by the models module.
- Specify the `build_model` factory and model-introspection helpers.

## Scope
- **In scope**: `MLP`, `LinearBaseline`, `ResidualBlock`, `TabResNet` classes; `build_model`, `compute_loss`, `compute_class_weights`, `model_hidden_dim`, `model_num_hidden_layers`, `model_dropout`, `model_training_mode` functions.
- **Out of scope**: Internal weight-initialization details; training loop behavior (see `trainer/trainer.feature.spec.md`).

---

## Type Aliases

| Name | Definition |
|------|-----------|
| `TrainingMode` | `Literal["mlp_dense", "linear_glm_baseline", "tabresnet", "imbalance_aware", "calibrated_classifier", "tree_teacher_distillation"]` |

---

## Architecture Classes

### MLP (nn.Module)

A fully-connected feed-forward network with configurable depth, width, and dropout regularization. Used as the default architecture for all training modes except `linear_glm_baseline` and `tabresnet`.

- **REQ-M01**: Constructor accepts `input_dim`, `output_dim`, `hidden_dim`, `num_hidden_layers`, `dropout`.
- **REQ-M02**: Architecture is `[Linear(input_dim → hidden_dim) → BatchNorm1d → ReLU (→ Dropout)?] × max(1, num_hidden_layers) → Linear(hidden_dim → output_dim)`.
  - Number of hidden blocks = `max(1, num_hidden_layers)`.
  - Dropout layer is included in a block only when `dropout > 0`.
- **REQ-M03**: `forward(x)` returns a tensor of shape `(batch, output_dim)`.

### LinearBaseline (nn.Module)

A single linear transformation with no hidden layers, equivalent to a generalized linear model (GLM). Used as a performance baseline to measure the contribution of non-linearity in deeper architectures.

- **REQ-M04**: Constructor accepts `input_dim`, `output_dim` only.
- **REQ-M05**: Architecture is a single `nn.Linear(input_dim, output_dim)`.
- **REQ-M06**: `forward(x)` returns a tensor of shape `(batch, output_dim)`.

### ResidualBlock (nn.Module)

A single residual block — the building unit of `TabResNet`. It adds the input identity directly to the transformed output to facilitate gradient flow in deeper networks.

- **REQ-M07**: Constructor accepts `hidden_dim`, `dropout`.
- **REQ-M08**: Architecture: `Linear(hidden_dim → hidden_dim) → BN → ReLU → Dropout → Linear(hidden_dim → hidden_dim) → BN → (+identity) → ReLU`.
  - Dropout position uses `nn.Identity()` when `dropout <= 0`.
- **REQ-M09**: Input and output shapes are identical: `(batch, hidden_dim)`.

### TabResNet (nn.Module)

A residual network adapted for tabular data. Projects input features to a hidden dimension, then stacks `ResidualBlock` modules, then projects to the output dimension. Used when `training_mode == "tabresnet"`.

- **REQ-M10**: Constructor accepts `input_dim`, `output_dim`, `hidden_dim`, `num_hidden_layers`, `dropout`.
- **REQ-M11**: Architecture: `Linear(input_dim → hidden_dim) → BN → ReLU → [ResidualBlock] × max(1, num_hidden_layers) → Linear(hidden_dim → output_dim)`.
- **REQ-M12**: `forward(x)` returns a tensor of shape `(batch, output_dim)`.

---

## Factory Function

### `build_model(input_dim, output_dim, training_mode, hidden_dim, num_hidden_layers, dropout) → nn.Module`

Single entry point for all model construction. Selects and instantiates the correct architecture class based on `training_mode`, so the rest of the pipeline never imports model classes directly.

- **REQ-M13**: When `training_mode == "linear_glm_baseline"`, return a `LinearBaseline`.
- **REQ-M14**: When `training_mode == "tabresnet"`, return a `TabResNet`.
- **REQ-M15**: For all other modes (`"mlp_dense"`, `"imbalance_aware"`, `"calibrated_classifier"`, `"tree_teacher_distillation"`), return an `MLP`.

---

## Utility Functions

### `compute_loss(model, x, y, criterion) → Tensor`

Runs a single forward pass and applies the given loss criterion. Returns a scalar loss tensor for use in the training and distillation loops.

- **REQ-M16**: Returns `criterion(model(x), y)`.

### `compute_class_weights(y_train, output_dim, device) → Tensor`

Computes inverse-frequency class weights from training labels so that rare classes receive proportionally higher loss weight during `imbalance_aware` training.

- **REQ-M17**: Returns a weight tensor of shape `(output_dim,)` on the specified `device`.
- **REQ-M18**: Zero-count classes are clamped to `min=1.0` before division to prevent divide-by-zero.
- **REQ-M19**: Weight formula: `total_samples / (num_classes × class_count_per_class)`.

### `model_hidden_dim(model) → int`

Inspects the live model instance and returns its hidden-layer width. Used by serialization and distillation to record or compress architecture hyperparameters.

- **REQ-M20**: Returns the model's hidden dimension; falls back to `128` if no `nn.Linear` is found.
- **REQ-M21**: For `LinearBaseline`, returns `out_features` of the single layer; for `TabResNet`, returns `out_features` of the input projection layer.

### `model_num_hidden_layers(model) → int`

Inspects the live model instance and returns the number of hidden layers. Used by distillation to derive compressed student architecture defaults.

- **REQ-M22**: `LinearBaseline → 1`; `TabResNet → max(1, len(model.blocks))`; `MLP → max(1, count_of_Linear_layers_in_net - 1)`.

### `model_dropout(model) → float`

Inspects the live model instance and returns its dropout rate by scanning for the first `nn.Dropout` layer. Returns `0.0` if none is found.

- **REQ-M23**: Returns `0.0` for `LinearBaseline` (no dropout layer exists).
- **REQ-M24**: For all other model types, scans `model.modules()` and returns `p` from the first `nn.Dropout` found; returns `0.0` if none is found.

### `model_training_mode(model) → TrainingMode`

Determines the `TrainingMode` string for a model instance based on its class. Used by serialization to record which mode produced the saved model.

- **REQ-M25**: `LinearBaseline → "linear_glm_baseline"`, `TabResNet → "tabresnet"`, all other types → `"mlp_dense"`.

---

## Acceptance Criteria

- **AC-M01 (REQ-M01) [P1]**: Given `input_dim=10, output_dim=3, hidden_dim=64, num_hidden_layers=2, dropout=0.1`, when `MLP(...)` is constructed, then it accepts a `(batch, 10)` input tensor without error.
- **AC-M02 (REQ-M02) [P1]**: Given `num_hidden_layers=0`, when an `MLP` is constructed, then it contains exactly 1 hidden block (max guard applied).
- **AC-M03 (REQ-M03) [P1]**: Given any valid `MLP`, when `forward(x)` is called with a `(B, input_dim)` tensor, then output shape is `(B, output_dim)`.
- **AC-M04 (REQ-M04) [P2]**: Given `input_dim=5, output_dim=2`, when `LinearBaseline(5, 2)` is constructed, then it contains exactly one `nn.Linear` layer.
- **AC-M05 (REQ-M05) [P2]**: Given `LinearBaseline(5, 2)`, when inspected, then its sole layer has `in_features=5` and `out_features=2`.
- **AC-M06 (REQ-M06) [P1]**: Given any valid `LinearBaseline`, when `forward(x)` is called with `(B, input_dim)`, then output shape is `(B, output_dim)`.
- **AC-M07 (REQ-M07) [P2]**: Given `hidden_dim=32, dropout=0.2`, when `ResidualBlock(32, 0.2)` is constructed, then it accepts a `(batch, 32)` tensor.
- **AC-M08 (REQ-M08) [P1]**: Given `dropout=0`, when a `ResidualBlock` is constructed, then the dropout position contains `nn.Identity` rather than `nn.Dropout`.
- **AC-M09 (REQ-M09) [P1]**: Given a `ResidualBlock`, when `forward(x)` is called with `(B, hidden_dim)`, then output shape is `(B, hidden_dim)`.
- **AC-M10 (REQ-M10) [P2]**: Given valid hyperparameters, when `TabResNet(...)` is constructed, then it accepts `(B, input_dim)` without error.
- **AC-M11 (REQ-M11) [P1]**: Given `num_hidden_layers=0`, when a `TabResNet` is constructed, then it contains at least 1 `ResidualBlock`.
- **AC-M12 (REQ-M12) [P1]**: Given any valid `TabResNet`, when `forward(x)` is called with `(B, input_dim)`, then output shape is `(B, output_dim)`.
- **AC-M13 (REQ-M13) [P1]**: Given `training_mode="linear_glm_baseline"`, when `build_model` is called, then the returned object is an instance of `LinearBaseline`.
- **AC-M14 (REQ-M14) [P1]**: Given `training_mode="tabresnet"`, when `build_model` is called, then the returned object is an instance of `TabResNet`.
- **AC-M15 (REQ-M15) [P1]**: Given `training_mode="mlp_dense"` or any other non-baseline mode, when `build_model` is called, then the returned object is an instance of `MLP`.
- **AC-M16 (REQ-M16) [P1]**: Given a model, input tensor x, labels y, and a criterion, when `compute_loss` is called, then it returns `criterion(model(x), y)` as a scalar tensor.
- **AC-M17 (REQ-M17) [P1]**: Given y_train with K classes, when `compute_class_weights` is called, then the returned tensor has shape `(K,)` and is on the specified device.
- **AC-M18 (REQ-M18) [P1]**: Given a class with zero occurrences in y_train, when `compute_class_weights` is called, then that class count is treated as 1 (no divide-by-zero error).
- **AC-M19 (REQ-M19) [P2]**: Given a balanced 2-class dataset with N samples per class, when `compute_class_weights` is called, then both weights equal `1.0`.
- **AC-M20 (REQ-M20) [P2]**: Given a model with no `nn.Linear` layer, when `model_hidden_dim` is called, then it returns `128`.
- **AC-M21 (REQ-M21) [P2]**: Given a `LinearBaseline` with `out_features=4`, when `model_hidden_dim` is called, then it returns `4`.
- **AC-M22 (REQ-M22) [P1]**: Given an `MLP` with 3 `Linear` layers in `net`, when `model_num_hidden_layers` is called, then it returns `max(1, 3-1) = 2`.
- **AC-M23 (REQ-M23) [P2]**: Given a `LinearBaseline`, when `model_dropout` is called, then it returns `0.0`.
- **AC-M24 (REQ-M24) [P2]**: Given an `MLP` constructed with `dropout=0.3`, when `model_dropout` is called, then it returns `0.3`.
- **AC-M25 (REQ-M25) [P2]**: Given a `TabResNet` instance, when `model_training_mode` is called, then it returns `"tabresnet"`.

---

## Invariants
- **INV-M01**: All `forward()` outputs have 2-D shape `(batch_size, output_dim)`.
- **INV-M02**: `build_model` never raises for any valid value in `TrainingMode`.
- **INV-M03**: Introspection helpers (`model_hidden_dim`, `model_num_hidden_layers`, `model_dropout`, `model_training_mode`) always return deterministic values for the same model instance.

---

## Edge Cases

- **EC-M01**: `num_hidden_layers=0` passed to `MLP` or `TabResNet`.
  Expected behavior: `max(1, 0)` guard produces 1 hidden block; model constructed without error.

- **EC-M02**: `dropout=0` passed to `MLP` or `ResidualBlock`.
  Expected behavior: No `nn.Dropout` layer in MLP blocks; `ResidualBlock` uses `nn.Identity` at the dropout position.

- **EC-M03**: Unknown/custom model type passed to an introspection helper.
  Expected behavior: Helper falls back to generic `model.modules()` iteration or default return values (`128`, `1`, `0.0`, `"mlp_dense"`) without raising.

---

## Dependencies

| Dependency | What It Provides | Failure Mode | Fallback |
|---|---|---|---|
| `torch` | Tensor operations, device placement | `ImportError` at module load | None — module unusable |
| `torch.nn` | `Module`, `Linear`, `BatchNorm1d`, `ReLU`, `Dropout`, `Identity` | `ImportError` at module load | None |

---

## Open Questions

None at this time.

---

## Traceability
- Requirement and acceptance trace matrix: [pytorch-backend-traceability.spec.md](../../../../traceability/pytorch-backend-traceability.spec.md#pt-models-001)
