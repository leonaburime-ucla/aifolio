# Behavior Spec — ai/ml/frameworks/pytorch/trainer.py

Spec ID:      PT-TRAINER-BEHAVIOR-001
Version:      1.0
Last Edited:  2026-02-25T00:00:00Z
Hash:         sha256:9d8792f5c6fb51e86113fc1dc5947b6547309f76ea7e524a7fca2558c17cea00

## Purpose

Documents the branching execution paths inside `train_model_from_file`. The function has several meaningful decision points: task inference, training-mode-to-criterion mapping, tree-teacher distillation path, batch skip / update_steps guard, and device selection. This file is the authoritative reference for writing branched integration and unit tests.

For requirement contracts, see `trainer.feature.spec.md`.

---

## `train_model_from_file` — Execution Flow

```
train_model_from_file(data_path, cfg, ...)
│
├─ [1] Seed: random, numpy, torch ← always first
│
├─ [2] Load data via load_tabular_file
│       → expand date columns
│       → apply exclude_columns
│
├─ [3] cfg.target_column in exclude_columns?
│   YES → raise ValueError("target_column cannot be excluded")
│   NO  → continue
│
├─ [4] cfg.task == "auto"?
│   YES → infer_task(y_raw) → task
│   NO  → task = cfg.task
│
├─ [5] training_mode in ("imbalance_aware", "calibrated_classifier") AND task != "classification"?
│   YES → raise ValueError
│   NO  → continue
│
├─ [6] prepare_tensors(x_rows, y_raw, task, cfg)
│       → x_train, x_test, y_train, y_test, vectorizer, scaler,
│          col_medians, encoder, target_scaler, input_dim, output_dim, class_names
│
├─ [7] Device selection
│   device arg provided? → use it
│   NO → torch.cuda.is_available()? → "cuda" else "cpu"
│
├─ [8] build_model(input_dim, output_dim, training_mode, hidden_dim, num_hidden_layers, dropout)
│       → move model to device
│
├─ [9] Criterion selection
│   ┌─ task == "classification"
│   │   ├─ training_mode == "imbalance_aware"
│   │   │   → CrossEntropyLoss(weight=compute_class_weights(y_train, output_dim, device))
│   │   ├─ training_mode == "calibrated_classifier"
│   │   │   → CrossEntropyLoss(label_smoothing=0.05)
│   │   └─ all other modes
│   │       → CrossEntropyLoss()
│   └─ task == "regression"
│       → MSELoss()
│
├─ [10] training_mode == "tree_teacher_distillation"?
│   YES →
│   │   ├─ task == "classification"
│   │   │   → fit RandomForestClassifier(n_estimators=120, max_depth=8) on x_train_np, y_train_np
│   │   └─ task == "regression"
│   │       → fit RandomForestRegressor(n_estimators=120, max_depth=10) on x_train_np, y_train_np
│   NO  → tree_teacher = None
│
├─ [11] Optimizer: Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
│
├─ [12] Training loop: for epoch in range(cfg.epochs)
│         for batch in batches(x_train, y_train, cfg.batch_size):
│         │
│         ├─ len(batch) < 2? → SKIP (no weight update)
│         │
│         ├─ tree_teacher present?
│         │   YES →
│         │   │   tree_preds = tree_teacher.predict / predict_proba(batch_x_np)
│         │   │   hard_loss = criterion(model(batch_x), batch_y)
│         │   │   ├─ classification:
│         │   │   │   soft_loss = KL-div(log_softmax(student/T=2), softmax(tree_probs/T=2)) × 4
│         │   │   └─ regression:
│         │   │       soft_loss = MSELoss(student_out, tree_preds_tensor)
│         │   │   loss = 0.5 × hard_loss + 0.5 × soft_loss
│         │   NO →
│         │       loss = criterion(model(batch_x), batch_y)
│         │
│         └─ loss.backward() → optimizer.step() → update_steps += 1
│
├─ [13] update_steps == 0?
│   YES → raise ValueError (no valid batches processed)
│   NO  → continue
│
├─ [14] Evaluate on test set (model.eval(), no_grad)
│   ├─ task == "classification" → accuracy
│   └─ task == "regression"    → rmse (inverse-transform via target_scaler)
│
└─ [15] model.eval()
         return (ModelBundle, Metrics)
```

---

## Decision Branch Summary

| Decision Point | Condition | Outcome |
|---|---|---|
| Task inference | `cfg.task == "auto"` | `infer_task(y_raw)` called |
| Task inference | `cfg.task != "auto"` | Explicit value used; `infer_task` never called |
| Mode restriction | `imbalance_aware` or `calibrated_classifier` + regression | `ValueError` raised |
| Criterion | classification + `imbalance_aware` | `CrossEntropyLoss(weight=...)` |
| Criterion | classification + `calibrated_classifier` | `CrossEntropyLoss(label_smoothing=0.05)` |
| Criterion | classification + all other modes | `CrossEntropyLoss()` |
| Criterion | regression | `MSELoss()` |
| Tree teacher | `tree_teacher_distillation` + classification | RF classifier, KL soft loss, T=2 |
| Tree teacher | `tree_teacher_distillation` + regression | RF regressor, MSE soft loss |
| Tree teacher | all other modes | No tree teacher; loss = hard only |
| Batch skip | `len(batch) < 2` | Skip; no gradient update |
| Convergence guard | `update_steps == 0` | `ValueError` raised |
| Device | `device` arg provided | Use it |
| Device | `device` arg absent + CUDA available | `"cuda"` |
| Device | `device` arg absent + no CUDA | `"cpu"` |
| Metric | classification | `accuracy` |
| Metric | regression | `rmse` (original scale via `target_scaler.inverse_transform`) |

---

## Tree-Teacher Distillation Detail

When `training_mode == "tree_teacher_distillation"`:

```
tree_temperature = 2.0  (hardcoded)
combined_loss_weight = 0.5  (hardcoded)

Classification soft loss:
  student_probs = log_softmax(student_logits / tree_temperature, dim=1)
  teacher_probs = softmax(tree_proba_tensor / tree_temperature, dim=1)
  soft_loss = KLDivLoss(reduction="batchmean")(student_probs, teacher_probs) × tree_temperature²

Regression soft loss:
  soft_loss = MSELoss()(student_out, tree_preds_tensor)

Combined:
  loss = 0.5 × hard_loss + 0.5 × soft_loss
```

Note: These values (`temperature=2.0`, `weight=0.5`) are hardcoded in the tree-teacher path only. They are not controlled by the user or by `TrainingConfig`. This is distinct from the knowledge-distillation in `distill.py`, where `temperature` and `alpha` are user-configurable.
