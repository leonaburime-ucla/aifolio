# Behavior Spec — ai/ml/frameworks/pytorch/distill.py

Spec ID:      PT-DISTILL-BEHAVIOR-001
Version:      1.0
Last Edited:  2026-02-25T00:00:00Z
Hash:         sha256:8f1ab9be20c9e9e80ce9fd29d9018e2ec718c299b3bfa7d2e60e910cfd72c6cf

## Purpose

Documents the branching execution paths inside `distill_model_from_file`. Key decision points include: teacher source resolution, task alignment, student architecture default derivation, classification vs regression loss path, and alpha boundary behavior. This file is the authoritative reference for writing branched tests for the distillation pipeline.

For requirement contracts, see `distill.feature.spec.md`.

---

## `distill_model_from_file` — Execution Flow

```
distill_model_from_file(data_path, cfg, teacher_path?, teacher_bundle?, ..., temperature, alpha)
│
├─ [1] Seed: random, numpy, torch ← before everything else
│
├─ [2] temperature <= 0?
│   YES → raise ValueError("temperature must be > 0")
│   NO  → continue
│
├─ [3] alpha outside [0, 1]?
│   YES → raise ValueError("alpha must be between 0 and 1")
│   NO  → continue
│
├─ [4] Teacher resolution
│   ├─ teacher_bundle provided?  → use directly (no disk load)
│   ├─ teacher_path provided?    → load via serialization.load_bundle(teacher_path)
│   └─ neither?                  → raise ValueError("teacher_path or teacher_bundle is required.")
│
├─ [5] Task alignment
│   ├─ cfg.task == "auto"
│   │   → task = teacher.task  (infer_task is NEVER called)
│   └─ cfg.task explicit
│       → task != teacher.task? → raise ValueError("Requested task does not match teacher task.")
│       → task == teacher.task? → task = cfg.task
│
├─ [6] cfg.target_column in exclude_columns?
│   YES → raise ValueError("target_column cannot be excluded")
│   NO  → continue
│
├─ [7] Load data via load_tabular_file → expand dates → exclude columns
│
├─ [8] Split: classification with >1 unique labels → stratified; else → not stratified
│
├─ [9] Feature transform using TEACHER artifacts (no re-fitting)
│       teacher.vectorizer.transform(x_rows)
│       impute_train_test_non_finite(...)
│       teacher.scaler.transform(x_np)
│
├─ [10] Label handling
│   ├─ task == "classification"
│   │   teacher.label_encoder is None?
│   │   YES → raise ValueError("Teacher model is missing a label encoder.")
│   │   NO  → teacher.label_encoder.transform(y_labels)
│   └─ task == "regression"
│       teacher.target_scaler is None?
│       YES → targets unscaled (raw float values)
│       NO  → teacher.target_scaler.transform(y_np)
│
├─ [11] Student architecture defaults (only for unspecified params)
│   ├─ student_hidden_dim    not given → max(16, teacher_hidden_dim // 2)
│   ├─ student_num_layers    not given → max(1, teacher_num_layers - 1)
│   └─ student_dropout       not given → min(0.5, teacher_dropout + 0.05)
│
├─ [12] build_model(input_dim, output_dim, cfg.training_mode, student_hidden_dim, ...)
│         → move student to device
│
├─ [13] teacher.eval()  ← set once, never changed
│        optimizer = Adam(student.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
│
├─ [14] Distillation loop: for epoch in range(cfg.epochs)
│         for batch in batches(x_train, y_train, cfg.batch_size):
│         │
│         ├─ len(batch) < 2? → SKIP
│         │
│         ├─ teacher outputs: with torch.no_grad(): teacher_out = teacher(batch_x)
│         │
│         ├─ task == "classification"?
│         │   YES →
│         │   │   hard_loss = CrossEntropyLoss(student_logits, y_labels)
│         │   │   soft_loss = KLDiv(
│         │   │                 log_softmax(student_logits / T, dim=1),
│         │   │                 softmax(teacher_logits / T, dim=1)
│         │   │               ) × T²
│         │   NO (regression) →
│         │       hard_loss = MSELoss(student_out, y_targets)
│         │       soft_loss = MSELoss(student_out, teacher_out)
│         │
│         ├─ combined = alpha × hard_loss + (1 - alpha) × soft_loss
│         │   alpha == 0.0 → pure soft loss (teacher mimicry)
│         │   alpha == 1.0 → pure hard loss (no distillation)
│         │   0 < alpha < 1 → blended
│         │
│         └─ loss.backward() → optimizer.step() → update_steps += 1
│
├─ [15] update_steps == 0?
│   YES → raise ValueError (no valid batches)
│   NO  → continue
│
├─ [16] Evaluate student on test set (eval(), no_grad)
│   ├─ classification → accuracy
│   └─ regression     → rmse (inverse-transform via teacher.target_scaler if present)
│
├─ [17] Assemble student bundle
│   vectorizer     = teacher.vectorizer      ← always from teacher
│   scaler         = teacher.scaler          ← always from teacher
│   label_encoder  = teacher.label_encoder   ← always from teacher
│   target_scaler  = teacher.target_scaler   ← always from teacher
│   class_names    = teacher.class_names     ← always from teacher
│   feature_medians: teacher.feature_medians if not None, else newly computed col_medians
│
└─ [18] student.eval()
         return (student_bundle, metrics)
```

---

## Decision Branch Summary

| Decision Point | Condition | Outcome |
|---|---|---|
| Teacher source | `teacher_bundle` provided | Used directly |
| Teacher source | `teacher_path` provided, no bundle | Loaded from disk |
| Teacher source | Neither provided | `ValueError` |
| Task | `cfg.task == "auto"` | Taken from `teacher.task`; `infer_task` never called |
| Task | Explicit, matches teacher | Used |
| Task | Explicit, conflicts with teacher | `ValueError` |
| Label encoder | Classification + encoder is `None` | `ValueError` |
| Target scaler | Regression + scaler is `None` | Targets unscaled |
| Stratification | Classification + >1 unique labels | Stratified split |
| Stratification | Classification + 1 unique label | Unstratified split |
| Loss type | Classification | Cross-entropy hard + KL soft |
| Loss type | Regression | MSE hard + MSE soft |
| Alpha boundary | `alpha == 0.0` | 100% soft loss |
| Alpha boundary | `alpha == 1.0` | 100% hard loss |
| Student defaults | `student_hidden_dim` not given | `max(16, teacher_hidden_dim // 2)` |
| Student defaults | `student_num_hidden_layers` not given | `max(1, teacher_layers - 1)` |
| Student defaults | `student_dropout` not given | `min(0.5, teacher_dropout + 0.05)` |
| Feature medians | Teacher has non-None `feature_medians` | Inherited from teacher |
| Feature medians | Teacher `feature_medians` is `None` | Newly computed `col_medians` used |
| Convergence | `update_steps == 0` | `ValueError` |
