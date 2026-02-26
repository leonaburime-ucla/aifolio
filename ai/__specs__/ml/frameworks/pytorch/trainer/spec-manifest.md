# Spec Manifest — trainer.py

| File | Status | Notes |
|------|--------|-------|
| `trainer.feature.spec.md` | Produced | Full training pipeline, prediction, and re-export contracts |
| `trainer.behavior.spec.md` | Produced | Branching execution flow: task inference, mode→criterion mapping, tree-teacher path, batch skip/convergence guard, device selection |
| `errors.spec.md` | Omitted | No structured error codes; only `ValueError` raises documented in requirements |
| `traceability.spec.md` | Omitted | Covered by shared `../../../../traceability/pytorch-backend-traceability.spec.md` |
