# Spec Manifest — distill.py

| File | Status | Notes |
|------|--------|-------|
| `distill.feature.spec.md` | Produced | Full distillation pipeline contract including teacher resolution, student defaults, and loss formulas |
| `distill.behavior.spec.md` | Produced | Branching execution flow: teacher source resolution, task alignment, classification vs regression loss paths, alpha boundary behavior, student default derivation |
| `errors.spec.md` | Omitted | All error conditions are `ValueError` raises documented in requirements; no structured HTTP error codes at this layer |
| `traceability.spec.md` | Omitted | Covered by shared `../../../../traceability/pytorch-backend-traceability.spec.md` |
