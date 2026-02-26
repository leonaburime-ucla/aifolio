# Spec Manifest — data.py

| File | Status | Notes |
|------|--------|-------|
| `data.spec.md` | Produced | Core function contract for `prepare_tensors` |
| `behavior.spec.md` | Omitted | `prepare_tensors` is a linear pipeline with no branching or ordering decisions; no behavior spec needed |
| `errors.spec.md` | Omitted | No structured error codes defined; all errors propagate from dependencies |
| `traceability.spec.md` | Omitted | Covered by shared `../../../traceability/pytorch-backend-traceability.spec.md` |
