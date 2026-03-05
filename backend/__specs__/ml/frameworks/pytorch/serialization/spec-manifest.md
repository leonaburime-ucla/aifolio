# Spec Manifest — serialization.py

| File | Status | Notes |
|------|--------|-------|
| `serialization.spec.md` | Produced | Save/load contracts, on-disk format, round-trip invariant |
| `behavior.spec.md` | Omitted | Both functions are linear (no branching); on-disk format is fully documented in `serialization.spec.md` |
| `errors.spec.md` | Omitted | No structured error codes; errors propagate from `torch.load`/`torch.save` as-is |
| `traceability.spec.md` | Omitted | Covered by shared `../../../traceability/pytorch-backend-traceability.spec.md` |
