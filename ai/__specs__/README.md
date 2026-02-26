# AI Specs Root

This folder is the single source of truth for AI-runtime specs.

## Layout Rule
- Mirror code paths from `ai/` under `ai/__specs__/`.
- Keep specs centralized here; do not duplicate `__specs__` in subfolders.

## Spec Metadata Contract
Each spec file should maintain:
- `Spec ID`
- `Version`
- `Last Edited`
- `Hash` (canonical content hash)

## Testing Flow
1. Finalize spec files in `ai/__specs__/...`.
2. Build/refresh `ai/__specs__/traceability/...` mapping from requirements to tests.
3. Implement tests from the traceability map.
