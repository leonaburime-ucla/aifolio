# AI ML Architecture Notes

## Layering Model

### API Transport Layer
- `ai/python/server.py`
- Responsibility: HTTP routing, status mapping, runtime availability checks.
- Should not own ML-specific validation logic.

### Framework Runtime Layer (Flat Files)
- `ai/ml/frameworks/pytorch/{handlers,trainer,models,data,distill,serialization}.py`
- `ai/ml/frameworks/tensorflow/{handlers,trainer,models,data,distill,serialization}.py`
- Responsibility: stable import surface for server and concrete runtime logic split into flat, testable modules.

### Core Shared Layer
- `ai/ml/core/contracts.py`
- `ai/ml/core/preprocessing.py`
- `ai/ml/core/artifacts.py`
- `ai/ml/core/types.py`
- `ai/ml/core/request_helpers.py`
- Responsibility: cross-framework helpers with deterministic behavior.

### Legacy Implementation Layer (migration target)
- `ai/ml/pytorch.py`
- `ai/ml/tensorflow.py`
- Current role: compatibility shims and public entrypoints while internals are delegated into framework modules.

## Why This Structure

1. Reduces duplication across frameworks.
2. Enables targeted unit/contract testing per layer.
3. Allows incremental extraction without breaking API routes.

## Migration Plan

1. Keep server pointed at `frameworks/*`.
2. Legacy modules remain as compatibility wrappers only.
3. Keep framework files flat; avoid nested framework subdirectories.
4. Add tests per layer after each extraction slice.
