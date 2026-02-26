# AI/ML Modularization TODO (PyTorch + TensorFlow + sklearn)

## Goals
- Break `ai/ml/pytorch.py` and `ai/ml/tensorflow.py` into modular, testable units.
- Add independent test tracks so fast checks run often and heavy matrix tests run last.
- Build a unified dataset-vs-algorithm harness across sklearn, PyTorch, and TensorFlow.

## Current Weaknesses (Observed)
- Runtime logic now lives in framework modules, but coverage is still missing.
- High duplication between PyTorch and TensorFlow pipelines increases drift risk.
- ML behavior is mostly tested indirectly via API tests; no dedicated `ai/ml` test suite yet.
- No explicit test markers for expensive dataset/algorithm matrix runs.

## Test Strategy (Independent Tracks)
- `unit` (fast, default): pure functions + validators + preprocessing + model factory.
- `contract` (fast): request payload validation and response shape tests for handlers.
- `smoke` (medium): one short epoch per framework/algorithm on tiny fixture rows.
- `slow_matrix` (slow, opt-in): real datasets in `ai/ml/data` across algorithm matrix.
- `perf` (optional): timing/regression checks for training and distillation budgets.

## Pytest Markers
- `@pytest.mark.unit`
- `@pytest.mark.contract`
- `@pytest.mark.smoke`
- `@pytest.mark.slow_matrix`
- `@pytest.mark.perf`

## Commands
- Fast/default: `pytest -m "unit or contract or smoke"`
- Full matrix (last): `pytest -m slow_matrix`
- Everything: `pytest`

## Phase 1: Baseline Safety Net
- [x] Add `pytest.ini` marker registration.
- [ ] Create `ai/ml/tests/` structure:
  - [ ] `tests/unit/`
  - [ ] `tests/contract/`
  - [ ] `tests/smoke/`
  - [ ] `tests/slow_matrix/`
- [ ] Add first contract tests for:
  - [ ] PyTorch `handle_train_request` validation paths.
  - [ ] TensorFlow `handle_train_request` validation paths.
  - [ ] Distillation unsupported-mode errors.

## Phase 2: Extract Shared Core Modules
- [x] Create core contracts/preprocessing/artifact helper modules.
- [x] Replace duplicated request parsing, bounds validation, and imputation logic with shared core calls.
- [x] Expand shared core typing/contracts to replace framework-local runtime dataclasses.

## Phase 3: Framework Separation
- [ ] Create `ai/ml/frameworks/pytorch/`:
  - [x] `models.py`
  - [x] `trainer.py`
  - [x] `distill.py`
  - [x] `serialization.py`
  - [x] `data.py`
  - [x] `handlers.py`
- [ ] Create `ai/ml/frameworks/tensorflow/`:
  - [x] `models.py`
  - [x] `trainer.py`
  - [x] `distill.py`
  - [x] `serialization.py`
  - [x] `data.py`
  - [x] `handlers.py`
- [x] Keep existing public entrypoints (`pytorch.py`/`tensorflow.py`) as thin adapters during migration.

## Phase 4: sklearn + Agentic Research Matrix Harness
- [ ] Define shared algorithm catalog (framework, mode, task support).
- [ ] Add matrix runner module:
  - [ ] iterates datasets from `ai/ml/data`
  - [ ] executes per-framework algorithm suites
  - [ ] captures status + metrics + failures in machine-readable report
- [ ] Add matrix tests under `tests/slow_matrix/`:
  - [ ] classification datasets (e.g., churn, fraud)
  - [ ] regression datasets (e.g., house prices, sales forecasting)

## Phase 5: CI and Execution Policy
- [ ] PR/commit CI: run `unit + contract + smoke` only.
- [ ] Nightly/manual CI: run `slow_matrix`.
- [ ] Persist matrix reports under `ai/ml/reports/` for trend tracking.

## Initial Suspect Areas for PyTorch Failure Investigation
- [ ] `handle_train_request` validation + coercion path for baseline mode.
- [ ] Training loop edge case: batch-size/test-size causing zero valid updates.
- [ ] Environment/runtime import path mismatch vs API process interpreter.
- [ ] Error propagation: ensure failed runs surface explicit reason in frontend and API.
