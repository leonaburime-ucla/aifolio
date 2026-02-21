# ML Frontend Roadmap (PyTorch + TensorFlow)

This note captures planned frontend behavior for:
- `/ml/pytorch`
- `/ml/tensorflow`
- `/ml/knowledge-distillation`

## Goals

1. Train both platforms on selected CSV/XLS datasets and report comparable results.
2. Run parameter sweeps (learning rate, batch size, hidden units/layers, epochs, etc.), identify best configs, and compare outcomes across platforms.
3. Distill larger trained models into smaller models and compare size vs performance.
4. Add a dataset table in an accordion (default open) with a combobox to choose which table/dataset is shown.

## Vertical Slice Structure (Frontend)

Create a dedicated `features/ml` slice (Orc-BASH style):

- `features/ml/types/*`
- `features/ml/api/*`
- `features/ml/state/zustand/*`
- `features/ml/hooks/*`
- `features/ml/orchestrators/*`
- `features/ml/views/components/*`

Keep route files under `app/ml/*/page.tsx` thin; they should compose the slice only.

## Long-Running Job Strategy

Parameter sweeps and training can exceed normal request windows.

Recommended pattern:
- Start job endpoint: `POST /ml/jobs/start`
- Poll status endpoint: `GET /ml/jobs/{id}`
- Optional stream endpoint for logs/progress: SSE/WebSocket
- Persist each trial result server-side and render incrementally in UI

Do not keep a single long-lived request for full sweep execution in the browser.

## Result Reporting (UI)

For each platform run (PyTorch / TensorFlow), show:
- dataset used
- task type (classification/regression)
- train/test split
- key params
- final metrics (accuracy/F1 or RMSE/MAE)
- training time
- model artifact size

For sweep comparison:
- sortable table of trials
- best trial highlight
- side-by-side “best PyTorch vs best TensorFlow” card

For distillation:
- teacher metrics + size
- student metrics + size
- delta table:
  - size reduction (%)
  - metric drop/gain
  - inference speed delta (if available)

## Required UI Components

1. `DatasetTableAccordion`
- default open
- contains datatable for active dataset preview/stats

2. `DatasetTableCombobox`
- switches dataset/table source
- drives datatable content and downstream train payload defaults

3. `RunSummaryPanel`
- latest run metadata + status

4. `TrialsDataTable`
- all sweep trials with sort/filter

5. `DistillationComparisonPanel`
- teacher vs student summary

## Suggested Execution Order

1. Build dataset combobox + open-by-default accordion datatable.
2. Add single-run training UI for each platform.
3. Add sweep job UI + async job tracking.
4. Add distillation UI + comparison metrics.

## Notes

- Use shared dataset IDs from backend (`ai/ml/data`) so all pages reference the same source of truth.
- Keep chart rendering in existing chart components where possible, but keep ML-specific orchestration inside `features/ml`.
