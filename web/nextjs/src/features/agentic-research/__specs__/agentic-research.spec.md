# Feature Spec: Agentic Research

Spec ID: `agentic-research`
Version: `1.1.0`
Status: `draft`
Last updated: `2026-02-23`

## Scope

In scope:
- Dataset manifest loading and default dataset selection.
- Dataset row loading and table model shaping.
- Tool list loading and group formatting contract.
- Chart selection precedence from PCA chart and chart-store chart specs.
- Agentic frontend tool call behavior for chart and dataset actions.

Out of scope:
- Model training pages under `/ml/*`.
- Rendering internals of shared chart/table components.
- Backend implementation details for `/sample-data`, `/sklearn-tools`, `/llm/ds`.

## Requirement Set

- REQ-001: On successful manifest load, selected dataset defaults to `state.selectedDatasetId ?? datasets[0]?.id ?? null`.
- REQ-002: Dataset load resets stale table/chart numeric state before applying new dataset payload.
- REQ-003: Active chart resolves deterministically as `pcaChartSpec ?? chartSpecs[0] ?? null`.
- REQ-004: Frontend tools return stable structured error codes for invalid remove/reorder/set-dataset operations.
- REQ-005: Chart add operation deduplicates by chart id and prepends newest chart.

## Deterministic Rules

- DR-001: Dataset options are mapped in manifest order with `id`, `label`, and optional `description` only.
- DR-002: Numeric matrix extraction applies fixed ratio threshold `>= 0.9` and max rows `1500`.
- DR-003: `reorderChartSpecs(orderedIds)` appends unspecified existing chart ids in their current order.
- DR-004: Dataset switch via tool clears chart specs before applying `selectedDatasetId` update.

## Acceptance Scenarios

- AC-001 (REQ-001): Given manifest `[A,B]` and no selected dataset, when manifest load succeeds, then selected dataset becomes `A`.
- AC-002 (REQ-002): Given prior table/chart state, when dataset load starts, then `tableRows/tableColumns/numericMatrix/featureNames/pcaChartSpec` are cleared before new payload apply.
- AC-003 (REQ-003): Given `pcaChartSpec=null` and `chartSpecs=[X,Y]`, active chart is `X`; given `pcaChartSpec=P`, active chart is `P`.
- AC-004 (REQ-004): Given unknown `chart_id`, remove tool returns `{status:"error", code:"CHART_NOT_FOUND"}` and does not throw.
- AC-005 (REQ-005): Given existing chart ids `[A,B]`, adding `B` results in `[B,A]`.

## Open Clarifications

- None.
