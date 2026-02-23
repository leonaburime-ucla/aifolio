# Feature Spec: Recharts

Spec ID: `recharts`
Version: `1.1.0`
Status: `draft`
Last updated: `2026-02-23`

## Scope

In scope:
- Chart store and adapter contracts for chart spec add/remove/clear flows.
- Chart orchestration contract exposed through injected chart management port.
- Renderer behavior contracts for Recharts/ECharts-supported chart types.
- Value formatting and axis formatting deterministic behavior.

Out of scope:
- LLM chart generation prompt behavior.
- Consumer page layout concerns.

## Requirement Set

- REQ-001: Chart add operation deduplicates by id and prepends latest chart.
- REQ-002: Chart remove operation removes by exact id and leaves non-matching ids unchanged.
- REQ-003: `useChartOrchestrator` returns output of injected management port unchanged.
- REQ-004: Unsupported chart types render explicit fallback panel instead of throwing.
- REQ-005: Value/axis formatter behavior is deterministic for currency, unit, integers, floats, and year-like values.

## Deterministic Rules

- DR-001: `addChartSpec` transforms `[A,B] + B -> [B,A]`.
- DR-002: `formatXAxisValue` for year-like numeric values in `[1000,3000]` returns rounded integer string.
- DR-003: Heatmap/box/dendrogram/scatter/bar hist variants route to ECharts renderer where configured.
- DR-004: Orchestrator remains consumer-agnostic and imports only feature-local ports/adapters.

## Acceptance Scenarios

- AC-001 (REQ-001): Given chart ids `[A,B]`, adding `B` results in `[B,A]`.
- AC-002 (REQ-002): Given chart ids `[A,B]`, removing `B` results in `[A]`; removing `X` leaves `[A,B]`.
- AC-003 (REQ-003): Given injected port returns `{chartSpecs:[A],removeChartSpec}`, orchestrator returns same contract.
- AC-004 (REQ-004): Given chart type `violin`, renderer returns unsupported-chart fallback block.
- AC-005 (REQ-005): Given `currency='USD'` and value `10`, formatter returns `$10.00` style output via Intl API.

## Open Clarifications

- None.
