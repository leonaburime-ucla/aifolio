# Behavior Spec: AG-UI Tool Routing

- id: `ag-ui-tool-routing.behavior`
- version: `1.0.0`
- status: `draft`
- lastUpdated: `2026-03-04`

## Deterministic Behavior Rules
- DR-001: Route alias input normalization is `trim().toLowerCase()`.
- DR-002: Slash-prefixed unknown route inputs are passed through as direct candidates.
- DR-003: Allowed route check compares canonical route membership only.
- DR-004: AG-UI tab values normalize case-insensitively to canonical tab ids.

## Scenarios
- Given `route="agentic research"`, result resolves to `/agentic-research`.
- Given `route="/ml/pytorch"`, result resolves directly to `/ml/pytorch`.
- Given invalid route, result returns `INVALID_ROUTE` and does not navigate.
- Given invalid tab, result returns `INVALID_TAB` and does not mutate active tab.
