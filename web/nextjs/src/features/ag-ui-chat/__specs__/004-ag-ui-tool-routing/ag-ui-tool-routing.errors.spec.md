# Errors Spec: AG-UI Tool Routing

- id: `ag-ui-tool-routing.errors`
- version: `1.0.0`
- status: `draft`
- lastUpdated: `2026-03-04`

## Error Codes

| code | level | severity | retryable | userMessage |
|---|---|---|---|---|
| `INVALID_ROUTE` | `orchestrator` | `warning` | `false` | Requested route is not supported. |
| `INVALID_TAB` | `orchestrator` | `warning` | `false` | Requested AG-UI tab is not supported. |

## Error Payload Contracts
- `INVALID_ROUTE` payload includes `allowedRoutes: string[]`.
- `INVALID_TAB` payload includes `allowedTabs: string[]`.

## Requirements Mapping
- REQ-003, REQ-004
