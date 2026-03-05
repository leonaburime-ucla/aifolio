# API Spec: AG-UI Tool Routing

- id: `ag-ui-tool-routing.api`
- version: `1.0.0`
- status: `draft`
- lastUpdated: `2026-03-04`

## Contracts

### Frontend Tool Call: `navigate_to_page`
- Input: `{ route: string }`
- Success: `{ status: "ok", resolvedRoute: string }`
- Error: `{ status: "error", code: "INVALID_ROUTE", allowedRoutes: string[] }`

### Frontend Tool Call: `switch_ag_ui_tab`
- Input: `{ tab: string }`
- Success: `{ status: "ok", tab: "charts" | "agentic-research" | "pytorch" | "tensorflow" }`
- Error: `{ status: "error", code: "INVALID_TAB", allowedTabs: string[] }`

### Route Canonicalization
Canonical routes required for this feature:
- `/`
- `/agentic-research`
- `/ml/pytorch`
- `/ml/tensorflow`

## Requirements Mapping
- REQ-001, REQ-002, REQ-003, REQ-004, REQ-006
