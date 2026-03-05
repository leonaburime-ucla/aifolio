# State Spec: AG-UI Tool Routing

- id: `ag-ui-tool-routing.state`
- version: `1.0.0`
- status: `draft`
- lastUpdated: `2026-03-04`

## Stores
- `agUiWorkspaceStore`
- `copilotMessageStore`

## Contracts
- Active tab state domain is `charts | agentic-research | pytorch | tensorflow`.
- Tab resolver MUST normalize aliases to canonical tab ids.
- State adapters expose selectors/actions without page-level direct store mutation.
- Chat message persistence state MUST survive route transitions and rehydration.

## Requirements Mapping
- REQ-004, REQ-006, REQ-007
