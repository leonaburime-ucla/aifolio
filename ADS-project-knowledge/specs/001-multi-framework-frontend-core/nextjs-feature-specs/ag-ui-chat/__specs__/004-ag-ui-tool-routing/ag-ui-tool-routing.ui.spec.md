# UI Spec: AG-UI Tool Routing

- id: `ag-ui-tool-routing.ui`
- version: `1.0.0`
- status: `draft`
- lastUpdated: `2026-03-04`

## Components
- `CopilotFrontendTools`
- `AgUiTabSwitchTool`
- `CopilotEffectsProvider`
- `AgUiPage`

## Contracts
- `/ag-ui` mounts both global tool registration and AG-UI-local tab switch tool.
- Tool registration components remain renderless and side-effect oriented.
- `/ag-ui` keeps workspace behavior deterministic under tool calls.
- `/ag-ui` chat UI restores persisted conversation state when remounted after navigation.

## Requirements Mapping
- REQ-005, REQ-006, REQ-007
