# Orchestrator Spec: AG-UI Tool Routing

- id: `ag-ui-tool-routing.orchestrator`
- version: `1.0.0`
- status: `draft`
- lastUpdated: `2026-03-04`

## Orchestrators
- `frontendTools.logic`
- `copilotChartBridge.orchestrator`
- `copilotMessagePersistence.orchestrator`

## Contracts
- Validation and normalization logic for tool calls lives in pure logic modules.
- React tool registration delegates to logic handlers and only performs side effects (router push, store action dispatch).
- Tool handlers return structured payloads; no thrown errors for invalid input branches.

## Requirements Mapping
- REQ-001, REQ-002, REQ-003, REQ-004, REQ-005
