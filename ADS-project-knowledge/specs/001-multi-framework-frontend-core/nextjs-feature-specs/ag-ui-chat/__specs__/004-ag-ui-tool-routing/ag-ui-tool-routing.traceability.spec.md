# Traceability Spec: AG-UI Tool Routing

- id: `ag-ui-tool-routing.traceability`
- version: `1.0.0`
- status: `draft`
- lastUpdated: `2026-03-04`

| REQ | Module(s) | Test(s) |
|---|---|---|
| REQ-001 | `logic/frontendTools.logic.ts`, `config/frontendTools.config.ts` | `__tests__/logic/frontendTools.logic.unit.test.ts` |
| REQ-002 | `config/frontendTools.config.ts` | `__tests__/logic/frontendTools.config.unit.test.ts` |
| REQ-003 | `logic/frontendTools.logic.ts` | `__tests__/logic/frontendTools.logic.unit.test.ts` |
| REQ-004 | `logic/frontendTools.logic.ts`, `react/views/components/AgUiTabSwitchTool.tsx` | `__tests__/logic/frontendTools.logic.unit.test.ts`, `__tests__/react/views/AgUiTabSwitchTool.unit.test.tsx` |
| REQ-005 | `react/views/components/CopilotFrontendTools.tsx`, `logic/frontendTools.logic.ts` | `__tests__/react/views/CopilotFrontendTools.unit.test.tsx` |
| REQ-006 | `src/app/ag-ui/page.tsx`, `react/views/providers/CopilotEffectsProvider.tsx` | `__tests__/integration/req-006.ag-ui-tool-wiring.integration.test.tsx` |
| REQ-007 | `react/orchestrators/copilotMessagePersistence.orchestrator.ts`, `react/state/zustand/copilotMessageStore.ts` | `__tests__/integration/req-007.ag-ui-chat-history-persistence.integration.test.tsx` |
