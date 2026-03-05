# Traceability Spec: AG-UI Tool Routing

- id: `ag-ui-tool-routing.traceability`
- version: `1.0.0`
- status: `draft`
- lastUpdated: `2026-03-04`

| REQ | Module(s) | Test(s) |
|---|---|---|
| REQ-001 | `typescript/logic/frontendTools.logic.ts`, `typescript/config/frontendTools.config.ts` | `__tests__/typescript/logic/frontendTools.logic.unit.test.ts` |
| REQ-002 | `typescript/config/frontendTools.config.ts` | `__tests__/typescript/logic/frontendTools.config.unit.test.ts` |
| REQ-003 | `typescript/logic/frontendTools.logic.ts` | `__tests__/typescript/logic/frontendTools.logic.unit.test.ts` |
| REQ-004 | `typescript/logic/frontendTools.logic.ts`, `typescript/react/views/components/AgUiTabSwitchTool.tsx` | `__tests__/typescript/logic/frontendTools.logic.unit.test.ts`, `__tests__/typescript/react/views/AgUiTabSwitchTool.unit.test.tsx` |
| REQ-005 | `typescript/react/views/components/CopilotFrontendTools.tsx`, `typescript/logic/frontendTools.logic.ts` | `__tests__/typescript/react/views/CopilotFrontendTools.unit.test.tsx` |
| REQ-006 | `src/app/ag-ui/page.tsx`, `typescript/react/views/providers/CopilotEffectsProvider.tsx` | `__tests__/typescript/integration/req-006.ag-ui-tool-wiring.integration.test.tsx` |
| REQ-007 | `typescript/react/orchestrators/copilotMessagePersistence.orchestrator.ts`, `typescript/react/state/zustand/copilotMessageStore.ts` | `__tests__/typescript/integration/req-007.ag-ui-chat-history-persistence.integration.test.tsx` |
