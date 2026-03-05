# AI Chat Traceability Spec

Version: `1.8.0`
Last updated: `2026-03-02`

| Requirement | Primary module(s) | Required test file(s) |
| --- | --- | --- |
| REQ-001 submit sets sending and appends user message before API send | `typescript/react/hooks/useChat.hooks.ts` | `__tests__/typescript/integration/req-001.empty-input-short-circuit.integration.test.ts`, `__tests__/typescript/integration/req-001.submit-order.integration.test.ts` |
| REQ-002 sending reset and attachment cleanup on all outcomes | `typescript/react/hooks/useChat.hooks.ts` | `__tests__/typescript/integration/req-002.sending-reset.integration.test.ts`, `__tests__/typescript/integration/err-002.fetch-models-fallback.integration.test.ts` |
| REQ-003 chartSpec fan-out deterministic order | `typescript/logic/chatComposition.logic.ts` | `__tests__/typescript/unit/req-003.chart-fanout.unit.test.ts` |
| REQ-004 model selection precedence | `typescript/logic/modelSelection.logic.ts` | `__tests__/typescript/unit/req-004.model-selection.unit.test.ts`, `__tests__/typescript/unit/dr-005.fallback-model-order.stability.unit.test.ts` |
| REQ-005 type contract location under `__types__` | `typescript/api/*`, `typescript/react/*`, `typescript/logic/*` | `__tests__/typescript/integration/req-005.contract-location.integration.test.ts` |
| REQ-006 core orchestrator remains page-agnostic (`activeDatasetId=null`) | `typescript/react/orchestrators/chatOrchestrator.ts` | `__tests__/typescript/integration/req-006.page-agnostic-dataset.wiring.integration.test.ts` |
| REQ-007 abort in-flight requests without stale updates | `typescript/react/hooks/useChat.hooks.ts` | `__tests__/typescript/integration/req-007.abort-unmount.integration.test.ts` |
| REQ-008 attachment policy and rejection behavior | `typescript/react/hooks/useChatSidebar.web.ts` | `__tests__/typescript/integration/req-008.invalid-attachments.integration.test.ts` |
| REQ-009 endpoint timeout semantics | `typescript/api/chatApi.ts` | `__tests__/typescript/integration/req-009.models-timeout.integration.test.ts`, `__tests__/typescript/integration/err-005.timeout-retryable-contract.integration.test.ts` |

| Boundary Rule | Primary module(s) | Required test file(s) |
| --- | --- | --- |
| AB-001 no cross-feature domain imports in ai-chat | `typescript/react/orchestrators/chatOrchestrator.ts` | `__tests__/typescript/integration/ab-001.no-cross-feature-domain-imports.integration.test.ts` |
| AB-002 page context injection happens in `screens/*` | `screens/*` + ai-chat contracts | `__tests__/typescript/integration/ab-002.screen-context-injection.integration.test.ts` |
| AB-003 logic layer remains framework agnostic | `typescript/logic/*` | `__tests__/typescript/integration/ab-003.logic-framework-agnostic.integration.test.ts` |
