# Reverse-Spec Inventory: ai-chat

Feature path: `web/nextjs/src/features/ai-chat/`
Inventory date: 2026-06-09
Spec version found: 1.9.0 (draft, per `__specs__/ai-chat.spec.md`)

---

## Entrypoints

No barrel/index file exists. All consumption is via direct deep imports. The effective public surface (consumed externally) is:

### Types (consumed by screens and other features)
| Symbol | File | Line |
|--------|------|------|
| `ChatIntegration` | `__types__/chat.types.ts` | 167 |
| `ChatStatePort` | `__types__/chat.types.ts` | 103 |
| `ChatMessage` | `__types__/chat.types.ts` | 27 |
| `ChatModelOption` | `__types__/chat.types.ts` | 35 |
| `ScreenFeedback` | `__types__/uiFeedback.types.ts` | 4 |
| `ChatOrchestrator` (re-export of `ChatIntegration`) | `react/orchestrators/chatOrchestrator.ts` | 29 |

### Hooks / Compositions (consumed by screens)
| Symbol | File | Line |
|--------|------|------|
| `useChatSurfaceOrchestrator` | `react/compositions/useChatSurface.orchestrator.ts` | 56 |
| `useChatOrchestrator` | `react/orchestrators/chatOrchestrator.ts` | 17 |
| `useAiChatStateAdapter` | `react/state/adapters/aiChatState.adapter.ts` | 8 |

### Logic (consumed by LandingPage screen store)
| Symbol | File | Line |
|--------|------|------|
| `createInitialChatStoreCoreState` | `logic/chatStore.logic.ts` | 16 |
| `appendMessage` | `logic/chatStore.logic.ts` | 37 |
| `appendInputHistory` | `logic/chatStore.logic.ts` | 49 |
| `resolveHistoryCursor` | `logic/chatStore.logic.ts` | 64 |

### View Components (consumed by screens)
| Symbol | File | Line |
|--------|------|------|
| `ChatSidebar` (default export) | `react/views/components/ChatSidebar.tsx` | 26 |
| `ChatBar` (default export) | `react/views/components/ChatBar.tsx` | 16 |
| `CopilotChatSidebar` (default export) | `react/views/components/CopilotChatSidebar.tsx` | 10 |
| `UIFeedback` (default export) | `react/views/components/UIFeedback.tsx` | 27 |

---

## External Integrations

### Backend API Endpoints
| Endpoint | Method | Called From | Purpose |
|----------|--------|-------------|---------|
| `{baseUrl}/chat` | POST | `api/chatApi.ts:94` | Direct chat (no dataset context) |
| `{baseUrl}/chat-research` | POST | `api/chatApi.ts:75` | Research-mode chat (includes dataset_id) |
| `{baseUrl}/llm/gemini-models` | GET | `api/chatApi.ts:199` | Fetch available model list |

### Base URL Resolution
- Browser: `/api/ai` (Next.js proxy to avoid CORS) via `@/core/config/aiApi.ts`
- Server: `AI_API_URL` env var, fallback `NEXT_PUBLIC_AI_API_URL`, fallback `http://127.0.0.1:8000`

### Request Payload Shape (chat endpoints)
```json
{
  "message": "<string>",
  "attachments": [],
  "model": "<string|null>",
  "messages": [{ "role": "user|assistant", "content": "...", "attachments": [] }],
  "dataset_id": "<string|null>"
}
```

### Third-Party Services / SDKs
| Package | Used In | Purpose |
|---------|---------|---------|
| `@copilotkit/react-ui` | `CopilotChatSidebar.tsx` | Alternative CopilotKit-powered sidebar |
| `react-markdown` | `ChatSidebar.tsx` | Render assistant markdown |
| `remark-gfm` | `ChatSidebar.tsx` | GitHub-flavored markdown tables/autolinks |
| `zustand` | `aiChatStore.ts` | Global state container |
| `zustand/react/shallow` | `aiChatState.adapter.ts` | Shallow equality selector |

---

## State Management

### Primary Store: `useAiChatStore`
- **Location**: `react/state/zustand/aiChatStore.ts`
- **Library**: Zustand (global singleton via `create`)
- **State Shape** (`AiChatState`):

| Field | Type | Purpose |
|-------|------|---------|
| `messages` | `ChatMessage[]` | Ordered transcript |
| `inputHistory` | `string[]` | Previous user inputs for arrow-key navigation |
| `historyCursor` | `number \| null` | Current position in inputHistory |
| `isSending` | `boolean` | Request-in-flight flag |
| `modelOptions` | `ChatModelOption[]` | Available LLM models |
| `selectedModelId` | `string \| null` | Active model for requests |
| `isModelsLoading` | `boolean` | Model fetch in-progress |
| `screenFeedback` | `ScreenFeedback \| null` | Persistent inline feedback |

- **Actions**: `addMessage`, `addInputToHistory`, `moveHistoryCursor`, `resetHistoryCursor`, `setSending`, `setModelOptions`, `setSelectedModelId`, `setModelsLoading`, `setScreenFeedback`
- **Logic Delegation**: Store actions delegate computation to pure functions in `logic/chatStore.logic.ts`

### State Adapter Pattern
- `useAiChatStateAdapter` exposes store via a `ChatStatePort` contract (state + actions separated)
- Screens can provide their own state port implementation (e.g., `LandingPage` has `useLandingChatStateAdapter` backed by an isolated store)

### Local UI State
- `useChatUiState()` hook in `react/hooks/useChat.hooks.ts:157` manages ephemeral input-layer state: `value`, `showTooltip`, `attachments`

---

## Cross-Feature Dependencies

### Imports FROM Other Features
| Source Feature | Imported Symbol | Used In |
|----------------|-----------------|---------|
| `features/charts/contracts` | `ChartSpec` | `__types__/chat.types.ts`, `__types__/api.types.ts`, `__types__/logic/chatComposition.types.ts`, `logic/chatApiNormalization.logic.ts` |

### Imports FROM Core
| Source Module | Imported Symbol | Used In |
|---------------|-----------------|---------|
| `@/core/config/aiApi` | `getAiApiBaseUrl` | `api/chatApi.ts:15` |

### Exports TO Other Features / Screens (Consumed Externally)

| Consumer | Imported From ai-chat | Purpose |
|----------|----------------------|---------|
| `ui/screens/AgenticResearchPage/chat/orchestrators/` | `ChatIntegration`, `useChatSurfaceOrchestrator`, `useAiChatStateAdapter` | Composes research-mode chat with custom chart/dataset ports |
| `ui/screens/AgenticResearchPage/views/` | `ChatOrchestrator` type, `ChatSidebar` component | Renders chat sidebar in research page |
| `ui/screens/LandingPage/chat/orchestrators/` | `ChatIntegration`, `useChatSurfaceOrchestrator` | Composes landing chat with isolated store + CopilotKit chart actions |
| `ui/screens/LandingPage/chat/state/zustand/` | `ChatMessage`, `ChatModelOption`, `ScreenFeedback`, store logic functions | Implements isolated landing page store mirroring ai-chat store logic |
| `ui/screens/LandingPage/chat/state/adapters/` | `ChatStatePort` | Type contract for landing chat adapter |
| `ui/screens/LandingPage/chat/views/` | `ChatSidebar` component | Renders sidebar in landing page |

---

## Internal Structure Map

### `__types__/` (Type Contracts)
| File | Role | Framework-agnostic? |
|------|------|---------------------|
| `chat.types.ts` | Core domain types: `ChatMessage`, `ChatState`, `ChatStateActions`, `ChatApiDeps`, `ChatLogicDeps`, `ChatIntegration`, `ChatDeps`, port types | Yes |
| `api.types.ts` | API response/request shapes, error codes, runtime dependency injection | Yes |
| `uiFeedback.types.ts` | `ScreenFeedback` and `NotificationFeedback` types | Yes |
| `logic/chatComposition.types.ts` | Input types for composition logic | Yes |
| `logic/chatOrchestrator.types.ts` | Input types for orchestrator factory | Yes |
| `logic/chatStore.types.ts` | Store state and action input types | Yes |
| `logic/chatSubmission.types.ts` | Submit/history normalization input types | Yes |
| `logic/modelSelection.types.ts` | Model selection result and input types | Yes |

### `api/` (API Layer)
| File | Role | Framework-agnostic? |
|------|------|---------------------|
| `chatApi.ts` | HTTP transport: `sendChatMessage`, `sendChatMessageDirect`, `fetchChatModels` | Yes (uses injected fetch) |
| `chatApi.adapter.ts` | Factory: `createChatApiAdapter` selects research vs direct mode | Yes |

### `logic/` (Business Logic)
| File | Role | Framework-agnostic? |
|------|------|---------------------|
| `chatApiNormalization.logic.ts` | `normalizeChatApiResult`, `parseJsonPayload`, `normalizeTextResult`, `createModelFetchErrorResult` | Yes |
| `chatComposition.logic.ts` | `mapChatStateWithDataset`, `createOnMessageReceived`, `composeChatStateActions` | Yes |
| `chatOrchestrator.logic.ts` | `createChatApiDeps`, `createChatDeps` (dependency factories) | Yes |
| `chatStore.logic.ts` | `createInitialChatStoreCoreState`, `appendMessage`, `appendInputHistory`, `resolveHistoryCursor` | Yes |
| `chatSubmission.logic.ts` | `normalizeSubmissionValue`, `buildChatHistoryWindow`, `createUserChatMessage`, `createAssistantChatMessage`, `shouldRestoreDraftValue` | Yes |
| `modelSelection.logic.ts` | `resolveFallbackModelSelection`, `resolveFetchedModelSelection`, `FALLBACK_CHAT_MODELS` | Yes |

### `react/` (React/Framework Layer)
| File | Role | Framework-agnostic? |
|------|------|---------------------|
| `hooks/useChat.hooks.ts` | `useChatUiState`, `useChatLogic`, `useChatIntegration`, `setFallbackModels`, `setFetchedModels`, `resolveSubmitFeedback` | No (React hooks, useEffect, useRef) |
| `hooks/useChatSidebar.web.ts` | `useChatSidebarUi` (auto-scroll, drag-drop, copy) | No (React hooks, DOM APIs) |
| `compositions/useChatSurface.orchestrator.ts` | `useChatSurfaceOrchestrator` (top-level composition wiring all deps) | No (React useMemo) |
| `orchestrators/chatOrchestrator.ts` | `useChatOrchestrator` (simple default orchestrator) | No (React hook) |
| `state/zustand/aiChatStore.ts` | `useAiChatStore` Zustand store definition | No (Zustand) |
| `state/adapters/aiChatState.adapter.ts` | `useAiChatStateAdapter` (store-to-port adapter) | No (Zustand + React) |
| `views/components/ChatBar.tsx` | Chat input bar component | No (JSX) |
| `views/components/ChatSidebar.tsx` | Full sidebar with messages, model selector, feedback | No (JSX) |
| `views/components/CopilotChatSidebar.tsx` | CopilotKit-powered alternative sidebar | No (JSX + CopilotKit) |
| `views/components/UIFeedback.tsx` | Inline error/warning/info feedback panel | No (JSX) |

---

## Test Coverage

### Test Location
All tests live in `web/nextjs/src/__tests__/features/ai-chat/` (mirrored structure, NOT colocated).

### Unit Tests (logic layer)
| Test File | Covers |
|-----------|--------|
| `logic/chatApiNormalization.logic.unit.test.ts` | Response normalization, JSON parsing |
| `logic/chatOrchestrator.logic.unit.test.ts` | Dependency factory functions |
| `logic/chatStore.logic.unit.test.ts` | Store state mutations |
| `logic/chatSubmission.logic.unit.test.ts` | Input normalization, history window, message creation |

### Unit Tests (react layer)
| Test File | Covers |
|-----------|--------|
| `react/hooks/useChat.hooks.unit.test.tsx` | `useChatLogic`, `useChatIntegration` behavior |
| `react/hooks/useChat.hooks.runtime.unit.test.tsx` | Runtime dep injection (`now`, `createId`) |
| `react/hooks/useChatSidebar.web.runtime.unit.test.tsx` | Sidebar UI behaviors (scroll, copy, drag-drop) |
| `react/orchestrators/chatOrchestrator.unit.test.tsx` | Default orchestrator wiring |
| `react/state/adapters/aiChatState.adapter.unit.test.tsx` | State adapter shape |
| `react/state/zustand/aiChatStore.unit.test.ts` | Store creation and actions |
| `react/views/components/ChatBar.unit.test.tsx` | ChatBar rendering and interaction |
| `react/views/components/ChatSidebar.unit.test.tsx` | ChatSidebar rendering |
| `react/views/components/CopilotChatSidebar.unit.test.tsx` | CopilotKit sidebar rendering |

### API Unit Tests
| Test File | Covers |
|-----------|--------|
| `api/chatApi.unit.test.ts` | HTTP transport, timeout, error handling |
| `api/chatApi.adapter.unit.test.ts` | Adapter factory mode selection |

### Spec-Requirement-Tagged Tests (unit)
| Test File | Spec Ref |
|-----------|----------|
| `unit/dr-002.history-window.boundary.unit.test.ts` | DR-002 |
| `unit/dr-003.history-cursor.totality.unit.test.ts` | DR-003 |
| `unit/dr-005.fallback-model-order.stability.unit.test.ts` | DR-005 |
| `unit/req-003.chart-fanout.unit.test.ts` | REQ-003 |
| `unit/req-004.model-selection.unit.test.ts` | REQ-004 |

### Integration Tests (architecture boundary and behavior)
| Test File | Spec Ref |
|-----------|----------|
| `integration/ab-001.no-cross-feature-domain-imports.integration.test.ts` | AB-001 |
| `integration/ab-002.screen-context-injection.integration.test.ts` | AB-002 |
| `integration/ab-003.logic-framework-agnostic.integration.test.ts` | AB-003 |
| `integration/dr-004.err-001.invalid-payload-normalization.integration.test.ts` | DR-004, ERR-001 |
| `integration/err-002.fetch-models-fallback.integration.test.ts` | ERR-002 |
| `integration/err-004.clipboard-failure.integration.test.ts` | ERR-004 |
| `integration/err-005.timeout-retryable-contract.integration.test.ts` | ERR-005 |
| `integration/req-001.empty-input-short-circuit.integration.test.ts` | REQ-001 (DR-001) |
| `integration/req-001.submit-order.integration.test.ts` | REQ-001 |
| `integration/req-002.sending-reset.integration.test.ts` | REQ-002 |
| `integration/req-005.contract-location.integration.test.ts` | REQ-005 |
| `integration/req-006.page-agnostic-dataset.wiring.integration.test.ts` | REQ-006 |
| `integration/req-007.abort-unmount.integration.test.ts` | REQ-007 |
| `integration/req-008.invalid-attachments.integration.test.ts` | REQ-008 |
| `integration/req-009.models-timeout.integration.test.ts` | REQ-009 |

### Shared Fixtures
| File | Purpose |
|------|---------|
| `fixtures/chatLogicDeps.fixture.ts` | Provides mock `ChatLogicDeps` for test harnesses |

### Gaps / Not Tested
- `UIFeedback.tsx` does not have a dedicated unit test file (though it may be covered transitively via ChatSidebar tests).
- No E2E/browser tests visible in this test tree.
- `modelSelection.logic.ts` is covered by `unit/req-004` and `unit/dr-005` but has no dedicated `logic/modelSelection.logic.unit.test.ts`.

---

## Tech Stack Notes

| Concern | Technology | Notes |
|---------|-----------|-------|
| Framework | React 18+ (Next.js App Router) | `"use client"` directives on view components |
| State | Zustand | Global singleton store, shallow selectors |
| Markdown rendering | `react-markdown` + `remark-gfm` | In `ChatSidebar` only |
| CopilotKit | `@copilotkit/react-ui` | Alternate sidebar; thin wrapper only |
| Dependency injection | Port/adapter pattern via types | State ports, chart action ports, runtime deps |
| API transport | Raw `fetch` (injected) | No Axios, no TanStack Query |
| Testing | Vitest/Jest (inferred from `.unit.test.ts` and `.integration.test.ts` suffixes) | Fixture-based, runtime-dep-injected |
| Architecture pattern | Orc-BASH (Orchestrator -> Business Logic -> API -> State -> Hooks) | Logic layer is fully framework-agnostic |
| Error contract | `ScreenFeedback` type | Persistent inline feedback, not toast-based |
| Model fallback | Hardcoded `FALLBACK_CHAT_MODELS` in `modelSelection.logic.ts` | Gemini 3 Flash Preview, 3.1 Pro Preview, 3 Pro Preview, 2.5 Pro |
