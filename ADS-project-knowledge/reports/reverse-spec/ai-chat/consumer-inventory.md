# Consumer Inventory — ai-chat

Extraction date: 2026-06-09
Source: `web/nextjs/src/features/ai-chat/`

---

## Consumer 1: LandingPage

**Routes:** `/`, `/chat`
**Screen component:** `ui/screens/LandingPage/views/LandingPageScreen.tsx`

### Imports from ai-chat

| Import | Source File | Purpose |
|--------|-----------|---------|
| `ChatIntegration` (type) | `__types__/chat.types.ts` | Orchestrator return type |
| `useChatSurfaceOrchestrator` | `react/compositions/useChatSurface.orchestrator.ts` | Composition root |
| `ChatSidebar` | `react/views/components/ChatSidebar.tsx` | UI surface (via dynamic import, SSR disabled) |
| `ChatMessage` (type) | `__types__/chat.types.ts` | Store type definition |
| `ChatModelOption` (type) | `__types__/chat.types.ts` | Store type definition |
| `ScreenFeedback` (type) | `__types__/uiFeedback.types.ts` | Store type definition |
| `ChatStatePort` (type) | `__types__/chat.types.ts` | Adapter interface |
| `createInitialChatStoreCoreState` | `logic/chatStore.logic.ts` | Store initialization |
| `appendMessage` | `logic/chatStore.logic.ts` | State mutation logic |
| `appendInputHistory` | `logic/chatStore.logic.ts` | State mutation logic |
| `resolveHistoryCursor` | `logic/chatStore.logic.ts` | State mutation logic |

### Composition Configuration

| Port | Injected Value | Effect |
|------|---------------|--------|
| `useStatePort` | `useLandingChatStateAdapter` (isolated Zustand store) | Messages are page-scoped, not shared with other pages |
| `useChartActionsPort` | `useCopilotChartActionsAdapter` (recharts/AI chart store) | Chart specs route to the landing page chart workspace |
| `mode` | `"direct"` | API calls go to POST `/chat` (no dataset_id) |
| `activeDatasetId` | (omitted — defaults to null) | No dataset context ever sent |
| `apiAdapter` | (omitted — created from mode) | Default behavior |

### Key Architectural Choices

- Creates its OWN Zustand store instance (not the global singleton)
- Reuses ai-chat's pure logic functions for state mutations (no duplication)
- Wraps `ChatSidebar` in `LandingChatSidebar` component that passes the custom orchestrator
- Uses `dynamic(import, { ssr: false })` for client-only rendering

---

## Consumer 2: AgenticResearchPage

**Route:** `/agentic-research`
**Screen component:** `ui/screens/AgenticResearchPage/views/AgenticResearchPageScreen.tsx`

### Imports from ai-chat

| Import | Source File | Purpose |
|--------|-----------|---------|
| `ChatIntegration` (type) | `__types__/chat.types.ts` | Orchestrator return type |
| `useChatSurfaceOrchestrator` | `react/compositions/useChatSurface.orchestrator.ts` | Composition root |
| `useAiChatStateAdapter` | `react/state/adapters/aiChatState.adapter.ts` | Global store adapter |
| `ChatSidebar` | `react/views/components/ChatSidebar.tsx` | UI surface (via dynamic import, SSR disabled) |
| `ChatOrchestrator` (type) | `react/orchestrators/chatOrchestrator.ts` | Prop typing |

### Composition Configuration

| Port | Injected Value | Effect |
|------|---------------|--------|
| `useStatePort` | `useAiChatStateAdapter` (global singleton) | Messages persist across SPA navigations within session |
| `useChartActionsPort` | `useAgenticResearchChartActionsAdapter` | Chart specs route to the agentic-research chart store |
| `mode` | `"research"` | API calls go to POST `/chat-research` (includes dataset_id) |
| `useActiveDatasetId` | `useAgenticResearchSelectedDatasetId` | Dynamic dataset from agentic-research state |
| `apiAdapter` | (omitted — created from mode) | Default behavior |

### Key Architectural Choices

- Uses the GLOBAL `useAiChatStore` singleton (shared state across navigations)
- Injects a dynamic dataset hook (reactive — changes when user switches dataset)
- Charts go to a page-specific chart store (not the landing page's chart workspace)
- Uses `dynamic(import, { ssr: false })` for client-only rendering

---

## Consumer 3: ag-ui-chat (feature)

**Route:** `/ag-ui` (does NOT use ai-chat at runtime)
**Feature path:** `web/nextjs/src/features/ag-ui-chat/`

### Imports from ai-chat

| Import | Source File | Purpose |
|--------|-----------|---------|
| `UseChatChartActionsPort` (type only) | `__types__/chat.types.ts` | Type reuse for its own orchestrator typing |

### Composition Configuration

Not applicable. This consumer imports only a type interface for structural compatibility. It does NOT:
- Use any ai-chat hooks, stores, or components at runtime
- Route through `useChatSurfaceOrchestrator`
- Share any state with ai-chat

### Key Architectural Choices

- Type-only coupling: imports the port interface to ensure its own chart action adapter satisfies the same contract
- Runtime independent: uses CopilotKit's own chat runtime, not ai-chat's API layer
- Could be decoupled entirely by duplicating the type definition (but the shared type ensures interface compatibility)

---

## Consumer Topology Diagram

```
                     features/ai-chat/
                           |
          +----------------+------------------+
          |                |                  |
  LandingPage      AgenticResearchPage    ag-ui-chat
  (isolated store)  (global store)        (type-only)
  mode: "direct"    mode: "research"      no runtime dep
  chart: recharts   chart: ag-research
  dataset: null     dataset: dynamic
  routes: /, /chat  route: /agentic-research
```

---

## Compatibility Risks

1. **Shared logic functions (LandingPage):** The LandingPage imports and re-uses `appendMessage`, `appendInputHistory`, `resolveHistoryCursor`, and `createInitialChatStoreCoreState`. Any signature change to these functions breaks the LandingPage store. These are effectively frozen public API.

2. **ChatIntegration shape stability:** Both LandingPage and AgenticResearchPage depend on the shape returned by `useChatSurfaceOrchestrator`. Adding/removing fields from `ChatIntegration` affects all consumers.

3. **Port interface stability:** `ChatStatePort`, `UseChatChartActionsPort`, and `ChatApiDeps` are consumed by external adapters. Interface changes require coordinated updates across all page-level orchestrators.
