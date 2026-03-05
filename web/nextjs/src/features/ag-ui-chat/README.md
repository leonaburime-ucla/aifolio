# copilot-chat

CopilotKit feature slice for chat UI, AG-UI transport, frontend tool registration, chart injection, and chat message persistence.

Open issues are tracked in [TODO.md](/Users/la/Desktop/Programming/AIfolio/web/nextjs/src/features/ag-ui-chat/TODO.md).

## Purpose

This feature owns:
- Copilot runtime wiring (`/api/copilotkit` -> backend `/agui`)
- Chat UI shell + assistant message rendering
- Frontend tool registration (`useCopilotAction`)
- Assistant payload parsing (`{ message, chartSpec }`)
- Chart bridge side effects (assistant message -> chart stores)
- Persisting chat history to local storage via Zustand

This feature does **not** run model inference itself; backend inference lives in Python (`/agui` route and related services).

## High-Level Flow

1. `CopilotChatProvider` mounts `CopilotKit` with runtime config.
2. `CopilotEffectsProvider` mounts side effects:
3. `useCopilotChartBridgeOrchestrator` (message -> chart store)
4. `useCopilotMessagePersistenceOrchestrator` (live <-> persisted messages)
5. `CopilotFrontendTools` + `AgenticResearchAiTools` register tool handlers.
6. `CopilotSidebar` renders `CopilotChat`.
7. User sends message; Copilot runtime calls Next route `/api/copilotkit`.
8. Next route (outside this folder) uses `createCopilotAppRouterHandler()` from this feature.
9. Runtime forwards to backend AG-UI endpoint (`/agui`).
10. Assistant response comes back as text payload (often JSON string with `message` and `chartSpec`).
11. Assistant renderer strips transport JSON for display text.
12. Chart bridge parses same payload and injects chart specs into target store.

## File Map

### adapters
- `adapters/copilotRuntime.adapter.ts`
  - Server-only runtime adapter for Next App Router API endpoint.
  - Creates `CopilotRuntime` + `LangGraphHttpAgent`.
  - Forwards to `${backendBaseUrl}/agui`.

### config
- `config/copilotRuntime.config.ts`
  - Client/server runtime config factories.
  - Defines `runtimeUrl` and agent id.
- `config/frontendTools.config.ts`
  - Tool name constants.
  - Route alias mapping + route validation helpers.

### orchestrators
- `orchestrators/frontendTools.logic.ts`
  - Pure handlers for tool logic (normalize chart payload, route resolution, tab resolution, PyTorch train pass-through).
  - No React hooks.
- `react/orchestrators/copilotAssistantMessage.orchestrator.ts`
  - Legacy assistant payload processor used by `CopilotAssistantMessageLegacy`.
  - Parses assistant JSON and pushes chart spec(s) into global chart store.
- `orchestrators/copilotChartBridge.orchestrator.ts`
  - AG-UI message bridge side effect.
  - Watches Copilot message context, parses assistant payload, and routes chart specs to:
  - agentic-research chart store when `activeTab === "agentic-research"`
  - otherwise global recharts store.
  - Contains processed-message dedupe by message id.
- `orchestrators/copilotMessagePersistence.orchestrator.ts`
  - Syncs live Copilot messages with persisted message store.
  - Handles initial hydration, no-op guards, and sync dedupe by serialized equality.

### state
- `state/zustand/copilotMessageStore.ts`
  - Persisted local store for chat messages and hydration flag.
  - Uses `zustand/persist`.
- `state/adapters/copilotMessageState.adapter.ts`
  - Narrow state port exposing `messages`, `hasHydrated`, `setMessages`.
  - Keeps components/hooks decoupled from direct store imports.

### utils
- `utils/copilotAssistantPayload.util.ts`
  - Core parser/normalizer for assistant transport payload.
  - Validates chart schema and allowed chart types.
  - Exposes:
  - `parseCopilotAssistantPayload`
  - `extractCopilotDisplayMessage`
  - `normalizeChartSpecInput`
- `utils/messagePersistence.util.ts`
  - Safe message serialization helpers for persistence.
  - Filters to stable `TextMessage` user/assistant entries.
  - Drops transient/internal/tool entries to avoid bad restore behavior.

### views/providers
- `views/providers/CopilotChatProvider.tsx`
  - Feature-local root provider (`CopilotKit` + effects provider).
- `views/providers/CopilotEffectsProvider.tsx`
  - Central side-effect mount point.
  - Mounts chart bridge, persistence orchestrator, tool registration components.

### views/components
- `views/components/CopilotSidebar.tsx`
  - Visible chat sidebar shell using `CopilotChat`.
  - Chooses assistant renderer by mode (`legacy` or `ag-ui`).
- `views/components/CopilotAssistantMessage.tsx`
  - AG-UI assistant renderer.
  - Displays user-facing message text by stripping transport JSON.
  - Contains current debug log for raw/parsed assistant payload.
- `views/components/CopilotAssistantMessageLegacy.tsx`
  - Legacy renderer for older chat mode.
  - Uses legacy orchestrator to apply chart side effects from message content.
- `views/components/CopilotFrontendTools.tsx`
  - Registers frontend tools with `useCopilotAction`:
  - `add_chart_spec`
  - `clear_charts`
  - `navigate_to_page`
  - `train_pytorch_model`

### types
- `types/copilotChat.types.ts`
  - Shared runtime config and agent type definitions.

### specs
- `__specs__/copilot-chat.spec.md`
  - Feature-level contract/spec narrative.
- `__specs__/copilot-chat.api.spec.ts`
  - API-layer expected behavior/spec fragments.
- `__specs__/copilot-chat.state.spec.ts`
  - State-layer expected behavior/spec fragments.
- `__specs__/copilot-chat.ui.spec.ts`
  - UI-layer expected behavior/spec fragments.
- `__specs__/copilot-chat.orchestrator.spec.ts`
  - Orchestrator-layer expected behavior/spec fragments.

## Message/Chart Processing Paths

### AG-UI mode (`CopilotAssistantMessage`)

1. Assistant message arrives in Copilot context.
2. Renderer shows `extractCopilotDisplayMessage(message.content)`.
3. Chart bridge separately watches Copilot message context.
4. Bridge parses same content with `parseCopilotAssistantPayload`.
5. If `chartSpec` exists, bridge injects chart(s) into selected store.

### Legacy mode (`CopilotAssistantMessageLegacy`)

1. Assistant renderer parses content via legacy orchestrator.
2. Legacy effect directly injects chart spec(s) into global chart store.

## Known Couplings

- AG-UI mode currently depends on message availability in `useCopilotMessagesContext()` for chart injection timing.
- Persistence and chart bridge observe different Copilot internals (`useCopilotChatInternal` vs `useCopilotMessagesContext`), which can diverge during transient states.
- `CopilotEffectsProvider` also mounts `AgenticResearchAiTools` from another feature, so AG-UI tool scope crosses feature boundaries at provider level.

## Quick Debug Anchors

- Assistant payload parse:
  - `views/components/CopilotAssistantMessage.tsx`
  - `utils/copilotAssistantPayload.util.ts`
- Chart injection:
  - `orchestrators/copilotChartBridge.orchestrator.ts`
- Message persistence:
  - `orchestrators/copilotMessagePersistence.orchestrator.ts`
  - `utils/messagePersistence.util.ts`
- Runtime forwarding:
  - `adapters/copilotRuntime.adapter.ts`
  - `config/copilotRuntime.config.ts`
