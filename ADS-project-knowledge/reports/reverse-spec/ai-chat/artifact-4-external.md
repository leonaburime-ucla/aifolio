# Artifact 4: External Systems, Consumers, and Human Workflows — ai-chat

Feature path: `web/nextjs/src/features/ai-chat/`
Extraction date: 2026-06-09
Pass: 4 (External Systems and Human Workflows)
Methodology: `AI-Dev-Shop-speckit/skills/reverse-spec/SKILL.md` v2.0.0
Prior artifacts consumed: `artifact-1-core-logic.md` (REQ-CHAT-001–030), `artifact-2-data-access.md` (REQ-CHAT-031–050), `artifact-3-boundaries.md` (REQ-CHAT-051–073)

---

## Consumer Inventory

### Internal Consumers (pages/screens importing from `features/ai-chat`)

| Consumer | Type | Imports Used | Customization | Evidence |
|----------|------|-------------|---------------|----------|
| AgenticResearchPage | page screen | `ChatIntegration` (type), `useChatSurfaceOrchestrator`, `useAiChatStateAdapter`, `ChatSidebar` (dynamic), `ChatOrchestrator` (type) | Custom chart port (agentic-research chart store), custom dataset ID hook, `mode: "research"` | `ui/screens/AgenticResearchPage/chat/orchestrators/agenticResearchChatOrchestrator.ts:1-24`, `ui/screens/AgenticResearchPage/views/AgenticResearchPageScreen.tsx:27-30` |
| LandingPage | page screen | `ChatIntegration` (type), `useChatSurfaceOrchestrator`, `ChatSidebar` (via LandingChatSidebar wrapper), `ChatMessage`, `ChatModelOption`, `ScreenFeedback`, `createInitialChatStoreCoreState`, `appendMessage`, `appendInputHistory`, `resolveHistoryCursor`, `ChatStatePort` | Isolated store (separate Zustand instance), CopilotKit chart actions port, `mode: "direct"`, no dataset ID | `ui/screens/LandingPage/chat/orchestrators/landingChatOrchestrator.ts:1-15`, `ui/screens/LandingPage/chat/state/zustand/landingChatStore.ts:1-59` |
| ag-ui-chat feature | cross-feature type | `UseChatChartActionsPort` (type only) | Type reuse for its own orchestrator typing | `features/ag-ui-chat/__types__/react/orchestrators/copilotAssistantMessageOrchestrator.types.ts:1` |

### Consumer Topology Summary

```
app/page.tsx ─────────> LandingPageScreen ─────> LandingChatSidebar ──> ChatSidebar
app/chat/page.tsx ────> LandingPageScreen ─────> (same as above)
app/agentic-research/ > AgenticResearchScreen ─> ChatSidebar (dynamic)

                        Both use: useChatSurfaceOrchestrator (composition root)
                        Difference: state port, chart port, dataset hook, API mode
```

### Pages Sharing the Same Route

| Route | Page Component | Chat Store | API Mode | Dataset | Chart Target |
|-------|---------------|-----------|----------|---------|--------------|
| `/` | `LandingPageScreen` | Isolated (`useLandingChatStore`) | `"direct"` → POST `/chat` | null (always) | CopilotKit chart store (`recharts/ai/`) |
| `/chat` | `LandingPageScreen` | Isolated (`useLandingChatStore`) | `"direct"` → POST `/chat` | null (always) | CopilotKit chart store (`recharts/ai/`) |
| `/agentic-research` | `AgenticResearchPageScreen` | Global (`useAiChatStore`) | `"research"` → POST `/chat-research` | Dynamic (from `useAgenticResearchStateAdapter`) | Agentic Research chart store |
| `/ag-ui` | `AgUiPage` | Does NOT use ai-chat (uses ag-ui-chat/CopilotKit) | n/a | n/a | n/a |

---

## Phase 10: External Systems

### External SaaS Configuration

This feature has minimal external SaaS dependencies from the frontend perspective:

| External System | Access Classification | Notes |
|----------------|----------------------|-------|
| Backend AI API (Python/FastAPI) | Tool-accessible (via HTTP) | Three endpoints: `/chat`, `/chat-research`, `/llm/gemini-models` |
| Google Gemini (via backend) | Inaccessible (frontend has no direct access) | Model list/names come from backend; frontend never calls Gemini directly |
| CopilotKit Runtime (alternative path) | Config-file-accessible (`@copilotkit/react-ui`) | Independent; shares no state with ai-chat |

No billing, identity, email, search, feature flag, or analytics providers are consumed by this feature.

### Infrastructure-as-Behavior

Covered in prior passes (REQ-CHAT-050, REQ-CHAT-060, REQ-CHAT-061). No additional infrastructure behavior discovered.

---

## Requirements Extracted This Pass

---

### REQ-CHAT-074: Consumer composition via `useChatSurfaceOrchestrator`

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** high
**Criticality reason:** fragile_consumer (all pages compose through this single orchestrator hook)
**Risk tags:** —

**Source evidence:**
- `react/compositions/useChatSurface.orchestrator.ts:56-114` (implementation)
- `ui/screens/AgenticResearchPage/chat/orchestrators/agenticResearchChatOrchestrator.ts:17-24` (consumer)
- `ui/screens/LandingPage/chat/orchestrators/landingChatOrchestrator.ts:9-15` (consumer)

**Observed behavior:** `useChatSurfaceOrchestrator` is the sole composition root for wiring the chat feature into any page. It accepts injectable ports via its options parameter:
- `useStatePort` — which store to use (global or isolated)
- `useChartActionsPort` — where chart specs are dispatched
- `useActiveDatasetId` / `activeDatasetId` — dataset context source
- `mode` — selects API endpoint (`"research"` or `"direct"`)
- `apiAdapter` — optional pre-built API deps override

The orchestrator assembles `ChatDeps` from these ports and calls `useChatIntegration(deps)` to produce the final `ChatIntegration` interface.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `UseChatSurfaceOptions` — all ports are optional with sensible defaults (`useAiChatStateAdapter`, empty chart actions, null dataset, `"research"` mode)
- Output: `ChatIntegration` — the unified interface consumed by `ChatSidebar` and `ChatBar`
- Side effects: not_applicable (composition only — side effects occur within `useChatIntegration`)
- Invariants: all ports resolve via hooks (adhering to Rules of Hooks); the `logic` dependency is a stable memoized object; defaults ensure the feature works without any customization
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- In non-React frameworks, this composition pattern maps to: Vue composable factory, Svelte writable store factory, Angular service with DI tokens. The key contract is: consumer provides ports, orchestrator assembles the integration.
- The `useMemo` wrappers provide React-specific memoization to prevent unnecessary re-renders; equivalent frameworks need their own reactivity optimization.

---

### REQ-CHAT-075: ChatSidebar accepts orchestrator as injectable prop

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** high
**Criticality reason:** fragile_consumer (primary UI surface; every page depends on this injection pattern)
**Risk tags:** —

**Source evidence:**
- `react/views/components/ChatSidebar.tsx:21-24` (type — `ChatSidebarProps`)
- `react/views/components/ChatSidebar.tsx:26-29` (implementation — default `chatOrchestrator = useChatOrchestrator`)
- `ui/screens/AgenticResearchPage/views/AgenticResearchPageScreen.tsx:80-84` (consumer — passes custom orchestrator)
- `ui/screens/LandingPage/chat/views/LandingChatSidebar.tsx:6-9` (consumer — passes landing orchestrator)

**Observed behavior:** `ChatSidebar` accepts a `chatOrchestrator` prop (type: `() => ChatOrchestrator`). It calls this function at render time to obtain the full `ChatIntegration` interface. This prop-injection pattern allows any page to:
1. Use the default orchestrator (global store, null dataset)
2. Pass a custom orchestrator that wires different stores/ports

The `ChatBar` component follows the same pattern.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `chatOrchestrator?: () => ChatOrchestrator` prop (defaults to `useChatOrchestrator`)
- Output: renders full chat UI using the provided integration
- Side effects: not_applicable (rendering)
- Invariants: the orchestrator IS a hook (called during render); it must follow Rules of Hooks; the prop defaults to a sensible global configuration
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- React: `chatOrchestrator` is a hook passed as a prop (render-time invocation). This is a "hook injection" pattern.
- Vue: equivalent is a composable passed as prop or resolved via `provide/inject`. The composable returns reactive state.
- Svelte: equivalent is a store factory passed as prop to the component. Component subscribes to the returned store.
- Angular: equivalent is an injection token providing a service instance with observables.
- Key constraint: the injected function MUST be called exactly once per render cycle and the reference should be stable.

---

### REQ-CHAT-076: LandingPage uses isolated chat store

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (architectural boundary — prevents state cross-contamination)
**Risk tags:** —

**Source evidence:**
- `ui/screens/LandingPage/chat/state/zustand/landingChatStore.ts:31-57` (implementation — separate `create()`)
- `ui/screens/LandingPage/chat/state/adapters/landingChatState.adapter.ts:8-37` (adapter)
- `ui/screens/LandingPage/chat/orchestrators/landingChatOrchestrator.ts:10` (wiring — `useStatePort: useLandingChatStateAdapter`)

**Observed behavior:** The LandingPage creates its own Zustand store instance (`useLandingChatStore`) separate from the global `useAiChatStore`. This store:
1. Uses the same initial state factory (`createInitialChatStoreCoreState({})`)
2. Delegates all state mutations to the same pure logic functions (`appendMessage`, `appendInputHistory`, `resolveHistoryCursor`)
3. Exposes the same `ChatStatePort` interface via `useLandingChatStateAdapter`
4. Is completely isolated — mutations on the landing page do not affect the agentic-research page and vice versa

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `useLandingChatStateAdapter()` called by `useChatSurfaceOrchestrator`
- Output: `ChatStatePort` backed by an isolated store instance
- Side effects: not_applicable
- Invariants: LandingPage messages never appear on AgenticResearchPage; store instances are separate singletons; both implement the same `ChatStatePort` interface; logic functions are shared (not duplicated)
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable (separate store, no cross-store interaction)
- Auth requirement: not_applicable

**Migration notes:**
- The "isolated vs shared" store topology is a configuration choice at composition time, not a code difference. Any framework implementation must support: (a) a global singleton store shared across multiple page components, AND (b) per-page isolated store instances. The `UseChatStatePort` abstraction enables this without changing hook/composable logic.

---

### REQ-CHAT-077: AgenticResearchPage uses shared global store with custom chart/dataset ports

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (composition pattern correctness)
**Risk tags:** —

**Source evidence:**
- `ui/screens/AgenticResearchPage/chat/orchestrators/agenticResearchChatOrchestrator.ts:17-24` (implementation)
- `features/agentic-research/react/state/adapters/chartActions.adapter.ts:19-34` (chart port)
- `features/agentic-research/react/state/adapters/agenticResearchState.adapter.ts` (dataset source)

**Observed behavior:** The AgenticResearchPage orchestrator:
1. Uses the global `useAiChatStateAdapter` (shared store — messages persist across SPA navigations within this page)
2. Injects `useAgenticResearchChartActionsAdapter` — chart specs go to the agentic-research chart store (not the global landing/CopilotKit chart store)
3. Injects `useAgenticResearchSelectedDatasetId` — reads the currently selected dataset from the agentic-research state
4. Sets `mode: "research"` — routes API calls to `/chat-research` endpoint (includes `dataset_id`)

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `useChatSurfaceOrchestrator({ useStatePort: useAiChatStateAdapter, useChartActionsPort: useAgenticResearchChartActionsAdapter, useActiveDatasetId: useAgenticResearchSelectedDatasetId, mode: "research" })`
- Output: `ChatIntegration` with research-mode API and dataset-aware payloads
- Side effects: chart specs are written to the agentic-research chart store; API calls include the active dataset ID
- Invariants: global store state is shared — navigating away and back preserves messages (until page refresh); chart specs go to the correct per-page store; dataset ID is reactive (changes when user switches dataset)
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-078: LandingPage uses CopilotKit chart actions port

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (chart routing correctness)
**Risk tags:** —

**Source evidence:**
- `ui/screens/LandingPage/chat/orchestrators/landingChatOrchestrator.ts:3` (import)
- `ui/screens/LandingPage/chat/orchestrators/landingChatOrchestrator.ts:11` (wiring — `useChartActionsPort: useCopilotChartActionsAdapter`)
- `features/recharts/react/ai/state/adapters/chartActions.adapter.ts:7-11` (implementation)

**Observed behavior:** The LandingPage injects `useCopilotChartActionsAdapter` as the chart port. This adapter:
1. Reads `addChartSpec` from `useAiChartStore` (the recharts/AI chart store)
2. Also exposes `clearChartSpecs` (not used by ai-chat, but available)
3. Chart specs from the landing chat go to the `ChartsWorkspaceSurface` displayed on the same page

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `useCopilotChartActionsAdapter()` returns `{ addChartSpec, clearChartSpecs }`
- Output: chart specs from AI responses are added to the landing page's chart workspace
- Side effects: chart store mutation (chart appears in the workspace grid)
- Invariants: landing page charts are separate from agentic-research charts; both satisfy the `ChatChartActionsPort` interface (`{ addChartSpec }`)
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-079: ChatSidebar is loaded with SSR disabled (client-only)

**Layer:** infrastructure
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (deployment correctness — prevents SSR hydration failures)
**Risk tags:** environmental_contract

**Source evidence:**
- `ui/screens/LandingPage/views/LandingPageScreen.tsx:15-19` (implementation — `dynamic(() => import(...), { ssr: false })`)
- `ui/screens/AgenticResearchPage/views/AgenticResearchPageScreen.tsx:27-30` (implementation — same pattern)

**Observed behavior:** Both page screens load the chat sidebar via Next.js `dynamic()` with `{ ssr: false }`. This:
1. Prevents server-side rendering of the chat component tree
2. Avoids hydration mismatches from hooks that access `window`, `navigator`, or `document`
3. The sidebar renders only in the browser after the initial page load
4. No loading fallback is provided (renders nothing until client-side import resolves)

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: page mount in browser environment
- Output: chat sidebar renders only after client-side JavaScript hydration
- Side effects: initial page load shows no chat sidebar until JS loads
- Invariants: chat hooks REQUIRE browser APIs (window, navigator, FileReader); SSR would fail; any framework with SSR must defer chat rendering to the client
- Error cases: if JavaScript fails to load, chat sidebar never appears
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- Vue/Nuxt: use `<ClientOnly>` wrapper or `defineAsyncComponent` with SSR: false equivalent.
- Svelte/SvelteKit: use `{#if browser}` guard or dynamic import in `onMount`.
- Angular/SSR: use `isPlatformBrowser` guard in component initialization.
- The underlying requirement: chat component tree MUST NOT execute during server-side rendering.

---

### REQ-CHAT-080: `.web.ts` suffix indicates platform-specific module

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (multi-platform architecture signal)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChatSidebar.web.ts` (sole file with this suffix in ai-chat)
- No `.native.ts` or `.desktop.ts` counterpart exists

**Observed behavior:** The file `useChatSidebar.web.ts` uses the `.web.ts` suffix to signal that it contains browser/web-specific behavior:
- `window.requestAnimationFrame` / `cancelAnimationFrame` — DOM animation API
- `navigator.clipboard.writeText` — clipboard API
- `FileReader.readAsDataURL` — file processing API
- React `DragEvent` handling — browser drag-and-drop

The suffix indicates architectural intent: this module could have a `.native.ts` (React Native) or `.desktop.ts` (Electron/Tauri) counterpart providing platform-equivalent behavior. Currently, only the `.web.ts` variant exists.

All browser APIs in this module are injectable via `ChatSidebarRuntimeDeps`, enabling both testing and future platform abstraction.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: browser environment with Web APIs
- Output: scroll management, clipboard copy, file drag-drop
- Side effects: DOM manipulation (scroll position), clipboard write, file read
- Invariants: all browser APIs are injectable via runtime deps; the module is explicitly marked as web-specific
- Error cases: all API failures are handled gracefully (clipboard → silent fail, FileReader → discard file)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- For multi-platform builds (React Native, desktop), create a `useChatSidebar.native.ts` with equivalent behavior using platform APIs (e.g., React Native Clipboard, document picker instead of FileReader).
- For non-React frameworks: the `.web` module's behavior maps to lifecycle hooks + DOM event handlers. Extract as a platform-agnostic interface with web-specific adapter.

---

## Client-Side Implicit Contracts

---

### REQ-CHAT-081: Messages render in chronological order

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (fundamental UX expectation)
**Risk tags:** —

**Source evidence:**
- `react/views/components/ChatSidebar.tsx:116-149` (implementation — `messages.map(...)`)
- `logic/chatStore.logic.ts:37-41` (implementation — `appendMessage` appends to end)
- `integration/req-001.submit-order.integration.test.ts:8` (test — ordering invariant)

**Observed behavior:** Messages render via `messages.map()` which preserves array order. New messages are always appended to the end via `appendMessage` (spread + append). The UI never reorders, inserts, or removes messages from the middle. Users see messages in the exact order they were added — oldest at top, newest at bottom.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `messages: ChatMessage[]` from state
- Output: rendered list in array index order (chronological)
- Side effects: not_applicable
- Invariants: messages are append-only; no reordering, editing, or deletion is supported; array order equals chronological order
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-082: Sending indicator shown during in-flight request

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (user knows system is working)
**Risk tags:** —

**Source evidence:**
- `react/views/components/ChatSidebar.tsx:150-155` (implementation — spinner conditional on `isSending`)
- `react/views/components/ChatBar.tsx:100-101` (implementation — submit button `disabled={isSending}`)
- `integration/req-002.sending-reset.integration.test.ts:62-123` (test — isSending lifecycle)

**Observed behavior:** When `isSending === true`:
1. A spinning indicator with text "Working" appears below the last message in the sidebar
2. The "Send" button in ChatBar is disabled (`disabled:cursor-not-allowed disabled:bg-zinc-500`)
3. The model selector remains enabled (user can change model for next request)
4. The textarea remains editable (user can type ahead)

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `isSending: boolean` from state
- Output: visual spinner + disabled send button when true; normal state when false
- Side effects: not_applicable (rendering)
- Invariants: `isSending` is the sole signal for request-in-flight; it is set true before API call and false in finally block; spinner renders at bottom of message list; send button is the only element disabled
- Error cases: stale `isSending` after navigate-away (see REQ-CHAT-044)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-083: Model selector reflects available options and current selection

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (model selection UX)
**Risk tags:** —

**Source evidence:**
- `react/views/components/ChatSidebar.tsx:78-96` (implementation — `<select>` element)
- `react/hooks/useChat.hooks.ts:402-437` (implementation — `refetchModels` populates options)

**Observed behavior:** The model selector is a native `<select>` element:
1. Displays all `modelOptions` as `<option>` elements with `model.label` as display text
2. `selectedModelId` is the controlled `value`
3. When `isModelsLoading` is true OR `modelOptions.length === 0`, the select is disabled
4. Empty state shows "Loading models..." (if loading) or "No models available" (if not loading)
5. Changing selection calls `setSelectedModelId(event.target.value || null)` — empty string maps to null
6. Has `aria-label="Select AI model"` for accessibility

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `{ modelOptions, selectedModelId, isModelsLoading }` from state
- Output: rendered select with options; disabled when loading or empty
- Side effects: `setSelectedModelId` called on change (updates state for next request)
- Invariants: selection affects the `model` field in the next API request (captured at submit time); empty selection maps to null; disabled state prevents user interaction during loading
- Error cases: no models + not loading → shows "No models available"
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-084: Input history navigation with arrow keys

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** low
**Criticality reason:** internal_only (UX convenience)
**Risk tags:** —

**Source evidence:**
- `react/views/components/ChatBar.tsx:80-88` (implementation — `onKeyDown` handler)
- `unit/dr-003.history-cursor.totality.unit.test.ts:5` (test — cursor logic)
- `react/hooks/useChat.hooks.ts:337-359` (implementation — `handleHistory`)

**Observed behavior:** In the ChatBar textarea:
1. `ArrowUp` key → `event.preventDefault()` + `handleHistory("up")` — navigates to previous input
2. `ArrowDown` key → `event.preventDefault()` + `handleHistory("down")` — navigates to next input
3. Both prevent default textarea cursor movement
4. Current draft is preserved (REQ-CHAT-022) and restored when navigating past the newest entry

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: keyboard events in textarea (ArrowUp/ArrowDown)
- Output: textarea value changes to show historical input
- Side effects: state mutations (historyCursor, input value)
- Invariants: up/down navigate through `inputHistory` array; draft is saved on first up; draft is restored on down-past-newest; empty history → no-op
- Error cases: empty history → returns early (no crash)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-085: Enter key submits (without modifier)

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (primary interaction pattern)
**Risk tags:** —

**Source evidence:**
- `react/views/components/ChatBar.tsx:74-79` (implementation — `onKeyDown` handler)

**Observed behavior:** In the ChatBar textarea:
1. `Enter` (without Shift, Meta, or Ctrl) → `event.preventDefault()` + `submit()`
2. `Shift+Enter` / `Meta+Enter` / `Ctrl+Enter` → default behavior (newline insertion)

This follows the standard chat input convention: Enter sends, Shift+Enter creates a newline.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `keydown` event with `key === "Enter"`
- Output: if no modifier → submit; if modifier → default behavior (newline)
- Side effects: `submit()` called (triggers full submission pipeline per REQ-CHAT-002)
- Invariants: `Shift`, `Meta`, and `Ctrl` all suppress the submit behavior; the check is `!event.shiftKey && !event.metaKey && !event.ctrlKey`
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-086: Markdown rendering of assistant messages

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (content display contract)
**Risk tags:** —

**Source evidence:**
- `react/views/components/ChatSidebar.tsx:128-134` (implementation — `ReactMarkdown` with `remarkGfm`)
- `react/views/components/ChatSidebar.tsx:135-136` (implementation — user messages rendered as plain text)

**Observed behavior:** Message rendering differs by role:
1. **Assistant messages** → rendered via `<ReactMarkdown remarkPlugins={[remarkGfm]}>` — supports tables, autolinks, strikethrough, task lists
2. **User messages** → rendered as plain text (`{message.content}`) — no markdown processing

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `message.content` string, `message.role`
- Output: assistant → markdown-rendered HTML; user → plain text
- Side effects: not_applicable (rendering)
- Invariants: only assistant messages get markdown treatment; user messages are literal text; GFM plugin enables tables and autolinks; raw HTML is NOT rendered (react-markdown default — safe against XSS)
- Error cases: malformed markdown → rendered as plain text by react-markdown (graceful degradation)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- Vue: use `markdown-it` or `marked` with GFM plugin, render via `v-html` with sanitization.
- Svelte: use `marked` + `{@html}` with DOMPurify.
- Angular: use `ngx-markdown` or pipe with `marked`.
- All implementations must NOT render raw HTML from AI output (XSS protection).

---

### REQ-CHAT-087: Auto-scroll to latest message

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (UX convenience)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChatSidebar.web.ts:81-95` (implementation — auto-scroll effect)
- `react/views/components/ChatSidebar.tsx:109` (implementation — `ref={scrollRef}`)

**Observed behavior:** When `messages.length` changes or `isSending` toggles, the scroll container is scrolled to the bottom via `requestAnimationFrame(() => container.scrollTop = container.scrollHeight)`. This keeps the most recent message visible without user intervention.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: change in `messages.length` or `isSending` value
- Output: scroll container scrolled to bottom
- Side effects: DOM scroll position mutation
- Invariants: uses `requestAnimationFrame` (not synchronous) to ensure DOM has rendered before scrolling; cleanup cancels pending RAF on unmount or re-trigger
- Error cases: if `scrollRef.current` is null → no scroll (safe)
- Transaction boundary: not_applicable
- Concurrency: RAF is cancelled on cleanup — no stale scroll operations
- Auth requirement: not_applicable

---

### REQ-CHAT-088: Copy-to-clipboard with transient visual feedback

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** low
**Criticality reason:** low_usage (convenience feature)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChatSidebar.web.ts:117-124` (implementation — `handleCopy`)
- `react/hooks/useChatSidebar.web.ts:98-108` (implementation — 2-second timeout reset)
- `react/views/components/ChatSidebar.tsx:139-145` (implementation — button with "Copy"/"check" toggle)
- `integration/err-004.clipboard-failure.integration.test.ts:6` (test)

**Observed behavior:**
1. User clicks "Copy" button on any message
2. `handleCopy(id, content)` calls `navigator.clipboard.writeText(content)`
3. On success: `copiedId` is set to the message ID → button shows "check" for 2 seconds
4. After 2 seconds: `copiedId` resets to null → button shows "Copy" again
5. On failure: `copiedId` stays null → no false-positive feedback

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: click on "Copy" button for a specific message
- Output: content copied to clipboard; 2-second visual indicator
- Side effects: clipboard write; transient UI state (`copiedId`)
- Invariants: only one message shows "copied" at a time (last copied wins); 2000ms timeout is fixed; failure is silent
- Error cases: clipboard API denied → no indicator shown (REQ-CHAT-018)
- Transaction boundary: not_applicable
- Concurrency: new copy replaces previous (last-write-wins for `copiedId`)
- Auth requirement: not_applicable

---

### REQ-CHAT-089: Drag-and-drop file attachment

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (file input mechanism)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChatSidebar.web.ts:132-153` (implementation — `handleDrop`)
- `react/hooks/useChatSidebar.web.ts:161-171` (implementation — drag state management)
- `react/views/components/ChatSidebar.tsx:66-70` (implementation — event bindings on `<aside>`)
- `react/views/components/ChatSidebar.tsx:71-74` (implementation — drag overlay)
- `integration/req-008.invalid-attachments.integration.test.ts:13` (test)

**Observed behavior:**
1. User drags files over the sidebar → `isDragging` becomes true → overlay appears ("Drop files to attach")
2. User drops files → `handleDrop` reads each file via `FileReader.readAsDataURL`
3. Successfully read files are passed to `addAttachments(files)`
4. Failed file reads are silently discarded (Promise.allSettled filtering)
5. User drags away without dropping → `handleDragLeave` resets `isDragging` to false
6. Attached files appear in a pill list between the messages area and the input bar
7. Each pill has an "x" button to remove the attachment by index

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: drag-over, drag-leave, drop browser events
- Output: files read as base64 dataUrl and staged in local UI state (`attachments`)
- Side effects: `isDragging` state toggle; `addAttachments` call after successful reads
- Invariants: drop zone is the entire sidebar `<aside>` element; partial failures don't block successful files; zero successful reads → no call to `addAttachments`
- Error cases: all files fail to read → no attachments added, no error shown
- Transaction boundary: not_applicable
- Concurrency: `Promise.allSettled` processes all files concurrently
- Auth requirement: not_applicable

---

### REQ-CHAT-090: Empty state placeholder

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (UX text)
**Risk tags:** —

**Source evidence:**
- `react/views/components/ChatSidebar.tsx:111-114` (implementation)

**Observed behavior:** When `messages.length === 0`, the sidebar displays: "Ask a question to get started." in muted text. This disappears as soon as the first message is added.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `messages.length === 0`
- Output: placeholder text displayed
- Side effects: not_applicable
- Invariants: placeholder is only shown when message array is empty; disappears on first message
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-091: UIFeedback uses ARIA roles for accessibility

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (accessibility compliance)
**Risk tags:** —

**Source evidence:**
- `react/views/components/UIFeedback.tsx:38` (implementation — `role={feedback.kind === "error" ? "alert" : "status"}`)
- `react/views/components/UIFeedback.tsx:58` (implementation — `aria-label="Dismiss feedback"`)
- `react/views/components/ChatBar.tsx:49` (implementation — `aria-disabled="true"`)
- `react/views/components/ChatBar.tsx:93` (implementation — `aria-label="Chat input"`)
- `react/views/components/ChatSidebar.tsx:83` (implementation — `aria-label="Select AI model"`)
- `react/views/components/ChatSidebar.tsx:173` (implementation — `aria-label="Remove attachment"`)

**Observed behavior:** Accessibility attributes across chat components:
1. `UIFeedback`: `role="alert"` for errors (announces immediately), `role="status"` for warning/info (polite announcement)
2. ChatBar textarea: `aria-label="Chat input"`
3. ChatBar "+" button: `aria-disabled="true"` (semantically disabled, not interactively)
4. Model selector: `aria-label="Select AI model"`
5. Dismiss button: `aria-label="Dismiss feedback"`
6. Remove attachment: `aria-label="Remove attachment"`

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: rendered chat components
- Output: ARIA attributes present for screen reader compatibility
- Side effects: not_applicable
- Invariants: errors use `role="alert"` (assertive); non-errors use `role="status"` (polite); interactive elements without visible text have `aria-label`
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- Any framework implementation MUST preserve these ARIA attributes. They are behavioral requirements (screen reader interaction), not React-specific patterns.
- `role="alert"` triggers immediate screen reader announcement — this is a user-facing contract.

---

### REQ-CHAT-092: Chat state persists during SPA navigation for global store consumers

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (navigation UX for agentic-research page)
**Risk tags:** —

**Source evidence:**
- `react/state/zustand/aiChatStore.ts:88` (implementation — module-level `create()`)
- `ui/screens/AgenticResearchPage/chat/orchestrators/agenticResearchChatOrchestrator.ts:19` (consumer — uses global store)
- REQ-CHAT-045 (artifact-2 — singleton topology)

**Observed behavior:** The global `useAiChatStore` is a module-level singleton. It persists in memory across SPA navigations (Next.js client-side routing). This means:
1. If a user on `/agentic-research` sends messages, navigates to `/`, then navigates back to `/agentic-research` — their messages are still visible
2. Model options loaded on one page are available on another (if both use the global store)
3. `isSending` flag persists across navigation (including the stale-state edge case from REQ-CHAT-044)

The LandingPage uses an isolated store, so its messages do NOT persist across navigation (they reset when the component unmounts and remounts, since it's a separate Zustand instance at module level — but since it's also a singleton, same-page re-navigations preserve state).

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: SPA navigation (Next.js router) between pages sharing the global store
- Output: messages, model options, and sending state persist in memory
- Side effects: not_applicable
- Invariants: only global store consumers see persistence; isolated store consumers (LandingPage) have their own lifecycle; full page refresh resets all stores to initial state
- Error cases: stale `isSending` after navigate-away (REQ-CHAT-044)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-093: No durable links or bookmarkable chat state

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (no URL-driven chat state)
**Risk tags:** —

**Source evidence:**
- Full scan of `features/ai-chat/` — zero references to `useSearchParams`, `router.push`, or URL state management
- `react/hooks/useChat.hooks.ts:25` — `globalThis.location?.pathname` used ONLY for debug logging, never for state

**Observed behavior:** The chat feature has NO URL-driven state:
1. No query parameters control chat behavior (no `?model=X`, no `?message=Y`)
2. No hash fragments link to specific messages
3. No deep links to specific chat sessions
4. No shareable URLs that recreate chat state
5. Navigating to `/chat` always starts with an empty chat (or preserved module-level state within the same session)

The LandingPage has a separate `UIFeedback` component that reads `?demo-toast=...` query params, but this is a page-level demo feature unrelated to ai-chat.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: any URL navigation to a page containing chat
- Output: chat state is determined by in-memory store, not by URL
- Side effects: not_applicable
- Invariants: URL changes do not affect chat state; chat state changes do not update the URL
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

## Human Workflows

---

### REQ-CHAT-094: Primary user workflow — type, send, see response

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** high
**Criticality reason:** high_traffic (core user interaction path)
**Risk tags:** —

**Source evidence:**
- `react/views/components/ChatBar.tsx:68-105` (implementation — textarea + send button)
- `react/views/components/ChatSidebar.tsx:116-155` (implementation — message display + spinner)
- `integration/req-001.submit-order.integration.test.ts:8` (test)
- `integration/req-002.sending-reset.integration.test.ts:62` (test)

**Observed behavior:** The complete happy-path user workflow:
1. User types in textarea → `value` state updates reactively
2. User presses Enter (or clicks "Send") → `submit()` fires
3. Input is trimmed and validated (empty → no-op)
4. User message appears in the message list immediately
5. Textarea clears; spinner appears ("Working")
6. API call executes to backend
7. On success: assistant message appears; spinner disappears
8. On failure: error feedback appears inline with "Try again" button

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: user types text and triggers send
- Output: immediate optimistic UI (user message shows), then server response or error
- Side effects: API call, state mutations per REQ-CHAT-002 ordering
- Invariants: user message appears BEFORE API call (optimistic); spinner shows DURING API call; response or error shows AFTER API call; textarea clears immediately on send
- Error cases: see failure matrix (REQ-CHAT-051)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-095: Retry failed message workflow

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (error recovery)
**Risk tags:** —

**Source evidence:**
- `react/views/components/ChatSidebar.tsx:99-107` (implementation — UIFeedback with onAction)
- `react/hooks/useChat.hooks.ts:326-329` (implementation — `retryLastSubmission`)
- `react/views/components/UIFeedback.tsx:44-53` (implementation — action button rendering)

**Observed behavior:** When a chat request fails:
1. Error feedback appears inline (above messages) with "Try again" button
2. User clicks "Try again" → `retryLastSubmission()` fires
3. The stored last submission payload is re-sent (without re-adding the user message)
4. Same sending/response lifecycle applies (spinner, success/error outcome)
5. User can also dismiss the error via "Dismiss" button → `setScreenFeedback(null)`
6. Successful retry replaces the error feedback with the assistant response

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: user clicks "Try again" on error feedback
- Output: last submission re-executed; same lifecycle as original submit
- Side effects: API call with stored payload
- Invariants: retry uses stored payload (model, history, dataset from original submission time); does NOT re-add user message; "Try again" only appears when `feedback.retryable === true`
- Error cases: no prior submission → no-op (guard in `retryLastSubmission`)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-096: Model selection affects next request only

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (model routing correctness)
**Risk tags:** —

**Source evidence:**
- `react/views/components/ChatSidebar.tsx:78-96` (implementation — select onChange)
- `react/hooks/useChat.hooks.ts:313` (implementation — `model: state.selectedModelId` captured at submit time)

**Observed behavior:** Changing the model selector immediately updates `selectedModelId` in state. This selection is captured at submit time and sent as the `model` field in the next API request. Changing the model does NOT:
1. Retroactively affect previous messages
2. Trigger a re-send of any previous request
3. Clear the conversation
4. Fetch new models (only bootstrap and manual refetch do that)

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: user changes model selector dropdown
- Output: `selectedModelId` updates; next submit uses the new model
- Side effects: `setSelectedModelId` state mutation
- Invariants: model change is prospective (affects future requests only); captured at submit time (not at selection time)
- Error cases: selecting empty option → `null` → backend receives `model: null`
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

## Frontend/Client Behavior — Multi-Framework Extraction

---

### REQ-CHAT-097: React-specific patterns requiring framework equivalents

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** high
**Criticality reason:** fragile_consumer (migration correctness — each pattern needs a framework-appropriate equivalent)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:224-234` (useRef for isMountedRef, lastSubmissionRef, draftValueRef)
- `react/hooks/useChat.hooks.ts:487-505` (useEffect for model bootstrap on mount)
- `react/state/zustand/aiChatStore.ts:88` (Zustand create)
- `react/state/adapters/aiChatState.adapter.ts:1` (useShallow)
- `react/compositions/useChatSurface.orchestrator.ts:69-106` (useMemo for memoized deps)
- `react/views/components/ChatSidebar.tsx:26-29` (component with prop injection)

**Observed behavior:** The following React-specific patterns need framework equivalents:

| React Pattern | Purpose | Vue Equivalent | Svelte Equivalent | Angular Equivalent |
|--------------|---------|---------------|------------------|-------------------|
| `useRef` for `isMountedRef` | Lifecycle guard for post-async operations | `onUnmounted` flag | `onDestroy` flag | `ngOnDestroy` flag |
| `useRef` for `lastSubmissionRef` | Mutable storage across renders | `ref()` (non-reactive) or closure variable | Module-level `let` or `#private` | Service instance property |
| `useRef` for `draftValueRef` | Cross-render mutable value | Closure `let` in composable | Module `let` in store | Service property |
| `useEffect(…, [])` for mount bootstrap | One-time initialization | `onMounted()` | `onMount()` | `ngOnInit()` |
| `useEffect` with deps for auto-scroll | React to state changes | `watch()` | `$:` reactive statement or `$effect` | `ngOnChanges` or RxJS subscription |
| `useMemo` for memoized deps | Prevent unnecessary recomputation | `computed()` | derived `$:` | `pipe` with `shareReplay` or `Signal` |
| Zustand `create()` | Global state singleton | Pinia `defineStore()` | `writable()` store | NgRx Store or Signal Store |
| `useShallow` selector | Shallow equality for re-render optimization | Pinia getter (automatic) | Store subscription (automatic) | Selector with `distinctUntilChanged` |
| Hook-as-prop injection | Composition at component level | `provide/inject` composable | Context API or prop factory | DI token with `useFactory` |
| `"use client"` directive | Mark client-side component | `<ClientOnly>` wrapper | `{#if browser}` or dynamic import | `isPlatformBrowser` guard |

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: React-specific hooks and patterns
- Output: equivalent lifecycle management, state management, and optimization in target framework
- Side effects: not_applicable (architectural mapping)
- Invariants: the BEHAVIORAL contracts (ordering, mount guard, memoization boundary) must be preserved regardless of framework; the specific API (useRef vs ref() vs let) is implementation detail
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-098: ChatIntegration is the sole public interface between feature and view

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** high
**Criticality reason:** fragile_consumer (any UI component only touches this interface)
**Risk tags:** —

**Source evidence:**
- `__types__/chat.types.ts:167` (type — `ChatIntegration = ChatUiState & ChatState & ChatActions`)
- `react/views/components/ChatSidebar.tsx:30-43` (consumer — destructures ChatIntegration)
- `react/views/components/ChatBar.tsx:20-29` (consumer — destructures ChatIntegration)

**Observed behavior:** `ChatIntegration` is the ONLY interface that view components consume. It is a flat union of:
1. `ChatUiState` — local input state (`value`, `showTooltip`, `attachments`, setters)
2. `ChatState` — store state (`messages`, `isSending`, `modelOptions`, etc.)
3. `ChatActions` — orchestrated actions (`submit`, `retryLastSubmission`, `handleHistory`, etc.)

No view component directly imports a store, hook, or logic function. They all go through the orchestrator which returns `ChatIntegration`.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: orchestrator function called by component
- Output: flat `ChatIntegration` object with all state + actions
- Side effects: not_applicable (interface contract)
- Invariants: view components are fully decoupled from state management implementation; any state backend satisfying `ChatIntegration` is valid; the interface is framework-agnostic in shape (only the delivery mechanism — hook vs composable vs service — is framework-specific)
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- In Vue: return a reactive object from a composable function. Components destructure from the return value.
- In Svelte: return a store object from a factory function. Components subscribe via `$store` syntax.
- In Angular: inject a service that exposes observables/signals matching the ChatIntegration shape.

---

### REQ-CHAT-099: Customization points form a port-based composition API

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** high
**Criticality reason:** fragile_consumer (enables the entire multi-page architecture)
**Risk tags:** —

**Source evidence:**
- `react/compositions/useChatSurface.orchestrator.ts:47-55` (type — `UseChatSurfaceOptions`)
- `__types__/chat.types.ts:108` (type — `UseChatStatePort`)
- `__types__/chat.types.ts:114` (type — `UseChatChartActionsPort`)
- `__types__/chat.types.ts:119-131` (type — `ChatApiDeps`)
- `__types__/chat.types.ts:173-187` (type — `ChatLogicDeps`)

**Observed behavior:** The composition root exposes exactly 6 customization points (ports):

| Port | Type | Default | Purpose |
|------|------|---------|---------|
| `useStatePort` | `UseChatStatePort` | `useAiChatStateAdapter` (global store) | Which state container to use |
| `useChartActionsPort` | `UseChatChartActionsPort` | Empty no-op port | Where chart specs are dispatched |
| `activeDatasetId` | `string \| null` | (undefined — defers to hook) | Static dataset override |
| `useActiveDatasetId` | `() => string \| null` | `() => null` | Dynamic dataset hook |
| `mode` | `"research" \| "direct"` | `"research"` | API endpoint selection |
| `apiAdapter` | `ChatApiDeps` | (created from mode) | Pre-built API override |

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `UseChatSurfaceOptions` with optional port overrides
- Output: fully-wired `ChatIntegration`
- Side effects: not_applicable (composition)
- Invariants: all ports are optional (sensible defaults exist); ports are typed interfaces (not concrete implementations); the composition root never imports a specific store or chart implementation directly — only via the injected port
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- This is the "dependency injection" boundary for the entire feature. In any framework, the equivalent composition root must accept these 6 ports (or their equivalents) and wire them into the integration.
- Port interfaces are framework-agnostic types — only the delivery mechanism (hook vs provide/inject vs DI token) changes.

---

### REQ-CHAT-100: Page layout contract — sidebar positioned as sticky right panel

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (layout consistency)
**Risk tags:** —

**Source evidence:**
- `ui/screens/LandingPage/views/LandingPageScreen.tsx:64-67` (implementation — sidebar container)
- `ui/screens/AgenticResearchPage/views/AgenticResearchPageScreen.tsx:79` (implementation — sidebar container)
- `react/views/components/ChatSidebar.tsx:65` (implementation — `h-[calc(100vh-64px)]`)

**Observed behavior:** The chat sidebar renders as a right-panel in a flex layout:
1. Parent: `flex flex-row` (horizontal layout)
2. Main content: `flex-1` (takes remaining space)
3. Sidebar container: `sticky top-16 h-[calc(100vh-64px)] w-[360px] shrink-0 overflow-hidden`
4. ChatSidebar itself: `h-[calc(100vh-64px)] w-full flex-col border-l`

Layout properties:
- Fixed width: 360px
- Full viewport height minus navbar (64px): `calc(100vh - 64px)`
- Sticky positioning (stays visible during main content scroll)
- Overflow hidden on container (sidebar manages its own scroll internally)
- Border on left side separating from main content

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: page renders with `showSidebar=true` (or `showChatSidebar=true`)
- Output: sidebar anchored to the right, full-height minus navbar, 360px wide
- Side effects: not_applicable (layout)
- Invariants: sidebar does not scroll with main content (sticky); sidebar width is fixed; navbar height is 64px (assumed from `top-16` + `calc(100vh-64px)`); sidebar visibility is controlled by page-level boolean prop
- Error cases: narrow viewport → main content compressed (no responsive breakpoint visible)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

## Open Questions Discovered This Pass

1. **Responsive behavior for small viewports:** The sidebar is always 360px wide with no responsive breakpoint or collapse behavior. On viewports < ~720px, the main content area would be extremely narrow. Is there a mobile/responsive design, or is this desktop-only? Priority: nice-to-have.

2. **ag-ui-chat feature relationship:** The `ag-ui-chat` feature imports `UseChatChartActionsPort` type from ai-chat. Is this the extent of their coupling, or should the ag-ui-chat feature be documented as a consumer that may expand its dependency? Priority: nice-to-have.

3. **LandingPage UIFeedback (toast) vs ai-chat UIFeedback (inline):** The LandingPage has a separate `UIFeedback` component that shows demo toasts via query params (`?demo-toast=error`). This is NOT the ai-chat `UIFeedback` — it is a page-level toast system using `react-hot-toast`. Are these ever confused by consumers? Priority: nice-to-have.

---

## Amendments to Prior Artifacts

### Amendment to REQ-CHAT-015 (Dataset context injection is screen-owned)

REQ-CHAT-015 documented the `mapChatStateWithDataset` function. This pass provides the full consumer evidence: AgenticResearchPage injects a dynamic `useActiveDatasetId` hook that reads from the agentic-research state, while LandingPage omits the dataset entirely (relying on the default `null`). The composition root supports both static and dynamic dataset injection patterns.

### Amendment to REQ-CHAT-017 (State adapter port pattern)

REQ-CHAT-017 documented the `ChatStatePort` abstraction. This pass confirms two concrete implementations exist:
1. `useAiChatStateAdapter` — global singleton (used by AgenticResearchPage)
2. `useLandingChatStateAdapter` — isolated instance (used by LandingPage)

Both are structurally identical adapters over different Zustand stores, confirming the port pattern works as designed.

### Amendment to REQ-CHAT-045 (Zustand store is a global singleton)

REQ-CHAT-045 documented the singleton topology. This pass clarifies the consumer mapping: only AgenticResearchPage uses the global store; LandingPage (which serves both `/` and `/chat`) uses its own isolated store. The `/ag-ui` page does not use ai-chat at all.

---

## Risk Tags/Markers Raised This Pass

| Marker | REQ ID | Severity | Description |
|--------|--------|----------|-------------|
| `[ENVIRONMENTAL CONTRACT]` | REQ-CHAT-079 | Advisory | Chat sidebar MUST NOT be server-side rendered (requires browser APIs) |

---

## Entrypoints Discovered or Removed

### Discovered
- `useChatSurfaceOrchestrator` options interface (`UseChatSurfaceOptions`) — the composition API surface for page-level customization
- `LandingChatSidebar` wrapper component — thin adapter that passes `useLandingChatOrchestrator` to `ChatSidebar`

### Removed
None.

---

## Summary

- **Requirements extracted this pass:** 27 (REQ-CHAT-074 through REQ-CHAT-100)
- **Cumulative total:** 100 (30 from Pass 1 + 20 from Pass 2 + 23 from Pass 3 + 27 from Pass 4)
- **Confidence breakdown (this pass):** observed: 25, tested: 2
- **Blocking markers:** 0
- **Important markers:** 0
- **Advisory markers:** 1 (`[ENVIRONMENTAL CONTRACT]`)

### Key Findings for Multi-Framework Migration

1. **Single composition root (`useChatSurfaceOrchestrator`)** — ALL pages wire through this one function with injectable ports. Any framework implementation needs one equivalent factory/composable/service that accepts the same 6 ports.

2. **`ChatIntegration` is the UI boundary** — View components only ever receive a flat `ChatIntegration` object. The orchestrator hook is the sole bridge between state/logic and rendering. This makes view components trivially portable once the integration layer exists.

3. **Three consumers, three configurations:**
   - LandingPage: isolated store + CopilotKit chart port + direct mode + no dataset
   - AgenticResearchPage: global store + agentic-research chart port + research mode + dynamic dataset
   - ag-ui-chat: type-only dependency (no runtime consumption of ai-chat)

4. **`.web.ts` platform suffix** — explicit architectural signal for platform-specific browser behavior. All browser APIs are injectable via runtime deps, enabling both testing and future platform variants.

5. **No durable links, no persistence, no URL state** — chat is entirely ephemeral and in-memory. No bookmarks, no deep links, no session restoration.

6. **Client-only rendering** — both page screens use `dynamic(import, { ssr: false })` to prevent server-side rendering of chat components. This is a hard constraint: chat hooks require browser APIs.

7. **Behavior contracts ANY implementation must satisfy:**
   - Messages render in chronological (append-only) order
   - Sending indicator shown during request; send button disabled
   - Enter sends, Shift+Enter creates newline
   - ArrowUp/Down navigate input history with draft preservation
   - Model selector reflects options; selection affects next request only
   - Error feedback is inline with retry action
   - Auto-scroll to latest message
   - Clipboard copy with 2-second transient feedback
   - File drag-drop with graceful failure handling
   - Assistant messages rendered as markdown; user messages as plain text
   - ARIA roles on error/status feedback elements
