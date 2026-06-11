# Artifact 1: Core Logic Extraction — ai-chat

Feature path: `web/nextjs/src/features/ai-chat/`
Extraction date: 2026-06-09
Pass: 1 (Core Logic)
Methodology: `AI-Dev-Shop-speckit/skills/reverse-spec/SKILL.md` v2.0.0

---

## Phase 0: Inventory Reconciliation — Entrypoint Map

All entrypoints from the inventory were verified by reading source files.

### Types (confirmed)
| Symbol | File | Status |
|--------|------|--------|
| `ChatIntegration` | `__types__/chat.types.ts:167` | confirmed |
| `ChatStatePort` | `__types__/chat.types.ts:103` | confirmed |
| `ChatMessage` | `__types__/chat.types.ts:27` | confirmed |
| `ChatModelOption` | `__types__/chat.types.ts:35` | confirmed |
| `ScreenFeedback` | `__types__/uiFeedback.types.ts:4` | confirmed |
| `ChatOrchestrator` (re-export) | `react/orchestrators/chatOrchestrator.ts:29` | confirmed |

### Hooks / Compositions (confirmed)
| Symbol | File | Status |
|--------|------|--------|
| `useChatSurfaceOrchestrator` | `react/compositions/useChatSurface.orchestrator.ts:56` | confirmed |
| `useChatOrchestrator` | `react/orchestrators/chatOrchestrator.ts:17` | confirmed |
| `useAiChatStateAdapter` | `react/state/adapters/aiChatState.adapter.ts:8` | confirmed |

### Logic (confirmed)
| Symbol | File | Status |
|--------|------|--------|
| `createInitialChatStoreCoreState` | `logic/chatStore.logic.ts:16` | confirmed |
| `appendMessage` | `logic/chatStore.logic.ts:37` | confirmed |
| `appendInputHistory` | `logic/chatStore.logic.ts:49` | confirmed |
| `resolveHistoryCursor` | `logic/chatStore.logic.ts:64` | confirmed |

### API Endpoints (confirmed)
| Endpoint | Method | Status |
|----------|--------|--------|
| `{baseUrl}/chat` | POST | confirmed |
| `{baseUrl}/chat-research` | POST | confirmed |
| `{baseUrl}/llm/gemini-models` | GET | confirmed |

### Discovered During Pass 1
| Symbol | File | Note |
|--------|------|------|
| `resolveSubmitFeedback` | `react/hooks/useChat.hooks.ts:42` | Exported helper, error-to-feedback mapping; behavioral contract |
| `setFallbackModels` | `react/hooks/useChat.hooks.ts:115` | Exported helper consumed by hook and tests |
| `setFetchedModels` | `react/hooks/useChat.hooks.ts:135` | Exported helper consumed by hook and tests |

No unreachable entrypoints were flagged.

---

## Phase 1: Test-First Extraction — Test Classification

### Integration Tests (behavior-test)
| Test | Spec Ref | Classification | Rationale |
|------|----------|----------------|-----------|
| `req-001.submit-order` | REQ-001/AC-001 | **behavior-test** | Asserts ordering of externally-observable state mutations during submit; uses real logic deps |
| `req-001.empty-input-short-circuit` | REQ-001/DR-001 | **behavior-test** | Asserts no side effects for invalid input through public `submit()` boundary |
| `req-002.sending-reset` | REQ-002 | **behavior-test** | Asserts `isSending` lifecycle across success/null/error outcomes |
| `dr-004.err-001.invalid-payload-normalization` | DR-004/ERR-001 | **behavior-test** | Exercises `sendChatMessageDirect` with invalid backend shapes, asserts null return |
| `err-002.fetch-models-fallback` | ERR-002 | **behavior-test** | Asserts deterministic fallback when fetchModels throws |
| `err-004.clipboard-failure` | ERR-004 | **behavior-test** | Asserts resilient clipboard behavior through public hook |
| `err-005.timeout-retryable-contract` | ERR-005 | **behavior-test** | Asserts timeout produces deterministic retryable error shape |
| `req-005.contract-location` | REQ-005 | **behavior-test** | Architectural invariant: type contracts live under `__types__/` |
| `req-006.page-agnostic-dataset.wiring` | REQ-006 | **behavior-test** | Asserts orchestrator injects null datasetId by default |
| `req-007.abort-unmount` | REQ-007 | **behavior-test** | Asserts no post-unmount state mutation via isMounted guard |
| `req-008.invalid-attachments` | REQ-008/ERR-007 | **behavior-test** | Asserts no throw and no attachment on file-read failure |
| `req-009.models-timeout` | REQ-009 | **behavior-test** | Asserts timeout error shape from fetchChatModels |
| `ab-001.no-cross-feature-domain-imports` | AB-001 | **behavior-test** | Architectural invariant: no cross-feature state/orchestrator imports |
| `ab-002.screen-context-injection` | AB-002 | **behavior-test** | Architectural invariant: no hardcoded route literals |
| `ab-003.logic-framework-agnostic` | AB-003 | **behavior-test** | Architectural invariant: logic layer has no react/zustand/next/DOM imports |

### Spec-Tagged Unit Tests (behavior-test)
| Test | Spec Ref | Classification | Rationale |
|------|----------|----------------|-----------|
| `dr-002.history-window.boundary` | DR-002 | **behavior-test** | Pure function, asserts window size + shape invariant |
| `dr-003.history-cursor.totality` | DR-003/INV-003 | **behavior-test** | Pure function totality: never throws for any cursor/direction |
| `dr-005.fallback-model-order.stability` | DR-005 | **behavior-test** | Pure function determinism: order stability |
| `req-003.chart-fanout` | REQ-003 | **behavior-test** | Pure function: deterministic chart dispatch |
| `req-004.model-selection` | REQ-004 | **behavior-test** | Pure function: selection precedence logic |

### Test Fixture
| File | Purpose |
|------|---------|
| `fixtures/chatLogicDeps.fixture.ts` | Provides REAL logic implementations as `ChatLogicDeps`; enables integration-style tests |

---

## Phase 2 & 2b: Code Extraction with Inline Docs

---

### REQ-CHAT-001: Empty input short-circuits submission

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (prevents accidental empty submissions)
**Risk tags:** —

**Source evidence:**
- `logic/chatSubmission.logic.ts:22` (implementation)
- `integration/req-001.empty-input-short-circuit.integration.test.ts:11` (test)

**Observed behavior:** When `value.trim()` is empty, `normalizeSubmissionValue` returns `null`. The `submit()` action returns immediately without calling API, adding messages, or modifying any state.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `{ value: string }` — raw user input
- Output: `null` (empty) or trimmed `string`
- Side effects: verified_none — test asserts no API call, no state mutation, no UI reset (cite: `req-001.empty-input-short-circuit.integration.test.ts:62-68`)
- Invariants: whitespace-only input is treated as empty
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable (client-side only)

---

### REQ-CHAT-002: Submit ordering invariant

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (ensures UI state consistency)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:291-324` (implementation)
- `integration/req-001.submit-order.integration.test.ts:8` (test)

**Observed behavior:** Submit follows strict ordering: (1) addInputToHistory, (2) addMessage (user), (3) setSending(true), (4) api.sendMessage.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: non-empty trimmed user value, current state (messages, selectedModelId, activeDatasetId, attachments)
- Output: sequence of state mutations followed by API call
- Side effects: UI value reset, history cursor reset, attachments cleared — all occur before API call
- Invariants: `addInputToHistory` < `addMessage` < `setSending(true)` < `sendMessage` (temporal ordering)
- Error cases: see REQ-CHAT-003 (sending reset)
- Transaction boundary: not_applicable
- Concurrency: not_applicable (single-threaded JS)
- Auth requirement: not_applicable

---

### REQ-CHAT-003: Sending flag reset on all outcomes

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (prevents stuck sending state)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:246-289` (implementation — `finally` block)
- `integration/req-002.sending-reset.integration.test.ts:62-123` (test)

**Observed behavior:** `setSending(false)` is called in a `finally` block regardless of success, null response, or thrown error. Attachments are cleared before submission (in `submit()` not in `runSubmission()`).
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: any outcome from `api.sendMessage` (resolved value, null, or thrown error)
- Output: `setSending(false)` always called if component is still mounted
- Side effects: `setScreenFeedback(error)` on failure
- Invariants: `isSending` is never left as `true` after the request lifecycle completes (mount-guarded)
- Error cases: on error → `setScreenFeedback(resolveSubmitFeedback(error))`; on null response → `setScreenFeedback(INVALID_RESPONSE_FEEDBACK)`
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-004: Model selection precedence

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (model UX correctness)
**Risk tags:** —

**Source evidence:**
- `logic/modelSelection.logic.ts:41-52` (implementation — `resolveFetchedModelSelection`)
- `unit/req-004.model-selection.unit.test.ts:14-79` (test)

**Observed behavior:** Selection precedence when models are fetched: (1) keep existing `selectedModelId` if present, (2) use `result.currentModel`, (3) use first model from `result.models`, (4) null if empty.
**Normative contract:** matches observed (spec ref: ai-chat.spec.md v1.8.0 REQ-004)
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `{ selectedModelId: string | null, result: { currentModel: string | null, models: ChatModelOption[] } }`
- Output: `{ modelOptions: ChatModelOption[], selectedModelId: string | null }`
- Side effects: verified_none (pure function, cite: `logic/modelSelection.logic.ts:41-52`)
- Invariants: existing selection is never overwritten; precedence chain is deterministic
- Error cases: empty models array → selectedModelId is null
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-005: Fallback model selection on fetch failure

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (resilient UX)
**Risk tags:** —

**Source evidence:**
- `logic/modelSelection.logic.ts:24-33` (implementation — `resolveFallbackModelSelection`)
- `logic/modelSelection.logic.ts:9-14` (FALLBACK_CHAT_MODELS constant)
- `integration/err-002.fetch-models-fallback.integration.test.ts:9` (test)
- `unit/dr-005.fallback-model-order.stability.unit.test.ts:8` (test)

**Observed behavior:** When `fetchModels` throws or returns error, fallback models are applied deterministically: `[gemini-3-flash-preview, gemini-3.1-pro-preview, gemini-3-pro-preview, gemini-2.5-pro]`. Selection picks existing `selectedModelId` or first fallback.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `{ selectedModelId: string | null }`, optional `{ fallbackModels?: ChatModelOption[] }`
- Output: `{ modelOptions: ChatModelOption[], selectedModelId: string | null }`
- Side effects: verified_none (pure function, cite: `logic/modelSelection.logic.ts:24-33`)
- Invariants: order stability across calls (DR-005); existing selection preserved
- Error cases: empty fallback array → selectedModelId is null
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-006: History window bounded to 10 entries

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (API payload size control)
**Risk tags:** —

**Source evidence:**
- `logic/chatSubmission.logic.ts:35-53` (implementation)
- `unit/dr-002.history-window.boundary.unit.test.ts:14` (test)

**Observed behavior:** `buildChatHistoryWindow` returns at most `windowSize` entries (default 10). Current user message with attachments is always the last entry. Older messages are truncated from the start.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `{ messages: ChatMessage[], userContent: string, attachments: ChatAttachment[] | undefined }`, optional `{ windowSize?: number }`
- Output: `ChatHistoryMessage[]` with length <= windowSize, last entry is `{ role: "user", content, attachments }`
- Side effects: verified_none (pure function, cite: `logic/chatSubmission.logic.ts:35-53`)
- Invariants: always includes current user message; maximum size is windowSize; attachments only on current message
- Error cases: not_applicable (always produces valid array)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-007: History cursor totality (never throws)

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** low
**Criticality reason:** internal_only (input navigation UX)
**Risk tags:** —

**Source evidence:**
- `logic/chatStore.logic.ts:64-101` (implementation)
- `unit/dr-003.history-cursor.totality.unit.test.ts:5` (test)

**Observed behavior:** `resolveHistoryCursor` never throws regardless of cursor position (negative, out-of-bounds, null). Returns bounded `nextCursor` (>= 0 and < length, or null) and a string value. Empty history returns `{ nextCursor: current, value: "" }`.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `{ inputHistory: string[], historyCursor: number | null, direction: "up" | "down" }`
- Output: `{ nextCursor: number | null, value: string }`
- Side effects: verified_none (pure function, cite: `logic/chatStore.logic.ts:64-101`)
- Invariants: nextCursor is always null or within [0, inputHistory.length - 1]; function is total (never throws for any input)
- Error cases: not_applicable (total function)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-008: Chart spec fan-out from assistant payload

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (chart rendering pipeline)
**Risk tags:** —

**Source evidence:**
- `logic/chatComposition.logic.ts:33-44` (implementation — `createOnMessageReceived`)
- `unit/req-003.chart-fanout.unit.test.ts:6` (test)

**Observed behavior:** `createOnMessageReceived` returns a handler that: (1) does nothing if `chartSpec` is null, (2) if array, calls `addChartSpec` once per item in order, (3) if single object, calls `addChartSpec` once.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `ChatAssistantPayload` containing `chartSpec: ChartSpec | ChartSpec[] | null`
- Output: void; side effect dispatches to `addChartSpec`
- Side effects: 0..N calls to `addChartSpec(spec)` in deterministic order
- Invariants: order matches array index order; null produces zero calls
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-009: API response normalization

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (parsing correctness)
**Risk tags:** —

**Source evidence:**
- `logic/chatApiNormalization.logic.ts:74-102` (implementation — `normalizeChatApiResult`)
- `logic/chatApiNormalization.logic.ts:37-55` (implementation — `parseJsonPayload`)
- `integration/dr-004.err-001.invalid-payload-normalization.integration.test.ts:5` (test)

**Observed behavior:** `normalizeChatApiResult` handles three backend result shapes: (1) object with `message` string + optional `chartSpec` → extract or return null if empty, (2) plain string → attempt JSON parse then plain text, (3) array of Gemini parts → join text parts. Returns null for invalid/unusable shapes (non-string message, empty parts).
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `ChatApiResponse["result"]` — union of `string | Array<{type, text?}> | {message?, chartSpec?} | undefined`
- Output: `ChatAssistantPayload | null`
- Side effects: verified_none (pure function, cite: `logic/chatApiNormalization.logic.ts:74-102`)
- Invariants: null is the safe fallback for any unusable shape; JSON embedded in strings is opportunistically parsed
- Error cases: any parse failure → null; non-string message → null; empty parts array → null
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-010: Model fetch timeout produces retryable error

**Layer:** api
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (network resilience)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.ts:188-265` (implementation — `fetchChatModels`)
- `integration/err-005.timeout-retryable-contract.integration.test.ts:10` (test)
- `integration/req-009.models-timeout.integration.test.ts:10` (test)

**Observed behavior:** `fetchChatModels` aborts after `timeoutMs` (default 5000ms) via `AbortController`. On `AbortError` → returns `{ status: "error", error: { code: "MODEL_FETCH_TIMEOUT", retryable: true, message: "Model endpoint timed out." } }`.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `{}` (empty record), optional `{ timeoutMs?: number, runtimeDeps?: ChatApiRuntimeDeps }`
- Output: `FetchChatModelsResult | null` — union of success shape or error shape
- Side effects: HTTP GET to `{baseUrl}/llm/gemini-models`; AbortController abort on timeout
- Invariants: timeout produces deterministic retryable error shape; non-AbortError produces `MODEL_FETCH_FAILED` with retryable=true
- Error cases: timeout → `MODEL_FETCH_TIMEOUT`; non-ok HTTP → null; invalid payload → `MODEL_FETCH_FAILED`; network error → `MODEL_FETCH_FAILED`
- Transaction boundary: not_applicable
- Concurrency: not_applicable (single request, no retry)
- Auth requirement: unknown (no auth header observed in code; backend may enforce separately)

---

### REQ-CHAT-011: Chat API adapter mode selection

**Layer:** api
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (endpoint routing)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.adapter.ts:27-40` (implementation)
- `api/chatApi.ts:75-86` (sendChatMessage → `/chat-research`)
- `api/chatApi.ts:94-105` (sendChatMessageDirect → `/chat`)

**Observed behavior:** `createChatApiAdapter({ mode })` returns a `ChatApiDeps` object. Mode `"research"` routes to `/chat-research` (includes `dataset_id` in payload). Mode `"direct"` routes to `/chat` (forces `dataset_id: null`).
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `{ mode: "research" | "direct" }`, optional dependency overrides
- Output: `ChatApiDeps` (sendMessage + fetchModels)
- Side effects: verified_none (factory function, cite: `api/chatApi.adapter.ts:27-40`)
- Invariants: `direct` mode always sets `dataset_id: null` in payload
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-012: Unmount guard prevents post-request state mutation

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (prevents React memory leak / zombie state)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:225-234` (implementation — `isMountedRef`)
- `react/hooks/useChat.hooks.ts:262` (guard: `if (!isMountedRef.current) return`)
- `integration/req-007.abort-unmount.integration.test.ts:16` (test)

**Observed behavior:** After component unmount, `isMountedRef.current` becomes false. All post-API state mutations (`setSending(false)`, `addMessage`, `setScreenFeedback`) are skipped if the component unmounted during the in-flight request.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: component lifecycle event (unmount during in-flight request)
- Output: no state mutations occur after unmount
- Side effects: verified_none post-unmount (cite: test asserts `setSending.mock.calls === [[true]]` only)
- Invariants: isMounted ref tracks lifecycle; all post-async operations check it
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-013: Error-to-feedback mapping

**Layer:** failure
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (user-facing error display)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:42-103` (implementation — `resolveSubmitFeedback`)
- `integration/req-002.sending-reset.integration.test.ts:117` (test — verifies `CHAT_REQUEST_FAILED` code)

**Observed behavior:** `resolveSubmitFeedback` maps unknown errors to stable `ScreenFeedback`:
- `CHAT_REQUEST_HTTP_ERROR` with status >= 500 → `CHAT_SERVICE_UNAVAILABLE` (retryable)
- `CHAT_REQUEST_HTTP_ERROR` with status < 500 → `CHAT_REQUEST_REJECTED` (retryable)
- `CHAT_RESPONSE_PARSE_ERROR` → `CHAT_RESPONSE_INVALID` (retryable)
- `AbortError` → `CHAT_REQUEST_ABORTED` (info, not retryable)
- Offline (`navigator.onLine === false`) → `CHAT_OFFLINE` (retryable)
- Unknown → `CHAT_REQUEST_FAILED` (retryable)
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `unknown` (caught error)
- Output: `ScreenFeedback` with `kind`, `code`, `message`, `retryable?`, `actionLabel?`
- Side effects: verified_none (pure function, cite: `react/hooks/useChat.hooks.ts:42-103`)
- Invariants: every error path produces a valid `ScreenFeedback`; null response (not error) produces `CHAT_RESPONSE_INVALID` via separate constant
- Error cases: function itself never throws
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- `navigator.onLine` check is browser-specific; server/non-browser runtime must skip or replace.

---

### REQ-CHAT-014: Invalid null response produces feedback

**Layer:** failure
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (UX for degraded backend)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:16-22` (implementation — `INVALID_RESPONSE_FEEDBACK` constant)
- `react/hooks/useChat.hooks.ts:264-266` (guard in `runSubmission`)

**Observed behavior:** When `api.sendMessage` resolves with `null` (invalid/unusable response), `INVALID_RESPONSE_FEEDBACK` is set: `{ kind: "error", code: "CHAT_RESPONSE_INVALID", message: "The AI service did not return a usable response. Try again.", retryable: true, actionLabel: "Try again" }`.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: null return from `api.sendMessage`
- Output: persistent screen feedback with code `CHAT_RESPONSE_INVALID`
- Side effects: `actions.setScreenFeedback(INVALID_RESPONSE_FEEDBACK)` called
- Invariants: null response never appends an assistant message
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-015: Dataset context injection is screen-owned

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (architectural correctness)
**Risk tags:** —

**Source evidence:**
- `logic/chatComposition.logic.ts:18-25` (implementation — `mapChatStateWithDataset`)
- `integration/req-006.page-agnostic-dataset.wiring.integration.test.ts:53` (test)
- `integration/ab-002.screen-context-injection.integration.test.ts:14` (test)

**Observed behavior:** The default orchestrator injects `activeDatasetId: null`. Screen-level orchestrators (e.g., AgenticResearchPage) inject their own dataset context. The feature layer never hardcodes route literals or screen-specific dataset IDs.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `{ state: Omit<ChatState, "activeDatasetId">, activeDatasetId: string | null }`
- Output: `ChatState` (with `activeDatasetId` injected)
- Side effects: verified_none (pure function, cite: `logic/chatComposition.logic.ts:18-25`)
- Invariants: feature layer is context-neutral; dataset comes from composition/screen layer
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-016: Chat request payload shape

**Layer:** api
**Status:** confirmed
**Confidence:** observed
**Criticality:** high
**Criticality reason:** fragile_consumer (backend contract; any shape change breaks the integration)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.ts:129-139` (implementation — fetch body construction)
- `__types__/api.types.ts:21-26` (type — `SendChatMessageInput`)

**Observed behavior:** POST body is `JSON.stringify({ message, attachments: [], model, messages: history[], dataset_id: string|null })`. `message` is the trimmed user input, `model` is selectedModelId, `messages` is the bounded history window, `attachments` are current attachment array, `dataset_id` is injected from screen context.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `SendChatMessageInput` + options with `datasetId`
- Output: HTTP POST to `{baseUrl}{endpoint}` with JSON body
- Side effects: network request (POST)
- Invariants: `Content-Type: application/json` header always set; dataset_id is null for direct mode
- Error cases: non-ok response → throws `ChatRequestError` with code `CHAT_REQUEST_HTTP_ERROR`; parse failure → throws `ChatRequestError` with code `CHAT_RESPONSE_PARSE_ERROR`
- Transaction boundary: not_applicable
- Concurrency: not_applicable (no request deduplication or queuing observed)
- Auth requirement: unknown (no auth header in code; may be handled by proxy/middleware)

---

### REQ-CHAT-017: State adapter port pattern

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (architectural portability)
**Risk tags:** —

**Source evidence:**
- `__types__/chat.types.ts:103-106` (type — `ChatStatePort`)
- `__types__/chat.types.ts:108` (type — `UseChatStatePort`)

**Observed behavior:** `ChatStatePort` splits store shape into `state` (read) and `actions` (write). Any screen can provide its own implementation of `UseChatStatePort` (e.g., LandingPage creates an isolated store). The adapter exposes `Omit<ChatState, "activeDatasetId">` — dataset is composed separately.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: Zustand store (or any state source) conforming to `ChatStatePort`
- Output: `{ state, actions }` satisfying the port contract
- Side effects: not_applicable (type contract)
- Invariants: state and actions are separated; no direct store access outside adapter
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-018: Clipboard copy failure resilience

**Layer:** failure
**Status:** confirmed
**Confidence:** tested
**Criticality:** low
**Criticality reason:** low_usage (convenience feature)
**Risk tags:** —

**Source evidence:**
- `integration/err-004.clipboard-failure.integration.test.ts:6` (test)

**Observed behavior:** When `navigator.clipboard.writeText` throws, `handleCopy` resolves without throwing and `copiedId` remains null (no false-positive copy indicator).
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: message ID + content string
- Output: void (resolves successfully regardless of clipboard outcome)
- Side effects: attempts clipboard write; on failure, no state change
- Invariants: never throws; copiedId stays null on failure
- Error cases: clipboard denied/failed → silent recovery
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-019: Invalid file attachment handling

**Layer:** failure
**Status:** confirmed
**Confidence:** tested
**Criticality:** low
**Criticality reason:** low_usage (file attachment edge case)
**Risk tags:** —

**Source evidence:**
- `integration/req-008.invalid-attachments.integration.test.ts:13` (test)

**Observed behavior:** When `FileReader.readAsDataURL` triggers `onerror`, the drop handler resolves without throwing and does not call `addAttachments`. Failed files are silently discarded.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: drag-drop event with files that fail to read
- Output: void (resolves successfully)
- Side effects: no attachments added to state
- Invariants: partial file-read failures don't crash UI or add corrupted attachments
- Error cases: FileReader error → silent discard
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-020: Model bootstrap on mount (lazy, once)

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (initial state hydration)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:487-505` (implementation — bootstrap effect in `useChatIntegration`)

**Observed behavior:** On first integration mount, if `modelOptions.length === 0` and not already loading, triggers a single `refetchModels()` call. Uses `initialModelBootstrapRequestedRef` to prevent duplicate fetches. If models are populated (e.g., from parent), no fetch occurs.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: mount lifecycle + current state `{ modelOptions, isModelsLoading }`
- Output: triggers `refetchModels()` exactly once when models are empty
- Side effects: one HTTP GET to models endpoint (lazy)
- Invariants: never double-fetches; skips if models already populated
- Error cases: fetch failure handled by refetchModels (falls back to FALLBACK_CHAT_MODELS)
- Transaction boundary: not_applicable
- Concurrency: ref-guarded single invocation
- Auth requirement: not_applicable

**Migration notes:**
- Uses React `useEffect` for mount detection; non-React implementation needs equivalent lifecycle hook or explicit init call.

---

### REQ-CHAT-021: Retry last submission

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (error recovery UX)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:326-329` (implementation — `retryLastSubmission`)
- `__types__/chat.types.ts:153` (type — `ChatActions.retryLastSubmission`)

**Observed behavior:** `retryLastSubmission()` re-runs the last saved submission payload (value, model, history, attachments, datasetId) via `runSubmission`. If no previous submission exists (`lastSubmissionRef.current === null`), returns immediately.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: none (uses stored last submission)
- Output: same behavior as original submit (setSending, API call, response handling)
- Side effects: same as REQ-CHAT-003 (sending lifecycle)
- Invariants: does not re-add user message or re-record history; only re-executes the API call portion
- Error cases: no prior submission → no-op
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-022: Draft preservation during history navigation

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (UX polish)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:337-359` (implementation — `handleHistory`)
- `logic/chatSubmission.logic.ts:96-104` (implementation — `shouldRestoreDraftValue`)

**Observed behavior:** When user presses "up" and cursor is null, current input is saved to `draftValueRef`. When navigating "down" past the end (nextValue is empty, cursor was non-null), the saved draft is restored. `shouldRestoreDraftValue` determines the restore condition.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: direction ("up"/"down"), current history cursor, current input value
- Output: updated input value (either from history or restored draft)
- Side effects: modifies UI state `setValue`
- Invariants: draft is captured only on first "up" from null cursor; restore only when navigating "down" past newest
- Error cases: empty history → no-op (guard at line 339)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-023: Store state initialization

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only
**Risk tags:** —

**Source evidence:**
- `logic/chatStore.logic.ts:16-29` (implementation — `createInitialChatStoreCoreState`)

**Observed behavior:** Creates deterministic initial state: `messages: [], inputHistory: [], historyCursor: null, isSending: false, modelOptions: [], selectedModelId: null, isModelsLoading: false, screenFeedback: null`.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `Record<string, never>` (empty object for API consistency)
- Output: `ChatStoreCoreState` with all fields at zero/null/empty defaults
- Side effects: verified_none (pure function, cite: `logic/chatStore.logic.ts:16-29`)
- Invariants: deterministic; always returns fresh object (no shared references)
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-024: Append message immutability

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only
**Risk tags:** —

**Source evidence:**
- `logic/chatStore.logic.ts:37-41` (implementation — `appendMessage`)

**Observed behavior:** Returns a new array `[...messages, message]` — does not mutate the input array.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `{ messages: ChatMessage[], message: ChatMessage }`
- Output: new `ChatMessage[]` with message appended
- Side effects: verified_none (pure function, new array, cite: `logic/chatStore.logic.ts:37-41`)
- Invariants: input array is never mutated; output is a new reference
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-025: Append input history resets cursor

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only
**Risk tags:** —

**Source evidence:**
- `logic/chatStore.logic.ts:49-56` (implementation — `appendInputHistory`)

**Observed behavior:** Returns `{ inputHistory: [...prev, value], historyCursor: null }`. Adding a new history entry always resets the cursor to null (not navigating).
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `{ inputHistory: string[], value: string }`
- Output: `{ inputHistory: string[], historyCursor: null }`
- Side effects: verified_none (pure function, cite: `logic/chatStore.logic.ts:49-56`)
- Invariants: historyCursor is always null after append; input array not mutated
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-026: Logic layer framework-agnostic invariant

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** high
**Criticality reason:** fragile_consumer (architectural boundary — breaks portability if violated)
**Risk tags:** —

**Source evidence:**
- `integration/ab-003.logic-framework-agnostic.integration.test.ts:14` (test)

**Observed behavior:** All files under `logic/` must not import `react`, `zustand`, or `next/` modules. Must not access `window.`, `document.`, or `navigator.` browser globals.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: all `.ts` files under `logic/` directory
- Output: zero matches for framework/DOM imports
- Side effects: not_applicable (architectural invariant)
- Invariants: logic layer is 100% framework-agnostic and browser-agnostic
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-027: No cross-feature state/orchestrator imports

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** high
**Criticality reason:** fragile_consumer (architectural boundary)
**Risk tags:** —

**Source evidence:**
- `integration/ab-001.no-cross-feature-domain-imports.integration.test.ts:14` (test)

**Observed behavior:** No file in `features/ai-chat/` imports `/state/` or `/orchestrators/` from other features. Only allowed cross-feature import is `@/features/charts/contracts/chart.types` (shared type contract).
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: all `.ts/.tsx` files under `features/ai-chat/`
- Output: zero imports from other features' state/orchestrator layers
- Side effects: not_applicable (architectural invariant)
- Invariants: ai-chat is self-contained except for shared type contracts
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-028: Type contracts live under `__types__/`

**Layer:** domain
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (contract discoverability)
**Risk tags:** —

**Source evidence:**
- `integration/req-005.contract-location.integration.test.ts:42` (test)

**Observed behavior:** All type imports within `ai-chat` feature must use `@/features/ai-chat/__types__/` path (or `@/features/charts/contracts/`). No legacy `types/` folder allowed. Seven specific contract files must exist.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: all `.ts/.tsx` files in feature
- Output: zero imports from legacy `types/` path; all contract files exist
- Side effects: not_applicable (architectural invariant)
- Invariants: contract files are the canonical source of truth for shapes
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-029: Dependency injection via ChatDeps bundle

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** high
**Criticality reason:** fragile_consumer (enables testability and portability of the entire chat feature)
**Risk tags:** —

**Source evidence:**
- `__types__/chat.types.ts:190-197` (type — `ChatDeps`)
- `logic/chatOrchestrator.logic.ts:31-40` (implementation — `createChatDeps`)
- `fixtures/chatLogicDeps.fixture.ts:23-31` (test fixture — real logic injected)

**Observed behavior:** All chat behavior receives dependencies via `ChatDeps = { state, actions, api, logic }`. Logic functions are injected (not imported directly by hooks). API functions are injected. State is injected via port. This enables any test to swap any dependency layer independently.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `{ state: ChatState, actions: ChatStateActions, api: ChatApiDeps, logic: ChatLogicDeps }`
- Output: fully-wired chat behavior
- Side effects: not_applicable (structural pattern)
- Invariants: no direct imports of logic/API in hooks — all via deps; enables full testability without module mocking
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-030: Runtime dependency injection (timestamps, IDs)

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (test determinism)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:105-108` (type — `ChatLogicRuntimeDeps`)
- `react/hooks/useChat.hooks.ts:218-219` (implementation — defaults to `Date.now` and `String(timestamp)`)

**Observed behavior:** `useChatLogic` accepts optional `runtimeDeps: { now?, createId? }`. Default: `now = Date.now`, `createId = (timestamp) => String(timestamp)`. Tests can inject deterministic time/ID sources.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: optional `{ now: () => number, createId: (timestamp: number) => string }`
- Output: timestamps and IDs used in message creation
- Side effects: not_applicable (injection point)
- Invariants: defaults are `Date.now` and stringified timestamp; injection is optional
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

## Phase 3: Convention Scanner

### Framework Conventions Detected

| Convention | Location | Impact |
|------------|----------|--------|
| Zustand global singleton store | `react/state/zustand/aiChatStore.ts` | State is shared across component tree; tests must reset or isolate |
| Zustand shallow selector | `react/state/adapters/aiChatState.adapter.ts` | Prevents unnecessary re-renders; target must replicate selector equality semantics |
| React `useEffect` for mount lifecycle | `react/hooks/useChat.hooks.ts:229-234` | isMounted ref pattern; non-React targets need equivalent lifecycle |
| React `useRef` for mutable cross-render state | `react/hooks/useChat.hooks.ts:224-226` | Draft ref, submission ref, mounted ref; framework-agnostic equivalent needed |
| Next.js `"use client"` directive | view components | Marks client-side components; irrelevant for non-Next.js targets |
| Port/adapter DI pattern | `__types__/chat.types.ts` | Enables framework swapping; preserve in any target |
| `process.env.NODE_ENV` conditional debug logging | `api/chatApi.ts:23` | Development-only logging; target may replicate or discard |
| `process.env.NEXT_PUBLIC_DEBUG_EFFECTS` | `react/hooks/useChat.hooks.ts:15` | Optional debug effects; framework-specific |

### Inferred Architectural Requirements

These are pattern-level observations, not testable behavioral contracts:

1. Logic layer is strictly pure functions with no side effects — fully extractable to any language.
2. React layer orchestrates lifecycle and effect timing — behavioral contracts from logic layer must be preserved regardless of framework.
3. State shape is defined by types, not by Zustand — any state container satisfying `ChatStatePort` is valid.

---

## Phase 3b: Numerical/String Precision

No precision-sensitive logic detected. The chat feature does not perform:
- Financial calculations
- Measurement accumulation
- Unicode normalization (messages are passed through as-is)
- Locale-sensitive comparisons

The only numerical constant is the history window size (10) which is an integer boundary, not a precision concern.

---

## Open Questions Discovered This Pass

1. **Auth requirement for API endpoints:** No auth header is set in `chatApi.ts`. Is authentication handled by the Next.js proxy (`/api/ai` route) or by the backend directly? Status: unknown.
2. **Abort/cancel mechanism for in-flight chat messages:** `fetchChatModels` uses AbortController, but `sendChatMessage` does not. Is there an intentional lack of cancellation for chat requests? Status: unknown.
3. **UIFeedback.tsx coverage:** No dedicated test file exists. Is it covered transitively via ChatSidebar tests or is this a gap? Status: unknown.

---

## Amendments to Prior Artifacts

None (this is Pass 1).

---

## Risk Tags/Markers Raised This Pass

None blocking. All requirements are confirmed with `tested` or `observed` confidence. No contradictions detected.

---

## Entrypoints Discovered or Removed

### Discovered
- `resolveSubmitFeedback` — exported error mapping function (behavioral contract)
- `setFallbackModels` — exported helper
- `setFetchedModels` — exported helper

### Removed
None.

---

## Summary

- **Total requirements extracted:** 30
- **Tested confidence:** 19 (63%)
- **Observed confidence:** 11 (37%)
- **Framework-agnostic (logic layer):** REQ-CHAT-001, 004, 005, 006, 007, 008, 009, 015, 023, 024, 025 (11 reqs)
- **React-specific (hook/lifecycle):** REQ-CHAT-002, 003, 012, 020, 021, 022, 030 (7 reqs)
- **API layer:** REQ-CHAT-010, 011, 016 (3 reqs)
- **Architectural invariants:** REQ-CHAT-026, 027, 028, 029 (4 reqs)
- **Failure/error handling:** REQ-CHAT-013, 014, 017, 018, 019 (5 reqs)

Chunking limit reached at 30. No additional requirements remain unextracted for the logic and API layers. The `react/views/` component behavioral contracts (rendering, event handling beyond clipboard/drag-drop) were not extracted in this pass as they are React-specific UI behavior rather than core logic contracts.
