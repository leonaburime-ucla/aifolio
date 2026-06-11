# Merged Requirements Index: ai-chat Reverse-Spec

Total: 100 requirements | 4 extraction passes | 2026-06-09

---

## 1. Message Submission Pipeline

- **REQ-CHAT-001** [domain] [tested] [medium]: Empty/whitespace input short-circuits submission with zero side effects.
- **REQ-CHAT-002** [domain] [tested] [medium]: Submit ordering invariant: addInputToHistory > addMessage > setSending(true) > API call.
- **REQ-CHAT-003** [domain] [tested] [medium]: setSending(false) fires in finally block on all outcomes (success, null, error).
- **REQ-CHAT-006** [domain] [tested] [medium]: History window bounded to 10 entries; current user message always last.
- **REQ-CHAT-016** [api] [observed] [high]: POST payload shape { message, attachments, model, messages, dataset_id }; Content-Type: application/json. Risk: fragile_consumer
- **REQ-CHAT-021** [domain] [observed] [medium]: retryLastSubmission re-runs stored payload without re-adding user message.
- **REQ-CHAT-037** [transaction] [tested] [medium]: Submit is non-atomic: sync prepare (1-7), async execute (8), mount-guarded finalize (9-11). Risk: temporal_coupling
- **REQ-CHAT-039** [transaction] [observed] [medium]: Retry uses stored submission (last-write-wins on multiple submits).
- **REQ-CHAT-085** [domain] [observed] [medium]: Enter (no modifier) submits; Shift/Meta/Ctrl+Enter inserts newline.
- **REQ-CHAT-094** [domain] [tested] [high]: Primary workflow: type > send > optimistic user msg > spinner > response or error.

## 2. Model Selection & Bootstrap

- **REQ-CHAT-004** [domain] [tested] [medium]: Selection precedence: existing > result.currentModel > first model > null.
- **REQ-CHAT-005** [domain] [tested] [medium]: Fallback models applied deterministically on fetch failure (4 hardcoded Gemini models).
- **REQ-CHAT-010** [api] [tested] [medium]: fetchChatModels aborts after 5000ms; produces MODEL_FETCH_TIMEOUT (retryable).
- **REQ-CHAT-020** [domain] [observed] [medium]: Model bootstrap fires once on mount when modelOptions empty; ref-guarded.
- **REQ-CHAT-047** [domain] [observed] [medium]: Model fetch must complete before submission includes model ID; null tolerated. Risk: temporal_coupling
- **REQ-CHAT-048** [concurrency] [observed] [medium]: Bootstrap model fetch once-only per lifecycle via ref guard.
- **REQ-CHAT-049** [concurrency] [observed] [low]: refetchModelsRef indirection prevents stale closure in bootstrap effect.
- **REQ-CHAT-054** [failure] [tested] [medium]: Model fetch failure is silent; fallback models applied, no user feedback.
- **REQ-CHAT-057** [integration] [tested] [medium]: GET /llm/gemini-models with AbortController; response { status, currentModel, models }.
- **REQ-CHAT-069** [performance] [tested] [medium]: Model fetch timeout default 5000ms (only timeout in feature).
- **REQ-CHAT-083** [domain] [observed] [medium]: Native select shows model options; disabled when loading/empty; aria-labeled.
- **REQ-CHAT-096** [domain] [observed] [medium]: Model change is prospective only; captured at submit time for next request.

## 3. Chat State Management

- **REQ-CHAT-023** [domain] [observed] [low]: createInitialChatStoreCoreState returns deterministic empty defaults.
- **REQ-CHAT-024** [domain] [observed] [low]: appendMessage is immutable (new array; never mutates input).
- **REQ-CHAT-025** [domain] [observed] [low]: appendInputHistory always resets historyCursor to null.
- **REQ-CHAT-031** [data] [observed] [low]: Message ID is String(Date.now()) by default; injectable via runtimeDeps.createId.
- **REQ-CHAT-032** [data] [observed] [medium]: ChatMessage shape: { id, role: user|assistant, content, createdAt, chartSpec? }.
- **REQ-CHAT-033** [data] [observed] [medium]: ChatModelOption shape: { id, label }; id opaque, passed through to backend.
- **REQ-CHAT-034** [data] [observed] [medium]: State is ephemeral (no persistence); lost on page refresh.
- **REQ-CHAT-045** [concurrency] [observed] [medium]: Global Zustand store is module-level singleton shared by all adapter consumers. Risk: concurrency_contract
- **REQ-CHAT-046** [concurrency] [observed] [low]: useShallow selector prevents re-renders for unrelated state changes.
- **REQ-CHAT-092** [domain] [observed] [medium]: Global store persists across SPA nav; isolated store is per-page lifecycle.
- **REQ-CHAT-093** [domain] [observed] [low]: No URL-driven state; no bookmarkable chat; purely in-memory.

## 4. API Communication

- **REQ-CHAT-009** [domain] [tested] [medium]: normalizeChatApiResult handles 3 backend shapes (object, string, Gemini parts); null for invalid.
- **REQ-CHAT-011** [api] [observed] [medium]: API adapter mode: research > /chat-research, direct > /chat (forces dataset_id: null).
- **REQ-CHAT-050** [api] [observed] [medium]: Next.js proxy at /api/ai/* is transparent passthrough; strips hop-by-hop only.
- **REQ-CHAT-056** [integration] [tested] [high]: Backend chat endpoint contract: POST JSON body, ChatApiResponse with result union type. Risk: fragile_consumer
- **REQ-CHAT-058** [integration] [observed] [medium]: No automatic retry; all recovery is user-initiated via retryLastSubmission.
- **REQ-CHAT-059** [integration] [observed] [medium]: Chat message send has no timeout or cancellation; can hang indefinitely.
- **REQ-CHAT-060** [integration] [observed] [medium]: Proxy forwards upstream errors verbatim; no transformation or timeout.
- **REQ-CHAT-061** [integration] [observed] [medium]: Base URL: browser /api/ai; server AI_API_URL > NEXT_PUBLIC_AI_API_URL > 127.0.0.1:8000. Risk: environmental_contract
- **REQ-CHAT-066** [security] [observed] [medium]: CORS avoided entirely via same-origin proxy; no cross-origin requests.
- **REQ-CHAT-070** [performance] [tested] [medium]: History window limits payload to 10 messages max.
- **REQ-CHAT-071** [performance] [observed] [low]: No message size limit enforced client-side; backend must validate.

## 5. Error Handling & User Feedback

- **REQ-CHAT-013** [failure] [observed] [medium]: resolveSubmitFeedback: total function mapping unknown errors to ScreenFeedback (6 codes).
- **REQ-CHAT-014** [failure] [observed] [medium]: Null API response produces CHAT_RESPONSE_INVALID feedback (retryable).
- **REQ-CHAT-051** [failure] [tested] [high]: Complete failure-to-feedback mapping: HTTP 5xx/4xx, parse, abort, offline, unknown. Risk: fragile_consumer
- **REQ-CHAT-052** [failure] [tested] [medium]: ChatRequestError is sole error class (CHAT_REQUEST_HTTP_ERROR | CHAT_RESPONSE_PARSE_ERROR).
- **REQ-CHAT-053** [failure] [observed] [high]: ScreenFeedback { kind, code, message, retryable?, actionLabel? } is sole user-facing error contract. Risk: fragile_consumer
- **REQ-CHAT-055** [failure] [observed] [medium]: 6 submission feedback codes + 2 internal model-fetch codes; stable string identifiers.
- **REQ-CHAT-091** [domain] [observed] [medium]: UIFeedback uses role="alert" for errors, role="status" for info/warning (a11y).
- **REQ-CHAT-095** [domain] [observed] [medium]: Retry workflow: inline error > "Try again" > stored payload re-sent > same lifecycle.

## 6. Input History Navigation

- **REQ-CHAT-007** [domain] [tested] [low]: resolveHistoryCursor is total (never throws); bounds cursor to valid range or null.
- **REQ-CHAT-022** [domain] [observed] [low]: Draft preserved on first ArrowUp; restored on ArrowDown past newest entry.
- **REQ-CHAT-084** [domain] [tested] [low]: ArrowUp/Down in textarea navigate input history; preventDefault blocks cursor.

## 7. Consumer Composition Contracts

- **REQ-CHAT-015** [domain] [tested] [medium]: Dataset context is screen-owned; feature layer is context-neutral (null default).
- **REQ-CHAT-017** [domain] [observed] [medium]: ChatStatePort splits read/write; any store implementation is valid.
- **REQ-CHAT-029** [domain] [observed] [high]: ChatDeps bundle { state, actions, api, logic } enables full DI; no direct imports in hooks. Risk: fragile_consumer
- **REQ-CHAT-030** [domain] [observed] [low]: Runtime deps (now, createId) injectable for test determinism.
- **REQ-CHAT-062** [integration] [observed] [low]: CopilotKit is independent (shares zero state/API with ai-chat).
- **REQ-CHAT-074** [domain] [observed] [high]: useChatSurfaceOrchestrator is sole composition root; 6 injectable ports. Risk: fragile_consumer
- **REQ-CHAT-075** [domain] [observed] [high]: ChatSidebar accepts orchestrator as injectable hook-prop (default useChatOrchestrator). Risk: fragile_consumer
- **REQ-CHAT-076** [domain] [observed] [medium]: LandingPage uses isolated store (separate Zustand instance); prevents cross-contamination.
- **REQ-CHAT-077** [domain] [observed] [medium]: AgenticResearchPage uses global store + custom chart/dataset ports + research mode.
- **REQ-CHAT-078** [domain] [observed] [medium]: LandingPage routes chart specs to CopilotKit recharts store.
- **REQ-CHAT-098** [domain] [observed] [high]: ChatIntegration is sole public interface (flat union: UiState + State + Actions). Risk: fragile_consumer
- **REQ-CHAT-099** [domain] [observed] [high]: 6 customization ports: statePort, chartActionsPort, activeDatasetId, useActiveDatasetId, mode, apiAdapter. Risk: fragile_consumer

## 8. UI Behavioral Parity Contracts

- **REQ-CHAT-008** [domain] [tested] [medium]: Chart spec fan-out: null > 0 calls, array > N in order, single > 1 call.
- **REQ-CHAT-018** [failure] [tested] [low]: Clipboard copy failure is silent; copiedId stays null.
- **REQ-CHAT-019** [failure] [tested] [low]: Invalid file attachments silently discarded (FileReader onerror).
- **REQ-CHAT-081** [domain] [observed] [medium]: Messages render in chronological append-only order via array map.
- **REQ-CHAT-082** [domain] [observed] [medium]: Sending indicator + disabled send button while isSending; textarea stays editable.
- **REQ-CHAT-086** [domain] [observed] [medium]: Assistant messages as markdown (remarkGfm); user messages as plain text.
- **REQ-CHAT-087** [domain] [observed] [low]: Auto-scroll to bottom on new message or sending change via rAF.
- **REQ-CHAT-088** [domain] [tested] [low]: Copy-to-clipboard with 2s transient "check" indicator; last-copy wins.
- **REQ-CHAT-089** [domain] [tested] [medium]: Drag-and-drop file attachment; Promise.allSettled; partial failures filtered.
- **REQ-CHAT-090** [domain] [observed] [low]: Empty state placeholder "Ask a question to get started." when messages empty.
- **REQ-CHAT-100** [domain] [observed] [medium]: Sidebar layout: sticky right 360px, calc(100vh-64px); no responsive collapse.

## 9. Framework Portability Contracts

- **REQ-CHAT-026** [domain] [tested] [high]: Logic layer is framework-agnostic (zero react/zustand/DOM imports). Risk: fragile_consumer
- **REQ-CHAT-027** [domain] [tested] [high]: No cross-feature state/orchestrator imports; only shared type contracts. Risk: fragile_consumer
- **REQ-CHAT-028** [domain] [tested] [medium]: Type contracts live under __types__/; no legacy types/ path.
- **REQ-CHAT-063** [integration] [observed] [medium]: Browser API deps: fetch, AbortController, setTimeout, clipboard, FileReader, rAF. Risk: environmental_contract
- **REQ-CHAT-079** [infrastructure] [observed] [medium]: ChatSidebar SSR-disabled via dynamic import { ssr: false }. Risk: environmental_contract
- **REQ-CHAT-080** [domain] [observed] [medium]: .web.ts suffix signals platform-specific module; browser APIs injectable.
- **REQ-CHAT-097** [domain] [observed] [high]: 10 React patterns mapped to Vue/Svelte/Angular equivalents. Risk: fragile_consumer

## 10. Concurrency & Lifecycle

- **REQ-CHAT-012** [domain] [tested] [medium]: isMountedRef prevents all post-unmount state mutations across async operations.
- **REQ-CHAT-035** [access-control] [observed] [medium]: No frontend-enforced auth; no headers/tokens; backend enforcement unknown.
- **REQ-CHAT-036** [access-control] [observed] [low]: No roles, tenants, or permissions; all state anonymous.
- **REQ-CHAT-038** [concurrency] [observed] [medium]: No domain-level concurrent submission guard; UI must disable button. Risk: concurrency_contract
- **REQ-CHAT-040** [concurrency] [observed] [low]: Model fetch idempotent-safe but not deduplicated; concurrent calls allowed.
- **REQ-CHAT-041** [concurrency] [observed] [high]: Feature relies on JS single-threaded event loop for state serialization. Risk: concurrency_contract
- **REQ-CHAT-042** [concurrency] [tested] [medium]: isMountedRef guard at every post-await site (3 in runSubmission, 1 in refetchModels).
- **REQ-CHAT-043** [concurrency] [observed] [low]: AbortController used only for model fetch; chat send has none.
- **REQ-CHAT-044** [concurrency] [observed] [medium]: Navigate-away leaves stale isSending=true in global store. Risk: needs_clarification
- **REQ-CHAT-064** [compliance] [observed] [medium]: No PII persistence; volatile memory only; messages forwarded to backend.
- **REQ-CHAT-065** [compliance] [observed] [medium]: Attachments as dataUrl with no size/type/count validation. Risk: needs_clarification
- **REQ-CHAT-067** [security] [observed] [medium]: No CSRF token; mitigated by JSON content-type CORS preflight.
- **REQ-CHAT-068** [security] [observed] [low]: No encryption/redaction; React escaping + react-markdown (no raw HTML).
- **REQ-CHAT-072** [observability] [observed] [low]: Dev-only debug logging (console); zero production output.
- **REQ-CHAT-073** [observability] [observed] [low]: No production observability (no Sentry, analytics, or perf tracking).
