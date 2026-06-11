# Artifact 3: Boundaries, Failures, and Compliance — ai-chat

Feature path: `web/nextjs/src/features/ai-chat/`
Extraction date: 2026-06-09
Pass: 3 (Boundaries, Failures, and Compliance)
Methodology: `AI-Dev-Shop-speckit/skills/reverse-spec/SKILL.md` v2.0.0
Prior artifacts consumed: `artifact-1-core-logic.md` (REQ-CHAT-001–030), `artifact-2-data-access.md` (REQ-CHAT-031–050)

---

## Phase 7: Failure Matrix

### Chat Message Send — Failure Matrix

| Error Case | Error Shape Produced | User-Facing Feedback Code | User-Facing Message | Retryable | Action Label | Evidence |
|------------|---------------------|---------------------------|---------------------|-----------|--------------|----------|
| HTTP response non-ok, status >= 500 | `ChatRequestError { code: "CHAT_REQUEST_HTTP_ERROR", status: N }` | `CHAT_SERVICE_UNAVAILABLE` | "The AI service returned an error. Try again in a moment." | true | "Try again" | `chatApi.ts:148-153`, `useChat.hooks.ts:51-60` |
| HTTP response non-ok, status < 500 | `ChatRequestError { code: "CHAT_REQUEST_HTTP_ERROR", status: N }` | `CHAT_REQUEST_REJECTED` | "The AI service rejected the request. Check the request and try again." | true | "Try again" | `chatApi.ts:148-153`, `useChat.hooks.ts:51-60` |
| Response body JSON parse failure | `ChatRequestError { code: "CHAT_RESPONSE_PARSE_ERROR", cause: original }` | `CHAT_RESPONSE_INVALID` | "The AI service returned an unreadable response. Try again." | true | "Try again" | `chatApi.ts:164-177`, `useChat.hooks.ts:63-76` |
| Response JSON valid but normalization returns null | `null` (not an error — resolved value) | `CHAT_RESPONSE_INVALID` | "The AI service did not return a usable response. Try again." | true | "Try again" | `chatApi.ts:156-163`, `useChat.hooks.ts:16-22`, `useChat.hooks.ts:264-266` |
| Request aborted (AbortError) | `DOMException { name: "AbortError" }` | `CHAT_REQUEST_ABORTED` | "The request was canceled before a response was returned." | false | (none) | `useChat.hooks.ts:78-84` |
| Network offline (`navigator.onLine === false`) | any error + offline check | `CHAT_OFFLINE` | "You're offline. Reconnect to the internet and try again." | true | "Try again" | `useChat.hooks.ts:86-93` |
| Unknown/generic error | any unmatched error | `CHAT_REQUEST_FAILED` | "Could not reach the AI service. Check your connection and try again." | true | "Try again" | `useChat.hooks.ts:96-102` |
| Component unmounted during request | (no error surfaced) | (none) | (none — silently discarded) | n/a | n/a | `useChat.hooks.ts:262,281,284` |

### Model Fetch — Failure Matrix

| Error Case | Return Shape | User-Facing Behavior | Retryable | Evidence |
|------------|-------------|---------------------|-----------|----------|
| Timeout (AbortError from AbortController) | `{ status: "error", error: { code: "MODEL_FETCH_TIMEOUT", retryable: true, message: "Model endpoint timed out." } }` | Fallback models applied silently; no error feedback shown to user | true | `chatApi.ts:248-253`, `err-005.timeout-retryable-contract.integration.test.ts` |
| Network/fetch exception (non-AbortError) | `{ status: "error", error: { code: "MODEL_FETCH_FAILED", retryable: true, message: "Model endpoint request failed." } }` | Fallback models applied silently | true | `chatApi.ts:255-259`, `chatApi.unit.test.ts:239-261` |
| HTTP response non-ok | `null` | Fallback models applied silently | n/a | `chatApi.ts:212-219` |
| Payload invalid (`status !== "ok"` or `!models`) | `{ status: "error", error: { code: "MODEL_FETCH_FAILED", retryable: true, message: "Model endpoint returned an invalid payload." } }` | Fallback models applied silently | true | `chatApi.ts:222-233` |
| Hook-level `fetchModels` throws | caught by `refetchModels` catch block | Fallback models applied silently | n/a | `useChat.hooks.ts:427-432` |
| Component unmounted during fetch | (guard fires) | Response discarded, no state mutation | n/a | `useChat.hooks.ts:416` |

### Clipboard Copy — Failure Matrix

| Error Case | Behavior | User Feedback | Evidence |
|------------|----------|---------------|----------|
| `navigator.clipboard.writeText` throws | `copiedId` stays null; no exception propagates | No visual "copied" indicator appears | `useChatSidebar.web.ts:118-123`, `err-004.clipboard-failure.integration.test.ts` |

### File Attachment Drop — Failure Matrix

| Error Case | Behavior | User Feedback | Evidence |
|------------|----------|---------------|----------|
| `FileReader.readAsDataURL` triggers `onerror` | `Promise.allSettled` filters out rejected results; no `addAttachments` call if all fail | No error shown; failed files silently discarded | `useChatSidebar.web.ts:140-152`, `req-008.invalid-attachments.integration.test.ts` |
| Zero files in drop event | Early return | No change | `useChatSidebar.web.ts:138` |
| Partial file failures | Only successfully-read files added; failed ones filtered out | Successful files appear; failed ones silently dropped | `useChatSidebar.web.ts:144-149` |

---

### REQ-CHAT-051: Complete failure-to-feedback mapping contract

**Layer:** failure
**Status:** confirmed
**Confidence:** tested
**Criticality:** high
**Criticality reason:** fragile_consumer (all error paths route through this single mapping; any change breaks the UI contract)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:42-103` (implementation — `resolveSubmitFeedback`)
- `integration/req-002.sending-reset.integration.test.ts:117` (test — verifies `CHAT_REQUEST_FAILED`)
- `api/chatApi.ts:28-44` (implementation — `ChatRequestError` class)

**Observed behavior:** `resolveSubmitFeedback` is a total function mapping `unknown` errors to `ScreenFeedback`. The mapping priority is:
1. `ChatRequestError` with `CHAT_REQUEST_HTTP_ERROR` code → branch on status >= 500 vs < 500
2. `ChatRequestError` with `CHAT_RESPONSE_PARSE_ERROR` code → `CHAT_RESPONSE_INVALID`
3. `DOMException` with `name === "AbortError"` → `CHAT_REQUEST_ABORTED` (info, not retryable)
4. `navigator.onLine === false` → `CHAT_OFFLINE`
5. Fallthrough → `CHAT_REQUEST_FAILED`
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `unknown` (any caught value from `api.sendMessage` rejection)
- Output: `ScreenFeedback` — always produces a valid shape; never throws
- Side effects: reads `globalThis.navigator?.onLine` (browser environment probe)
- Invariants: priority ordering is fixed — HTTP status check first, then parse error, then abort, then offline, then fallthrough; every branch produces `retryable: true` except `CHAT_REQUEST_ABORTED`
- Error cases: function itself is total (never throws)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- `navigator.onLine` is browser-specific; SSR or non-browser runtimes must skip this branch or inject an equivalent check.
- The duck-typing check (`"code" in error && error.code === "..."`) is intentional — avoids `instanceof` checks that break across module boundaries.

---

### REQ-CHAT-052: ChatRequestError is the sole error class for chat API failures

**Layer:** failure
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (error contract boundary)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.ts:28-44` (implementation — class definition)
- `api/chatApi.ts:148-153` (usage — non-ok HTTP)
- `api/chatApi.ts:171-177` (usage — parse error)
- `chatApi.unit.test.ts:86-152` (test — verifies both error codes)

**Observed behavior:** `ChatRequestError extends Error` with: `code: "CHAT_REQUEST_HTTP_ERROR" | "CHAT_RESPONSE_PARSE_ERROR"`, `status?: number`, standard `cause`. It is the only custom error class in the chat API layer. It is thrown (not returned) from `sendChatMessageToEndpoint`.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: constructor `{ code, message, status?, cause? }`
- Output: Error subclass with `name: "ChatRequestError"`, typed `code`, optional `status`
- Side effects: not_applicable
- Invariants: `code` is a closed union of exactly two values; `status` is only present when `code === "CHAT_REQUEST_HTTP_ERROR"`
- Error cases: not_applicable (this IS the error type)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-053: ScreenFeedback shape is the sole user-facing error contract

**Layer:** failure
**Status:** confirmed
**Confidence:** observed
**Criticality:** high
**Criticality reason:** fragile_consumer (UI components, retry logic, and dismiss logic all depend on this shape)
**Risk tags:** —

**Source evidence:**
- `__types__/uiFeedback.types.ts:4-10` (type definition)
- `react/views/components/UIFeedback.tsx:12-16` (consumer — style map keyed on `kind`)
- `react/views/components/ChatSidebar.tsx:99-107` (consumer — render + action wiring)

**Observed behavior:** `ScreenFeedback = { kind: "error" | "warning" | "info", code: string, message: string, retryable?: boolean, actionLabel?: string }`. This is the ONLY feedback shape used for persistent inline errors in the chat feature. The `UIFeedback` component renders it with role-based accessibility (`role="alert"` for error, `role="status"` for others), styled per `kind`, with optional action button and dismiss button.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `ScreenFeedback | null` from state
- Output: rendered inline feedback panel or `null` (hidden)
- Side effects: not_applicable (pure render)
- Invariants: `kind` drives visual style and ARIA role; `retryable` + `actionLabel` together enable the action button; `onAction` callback triggers `retryLastSubmission()`; `onDismiss` sets `screenFeedback(null)`
- Error cases: `feedback === null` → returns null (no render)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-054: Model fetch failure is silent (no user-facing error)

**Layer:** failure
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (UX choice — graceful degradation)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:402-437` (implementation — `refetchModels` applies fallback on any failure)
- `integration/err-002.fetch-models-fallback.integration.test.ts:9` (test)

**Observed behavior:** When `fetchModels` returns error or throws, `refetchModels` catches and applies `FALLBACK_CHAT_MODELS` without setting any `screenFeedback`. The user sees hardcoded fallback models in the selector but receives no error notification. Model fetch failure is entirely invisible to the user.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: model fetch failure (any path)
- Output: fallback models applied; `setModelsLoading(false)` called; NO `setScreenFeedback(...)` called
- Side effects: state mutation (model options updated to fallbacks)
- Invariants: user never sees an error for model fetch failure; they only see fallback model options
- Error cases: not_applicable (this IS the error handler)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-055: Feedback codes are stable, string-typed identifiers

**Layer:** failure
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (API contract for UI consumers)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:16-22,42-103` (implementation — all codes defined inline)
- `__types__/api.types.ts:57-61` (type — `ChatApiError` codes)

**Observed behavior:** The complete set of feedback codes used in the feature:

**Chat submission feedback codes (set via `setScreenFeedback`):**
- `CHAT_SERVICE_UNAVAILABLE` — HTTP 5xx from backend
- `CHAT_REQUEST_REJECTED` — HTTP 4xx from backend
- `CHAT_RESPONSE_INVALID` — parse failure or null normalization result
- `CHAT_REQUEST_ABORTED` — AbortError (info level)
- `CHAT_OFFLINE` — navigator.onLine === false
- `CHAT_REQUEST_FAILED` — unknown/generic failure

**Model fetch error codes (internal, never shown to user):**
- `MODEL_FETCH_TIMEOUT` — AbortController timeout fired
- `MODEL_FETCH_FAILED` — any other model fetch failure

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: not_applicable (enumeration)
- Output: codes are string constants used for programmatic handling (retry wiring, test assertions)
- Side effects: not_applicable
- Invariants: codes are not localized (English); messages are user-facing English strings; `retryable` is the semantic flag for action enablement, not the code
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

## Phase 8: Integration Boundary Extraction

### Integration Boundary 1: Backend AI API

---

### REQ-CHAT-056: Backend API contract — chat message endpoint

**Layer:** integration
**Status:** confirmed
**Confidence:** tested
**Criticality:** high
**Criticality reason:** fragile_consumer (any backend response shape change breaks the frontend)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.ts:129-139` (implementation — request construction)
- `__types__/api.types.ts:4-12` (type — `ChatApiResponse`)
- `__types__/api.types.ts:21-26` (type — `SendChatMessageInput`)
- `chatApi.unit.test.ts:9-46` (test — request shape assertion)
- `chatApi.unit.test.ts:48-84` (test — direct mode assertion)

**Observed behavior:**

**Request contract:**
- Method: POST
- Endpoints: `/chat-research` (research mode) or `/chat` (direct mode)
- Headers: `Content-Type: application/json` (only header set)
- Body: `{ message: string, attachments: ChatAttachment[], model: string|null, messages: ChatHistoryMessage[], dataset_id: string|null }`
- No auth header, no API key, no bearer token

**Response contract (expected):**
- Status 200 with JSON body conforming to `ChatApiResponse`:
  ```
  { status: "ok" | "error",
    result?: string | Array<{type:string, text?:string}> | {message?:string, chartSpec?: ChartSpec|ChartSpec[]|null},
    error?: string,
    model?: string }
  ```
- Non-200 → `ChatRequestError` with `CHAT_REQUEST_HTTP_ERROR`
- 200 but invalid JSON → `ChatRequestError` with `CHAT_RESPONSE_PARSE_ERROR`
- 200, valid JSON, but result normalizes to null → returns `null` (not error)

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `SendChatMessageInput` + `SendChatMessageOptions`
- Output: `ChatAssistantPayload | null`
- Side effects: network POST to backend
- Invariants: always sets `Content-Type: application/json`; direct mode forces `dataset_id: null`; no credentials header; no retry
- Error cases: see failure matrix above
- Transaction boundary: not_applicable
- Concurrency: not_applicable (no request deduplication)
- Auth requirement: unknown (frontend sends none; backend enforcement is opaque)

---

### REQ-CHAT-057: Backend API contract — model list endpoint

**Layer:** integration
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (model selector population)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.ts:188-265` (implementation)
- `__types__/api.types.ts:14-19` (type — `ModelsApiResponse`)
- `chatApi.unit.test.ts:154-237` (test — all paths)

**Observed behavior:**

**Request contract:**
- Method: GET
- Endpoint: `/llm/gemini-models`
- Headers: none set explicitly (browser defaults via fetch)
- Body: none
- AbortController signal attached for timeout

**Response contract (expected):**
- Status 200 with JSON body conforming to `ModelsApiResponse`:
  ```
  { status: "ok" | "error",
    currentModel?: string,
    models?: Array<{id: string, label: string}>,
    error?: string }
  ```
- Non-200 → returns `null`
- 200, `status !== "ok"` or `!models` → returns `MODEL_FETCH_FAILED` error result
- 200, valid → returns `FetchChatModelsSuccessResult`

**Timeout behavior:**
- Default timeout: 5000ms (`DEFAULT_MODELS_TIMEOUT_MS`)
- Configurable via `options.timeoutMs`
- On timeout: `controller.abort()` → `AbortError` → `MODEL_FETCH_TIMEOUT` result
- Timeout is always cleared in finally path (no timer leak)

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `FetchChatModelsInput` (empty `Record<string, never>`) + `FetchChatModelsOptions`
- Output: `FetchChatModelsResult | null`
- Side effects: HTTP GET with abort signal
- Invariants: timeout always cleared; AbortError specifically produces `MODEL_FETCH_TIMEOUT` (not `MODEL_FETCH_FAILED`); non-AbortError DOMExceptions produce `MODEL_FETCH_FAILED`
- Error cases: see model fetch failure matrix
- Transaction boundary: not_applicable
- Concurrency: no request deduplication; concurrent calls allowed
- Auth requirement: unknown (no auth sent)

---

### REQ-CHAT-058: No retry semantics for any API call

**Layer:** integration
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (intentional simplicity)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.ts:114-179` (implementation — `sendChatMessageToEndpoint` has no retry loop)
- `api/chatApi.ts:188-265` (implementation — `fetchChatModels` has no retry loop)
- `react/hooks/useChat.hooks.ts:402-437` (implementation — `refetchModels` single attempt)

**Observed behavior:** Neither the chat message send nor the model fetch implements automatic retry. Each API call is a single attempt. On failure:
- Chat message: error surfaces as `ScreenFeedback` with `retryable: true`, user manually clicks "Try again" → `retryLastSubmission()`
- Model fetch: fallback models applied silently, no retry

There is no exponential backoff, no jitter, no circuit breaker, no retry queue anywhere in the feature.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: any API call
- Output: single attempt; success or failure
- Side effects: not_applicable
- Invariants: zero automatic retries; all retry is user-initiated via `retryLastSubmission()`
- Error cases: see failure matrices
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-059: Chat message send has no timeout or cancellation

**Layer:** integration
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (potential UX issue for slow AI responses)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.ts:114-179` (implementation — no AbortController, no timeout)
- `react/hooks/useChat.hooks.ts:250-260` (implementation — `await api.sendMessage(...)` with no timeout wrapper)

**Observed behavior:** `sendChatMessageToEndpoint` calls `fetch()` without any `signal` option and without a timeout wrapper. The request can hang indefinitely until:
1. The server responds (success or error)
2. The TCP connection times out (browser/OS level, typically 60-300s)
3. The component unmounts (response discarded via isMountedRef guard, but request continues)

There is no user-initiated cancel mechanism for in-flight chat messages.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: fetch call to chat endpoint
- Output: no timeout; waits indefinitely for server response
- Side effects: `isSending === true` persists until response arrives or component unmounts
- Invariants: no AbortController; no cancellation API exposed to users; unmount guard only prevents state mutation — does not abort the network request
- Error cases: see REQ-CHAT-044 (stale isSending after navigate-away)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Open questions:**
- For AI model responses that may take 10-60+ seconds, is indefinite wait acceptable? The user has no cancel button. Navigating away leaves a zombie request.

---

### Integration Boundary 2: Next.js Proxy Route

---

### REQ-CHAT-060: Proxy is a transparent passthrough with no error transformation

**Layer:** integration
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (infrastructure contract)
**Risk tags:** —

**Source evidence:**
- `app/api/ai/[...path]/route.ts:26-73` (implementation — `proxyRequest`)
- `app/api/ai/[...path]/route.ts:9-13` (implementation — hop-by-hop headers)

**Observed behavior:** The proxy:
1. Strips hop-by-hop headers: `connection`, `content-length`, `host`
2. Forwards all remaining request headers (including any cookies)
3. Forwards request body as `ArrayBuffer`
4. Forwards query parameters
5. Returns upstream status, statusText, headers, and body verbatim
6. Uses `redirect: "manual"` (no automatic redirect following)
7. Uses `cache: "no-store"` (no caching)
8. For HEAD requests: response body is null (per HTTP spec)

**Error passthrough behavior:**
- Backend 503 → proxy returns 503 to client
- Backend 401 → proxy returns 401 to client
- Backend connection refused → proxy itself throws (Next.js returns 500)
- Backend timeout → proxy hangs until backend timeout fires (no proxy-level timeout)

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: any HTTP request to `/api/ai/*`
- Output: upstream response forwarded verbatim
- Side effects: server-side fetch to `{AI_API_URL}/{path}{query}`
- Invariants: no body transformation; no header injection; no auth addition; no rate limiting; no response caching; all HTTP methods supported (GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD)
- Error cases: upstream unreachable → Next.js 500 (not caught by proxy code); upstream error responses forwarded as-is
- Transaction boundary: not_applicable
- Concurrency: Next.js handles concurrent requests
- Auth requirement: none at proxy level

---

### REQ-CHAT-061: Base URL resolution strategy

**Layer:** integration
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (deployment configuration)
**Risk tags:** environmental_contract

**Source evidence:**
- `core/config/aiApi.ts:1-28` (implementation)
- `api/chatApi.ts:15,60-61` (usage — injected `resolveBaseUrl`)

**Observed behavior:**

**Browser context (client components):**
- Returns `/api/ai` (same-origin proxy path)
- Detection: `typeof window !== "undefined"`

**Server context (SSR, API routes):**
- Priority: `process.env.AI_API_URL` > `process.env.NEXT_PUBLIC_AI_API_URL` > `"http://127.0.0.1:8000"`

**Injectable:** The `chatApi` functions accept `runtimeDeps.resolveBaseUrl` override for testing.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: runtime environment (browser vs server) + env vars
- Output: base URL string
- Side effects: reads `process.env` (server) or returns constant (browser)
- Invariants: browser always uses proxy (`/api/ai`); server has 3-level fallback chain; default is `http://127.0.0.1:8000`
- Error cases: missing env vars → falls back to localhost
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

`[ENVIRONMENTAL CONTRACT]` — Deployment must set `AI_API_URL` or `NEXT_PUBLIC_AI_API_URL` for non-local environments. Missing both means all server-side API calls go to `127.0.0.1:8000`.

---

### Integration Boundary 3: CopilotKit

---

### REQ-CHAT-062: CopilotKit integration is a thin opaque wrapper

**Layer:** integration
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** low_usage (alternative sidebar, minimal custom code)
**Risk tags:** —

**Source evidence:**
- `react/views/components/CopilotChatSidebar.tsx:1-19` (implementation)
- `app/api/copilotkit/route.ts:1-11` (backend route)

**Observed behavior:** `CopilotChatSidebar` renders `<CopilotChat>` from `@copilotkit/react-ui` with fixed labels (`title: "AI Chat"`, `initial: "Ask a question to get started."`). It does NOT use the ai-chat feature's state, hooks, types, or API layer. It is a completely independent chat surface backed by CopilotKit's own runtime.

The backend endpoint (`/api/copilotkit`) delegates to a separate feature module (`features/ag-ui-chat/api/copilotRuntime.adapter`), NOT to the ai-chat feature.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: CopilotKit manages its own state and API calls
- Output: rendered chat UI via CopilotKit components
- Side effects: CopilotKit makes its own API calls to `/api/copilotkit`
- Invariants: CopilotChatSidebar shares ZERO state with the ai-chat feature; they are independent chat surfaces; the only connection is they can appear in the same screen layout
- Error cases: CopilotKit handles its own errors internally (opaque)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable (CopilotKit manages its own)

---

### Integration Boundary 4: Browser Environment

---

### REQ-CHAT-063: Browser API dependencies

**Layer:** integration
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (environment assumptions)
**Risk tags:** environmental_contract

**Source evidence:**
- `api/chatApi.ts:57-58` (implementation — `globalThis.fetch`)
- `api/chatApi.ts:62` (implementation — `new AbortController()`)
- `useChatSidebar.web.ts:64-72` (implementation — `window.requestAnimationFrame`, `window.setTimeout`, `navigator.clipboard`)
- `useChatSidebar.web.ts:36-49` (implementation — `FileReader`)
- `useChat.hooks.ts:86` (implementation — `globalThis.navigator?.onLine`)
- `useChat.hooks.ts:25` (implementation — `globalThis.location?.pathname`)

**Observed behavior:** The feature requires these browser APIs:
1. `fetch` (global) — HTTP transport
2. `AbortController` — timeout mechanism for model fetch
3. `setTimeout` / `clearTimeout` — timeout scheduling
4. `navigator.clipboard.writeText` — copy to clipboard (graceful failure)
5. `FileReader.readAsDataURL` — file attachment processing (graceful failure)
6. `navigator.onLine` — offline detection (optional — uses `?.` operator)
7. `window.requestAnimationFrame` / `cancelAnimationFrame` — scroll animation
8. `globalThis.location?.pathname` — debug logging only

All browser APIs used in business-critical paths (`fetch`, `AbortController`, `setTimeout`) are injectable via runtime deps for testing. Browser APIs used in UI-only paths (`clipboard`, `FileReader`, `rAF`) are injectable via `ChatSidebarRuntimeDeps`.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: browser environment with standard Web APIs
- Output: feature functions correctly when APIs are present
- Side effects: not_applicable (inventory)
- Invariants: all critical APIs are injectable; non-critical APIs degrade gracefully (clipboard fail → no crash, FileReader fail → no attachment)
- Error cases: missing `navigator.clipboard` → copy silently fails; missing `navigator.onLine` → offline detection skipped (goes to fallthrough)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

## Phase 9: Privacy, Compliance, and Security

---

### REQ-CHAT-064: No PII persistence or retention

**Layer:** compliance
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (privacy by design — ephemeral chat)
**Risk tags:** —

**Source evidence:**
- `react/state/zustand/aiChatStore.ts:88-89` (implementation — no persist middleware)
- REQ-CHAT-034 (artifact-2 — verified no persistence)

**Observed behavior:** Chat messages that may contain user-entered PII are stored only in browser process memory (Zustand store without persistence). They are:
- Never written to localStorage, sessionStorage, IndexedDB, or cookies
- Never sent to any analytics or logging endpoint from the frontend
- Lost on page refresh, tab close, or SPA navigation that remounts the store
- Not included in any data export or deletion flow (none exists)

Messages ARE sent to the backend via POST requests. Backend-side retention/logging is outside this feature's scope.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: user-typed messages potentially containing PII
- Output: stored in volatile memory only; forwarded to backend for AI processing
- Side effects: backend receives messages (retention policy is backend's concern)
- Invariants: frontend never persists chat content to any durable storage; no indexing, no search, no export
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-065: Attachments sent as dataUrl with no client-side validation

**Layer:** compliance
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (potential data size and security surface)
**Risk tags:** —

**Source evidence:**
- `useChatSidebar.web.ts:36-49` (implementation — `defaultReadFileAsDataUrl`)
- `__types__/chat.types.ts:51-56` (type — `ChatAttachment`)
- `api/chatApi.ts:133` (implementation — attachments sent in request body)

**Observed behavior:** File attachments are:
1. Read as `dataUrl` via `FileReader.readAsDataURL` (base64-encoded)
2. Stored as `ChatAttachment = { name: string, type: string, size: number, dataUrl: string }`
3. Sent to backend as part of the `attachments` array in the JSON POST body

**No client-side validation exists for:**
- File size (no maximum enforced)
- File type (no allowlist/blocklist)
- File count (no maximum per message)
- Content scanning (no malware/virus check)
- dataUrl length (no maximum enforced)

The `type` field defaults to `"application/octet-stream"` if the File has no type. The `size` field records the original file size but is not used for any validation.

**Normative contract:** unknown (no documented limits)
**Rewrite decision:** human_decision_required

**Behavior classification:** unclear
**Preservation decision:** human_decision_required

**Contract:**
- Input: any file(s) dropped onto the chat surface
- Output: base64 dataUrl representation stored in memory and sent to backend
- Side effects: large files will inflate JSON body size significantly (base64 is ~33% larger than binary)
- Invariants: no size limit; no type filter; no count limit; backend must handle or reject oversized payloads
- Error cases: FileReader failure → file silently discarded (REQ-CHAT-019)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Open questions:**
- Is there a backend-enforced maximum request body size? Without frontend validation, users could attach extremely large files (e.g., 100MB), producing a JSON payload of ~133MB.
- Should certain file types be blocked (executables, archives)?

`[NEEDS CLARIFICATION]` — No file size or type validation exists on the client. Determine if this is intentional (backend validates) or a gap requiring frontend guards.

---

### REQ-CHAT-066: CORS handled entirely by same-origin proxy

**Layer:** security
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (deployment security)
**Risk tags:** —

**Source evidence:**
- `core/config/aiApi.ts:22-28` (implementation — browser uses `/api/ai`)
- `app/api/ai/[...path]/route.ts:26-73` (implementation — proxy)

**Observed behavior:** The frontend avoids CORS by routing all browser-originated API calls through the same-origin Next.js proxy at `/api/ai/*`. This means:
1. Browser requests go to same-origin → no CORS preflight needed
2. The proxy makes server-to-server requests to the backend (no CORS)
3. No `Access-Control-*` headers are set by the proxy
4. No CORS configuration exists in the frontend codebase

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: browser fetch to `/api/ai/*`
- Output: same-origin response (no CORS headers needed)
- Side effects: not_applicable
- Invariants: frontend NEVER makes cross-origin requests to the backend directly; all requests go through the proxy
- Error cases: if proxy is misconfigured/down → standard fetch failure (not CORS error)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-067: No CSRF protection mechanism

**Layer:** security
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (security consideration)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.ts:129-139` (implementation — no CSRF token in headers or body)
- `app/api/ai/[...path]/route.ts:16-23` (implementation — no CSRF validation)

**Observed behavior:** No CSRF token is generated, stored, or sent with chat API requests. The proxy does not validate any CSRF token. The POST requests to `/api/ai/chat` and `/api/ai/chat-research` are protected only by:
1. Same-origin policy (requests from the same domain)
2. `Content-Type: application/json` header (which blocks simple form POSTs — provides limited CSRF protection via browser's CORS preflight requirement for non-simple content types from other origins)

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: POST requests from the browser
- Output: no CSRF token validation
- Side effects: not_applicable
- Invariants: JSON content-type provides implicit CSRF protection (browsers require CORS preflight for cross-origin JSON POSTs); same-origin proxy adds another layer; no explicit CSRF token mechanism
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-068: No encryption or redaction of message content

**Layer:** security
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (no field-level security)
**Risk tags:** —

**Source evidence:**
- Full scan of `features/ai-chat/` — zero references to encryption, redaction, sanitization, or masking

**Observed behavior:** User messages and assistant responses are stored and transmitted as plain text (JSON strings). No:
- Field-level encryption
- PII redaction or masking
- Content sanitization (XSS prevention is handled by React's default JSX escaping)
- Input validation beyond empty-string check

Assistant messages rendered via `react-markdown` have standard React escaping for non-markdown content. Markdown rendering with `remarkGfm` may render links, images, and tables from the AI response.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: plaintext messages from user and AI
- Output: rendered as-is (user) or via markdown (assistant)
- Side effects: not_applicable
- Invariants: no sanitization; no redaction; React handles XSS for plain content; markdown rendering may create clickable links from AI output
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- If AI responses contain malicious markdown (e.g., `[click](javascript:...)` URLs), `react-markdown` with `remarkGfm` has standard link handling but no explicit URL sanitization. React-markdown does NOT render raw HTML by default (safe), but links are clickable.

---

## Performance Envelopes

---

### REQ-CHAT-069: Model fetch timeout is 5000ms by default

**Layer:** performance
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (timeout contract clients depend on)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.ts:21` (implementation — `const DEFAULT_MODELS_TIMEOUT_MS = 5000`)
- `api/chatApi.ts:194` (implementation — `options?.timeoutMs ?? DEFAULT_MODELS_TIMEOUT_MS`)
- `integration/req-009.models-timeout.integration.test.ts:10` (test)
- `integration/err-005.timeout-retryable-contract.integration.test.ts:10` (test)

**Observed behavior:** The model fetch endpoint has a 5-second timeout by default. Configurable via `options.timeoutMs`. After timeout fires, the AbortController aborts the fetch and a deterministic `MODEL_FETCH_TIMEOUT` result is produced.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `timeoutMs` option (default 5000)
- Output: abort + error result after timeout elapses
- Side effects: AbortController fires
- Invariants: 5000ms is the only hardcoded timeout in the entire feature; chat message send has NO timeout
- Error cases: see model fetch failure matrix
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-070: History window limits payload to 10 messages

**Layer:** performance
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (controls request payload size)
**Risk tags:** —

**Source evidence:**
- `logic/chatSubmission.logic.ts:35-53` (implementation — `buildChatHistoryWindow`)
- `unit/dr-002.history-window.boundary.unit.test.ts:14` (test)

**Observed behavior:** The `buildChatHistoryWindow` function truncates message history to at most `windowSize` entries (default 10). The current user message is always the last entry. Older messages are dropped from the beginning. This bounds the request payload size to a maximum of 10 history entries regardless of total transcript length.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: full message array + current message + windowSize (default 10)
- Output: at most 10 entries in the `messages` field sent to backend
- Side effects: not_applicable
- Invariants: window size is the only message-count limit; no per-message character limit; no total payload byte limit
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-071: No message size limit enforced client-side

**Layer:** performance
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** low_usage (edge case for extremely long messages)
**Risk tags:** —

**Source evidence:**
- `logic/chatSubmission.logic.ts:22-27` (implementation — only checks empty, no max length)
- `api/chatApi.ts:132` (implementation — sends `message: input.value` without truncation)
- `react/views/components/ChatBar.tsx:68-94` (implementation — textarea with no `maxLength`)

**Observed behavior:** There is no character/byte limit on:
- User input message length (textarea has no `maxLength` attribute)
- Individual history message content
- Total request body size

The textarea has a visual max-height (`max-h-36` in fixed mode, `max-h-60` in embedded mode) with `resize-none`, which provides soft visual containment but does not limit actual input length.
**Normative contract:** unknown (no documented limit)
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: user text of any length
- Output: entire text sent to backend
- Side effects: extremely long messages produce large request bodies
- Invariants: no client-side truncation or rejection based on length
- Error cases: backend may reject oversized bodies (opaque to frontend)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

## Observability

---

### REQ-CHAT-072: Development-only debug logging

**Layer:** observability
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (development tooling)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.ts:22` (implementation — `const DEBUG_AI_PROXY = process.env.NODE_ENV === "development"`)
- `api/chatApi.ts:122-127,141-147,157-162,165-170,200-204,213-218,224-228,244-247` (usage — `console.warn` calls)
- `react/hooks/useChat.hooks.ts:15` (implementation — `const DEBUG_EFFECTS = process.env.NEXT_PUBLIC_DEBUG_EFFECTS === "1"`)
- `react/hooks/useChatSidebar.web.ts:5` (implementation — same flag)
- `react/views/components/ChatSidebar.tsx:12` (implementation — same flag)

**Observed behavior:** Two debug logging systems exist:

1. **API debug logging** (`DEBUG_AI_PROXY`):
   - Active when `NODE_ENV === "development"`
   - Logs request URLs, non-ok responses, invalid payloads, parse failures, model fetch errors via `console.warn`
   - Prefixed with `[ai-chat]` or `[ai-chat] fetch-models`

2. **Effect debug logging** (`DEBUG_EFFECTS`):
   - Active when `NEXT_PUBLIC_DEBUG_EFFECTS === "1"` (opt-in even in development)
   - Logs effect executions, mount events, state transitions via `console.log`
   - Prefixed with `[chat-debug]`

Neither system sends data to external services. Both are `console.*` only. Neither is active in production builds.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: environment flags
- Output: `console.warn` / `console.log` calls in development only
- Side effects: browser console output (no network calls)
- Invariants: zero production logging; zero analytics; zero error tracking sent from frontend code; no Sentry, no PostHog, no LangSmith integration in this feature
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-073: No production observability instrumentation

**Layer:** observability
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (observability gap)
**Risk tags:** —

**Source evidence:**
- Full scan of `features/ai-chat/` — zero imports from Sentry, PostHog, analytics, or logging libraries
- Full scan — no `fetch` instrumentation, no performance timing, no error reporting to external services

**Observed behavior:** The ai-chat feature has NO production observability:
- No error tracking (no Sentry breadcrumbs or error reporting)
- No analytics events (no PostHog, no Amplitude, no GA)
- No performance metrics (no timing spans, no latency recording)
- No structured logging beyond development console

Errors are surfaced to the user via `ScreenFeedback` and nowhere else.

**Normative contract:** matches observed (despite AGENTS.md recommending LangSmith/Sentry/PostHog integration)
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: any error or event in the feature
- Output: no external observability emission
- Side effects: verified_none for external services (cite: no analytics imports in any feature file)
- Invariants: feature is fully dark to monitoring systems
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Open questions:**
- The project's `AGENTS.md` describes LangSmith, Sentry, and PostHog integration. Is the absence of observability in ai-chat intentional (MVP/prototype stage) or a gap to be filled?

---

## Open Questions Discovered This Pass

1. **Attachment size/type validation (REQ-CHAT-065):** No client-side file size or type limits exist. Is validation performed server-side, or is this a gap requiring frontend guards? Priority: important.

2. **Chat request indefinite hang (REQ-CHAT-059):** Chat message send has no timeout. For slow AI responses, `isSending` can persist indefinitely. Is a timeout needed, or is this accepted? Priority: important.

3. **Production observability (REQ-CHAT-073):** Zero production monitoring exists. Is this intentional for this feature stage? Priority: nice-to-have.

4. **Markdown XSS surface (REQ-CHAT-068):** AI responses rendered via `react-markdown` + `remarkGfm` may contain clickable links. Is URL sanitization needed? `react-markdown` does not render raw HTML by default (safe), but does render `[text](url)` links. Priority: nice-to-have.

---

## Amendments to Prior Artifacts

### Amendment to REQ-CHAT-013 (Error-to-feedback mapping)

REQ-CHAT-013 documented the mapping function. This pass provides the complete failure matrix (Phase 7) showing all entry paths into `resolveSubmitFeedback` and the exhaustive code enumeration. The Pass 1 extraction is accurate but this pass adds the full error surface including `ChatRequestError` class definition and duck-typing detection mechanism.

### Amendment to REQ-CHAT-010 (Model fetch timeout)

REQ-CHAT-010 documented the timeout contract. This pass adds the integration boundary context: the timeout is specifically implemented via `AbortController` + `setTimeout` with injectable runtime deps for both. The `clearTimeout` always fires in a finally-equivalent position (line 263). Timer leak is impossible.

### Amendment to REQ-CHAT-050 (Proxy route)

REQ-CHAT-050 documented the proxy behavior. This pass adds the failure passthrough contract: the proxy has no error transformation, no retry, no timeout of its own. Backend failures propagate verbatim to the client. If the backend is unreachable, the proxy's `fetch()` itself rejects and Next.js returns a 500.

---

## Risk Tags/Markers Raised This Pass

| Marker | REQ ID | Severity | Description |
|--------|--------|----------|-------------|
| `[NEEDS CLARIFICATION]` | REQ-CHAT-065 | Important | No file size or type validation on attachments — gap or server-validated? |
| `[ENVIRONMENTAL CONTRACT]` | REQ-CHAT-061 | Advisory | Deployment must set `AI_API_URL` env var for non-local environments |
| `[ENVIRONMENTAL CONTRACT]` | REQ-CHAT-063 | Advisory | Feature requires standard browser Web APIs (fetch, AbortController, FileReader, clipboard) |

---

## Entrypoints Discovered or Removed

### Discovered
- `ChatRequestError` class — internal error type for API failures (not exported but structurally important)
- `resolveRuntimeDeps` — internal factory for injectable API dependencies
- `defaultReadFileAsDataUrl` — internal file reader helper
- `INVALID_RESPONSE_FEEDBACK` constant — internal feedback shape for null responses
- `DEBUG_AI_PROXY` / `DEBUG_EFFECTS` — development logging flags

### Removed
None.

---

## Summary

- **Requirements extracted this pass:** 23 (REQ-CHAT-051 through REQ-CHAT-073)
- **Cumulative total:** 73 (30 from Pass 1 + 20 from Pass 2 + 23 from Pass 3)
- **Confidence breakdown (this pass):** tested: 6, observed: 17
- **Blocking markers:** 1 (`[NEEDS CLARIFICATION]` on attachment validation)
- **Important markers:** 0 new (prior pass markers still active)
- **Advisory markers:** 2 (`[ENVIRONMENTAL CONTRACT]`)

### Key Findings

1. **Comprehensive failure mapping exists** — every error path produces a deterministic `ScreenFeedback` shape with stable codes, user messages, and retry flags. The mapping is total (never throws).

2. **No automatic retry anywhere** — all recovery is user-initiated via "Try again" button. Model fetch degrades silently to hardcoded fallbacks.

3. **Chat message send has no timeout or cancellation** — unlike model fetch (5s timeout), chat requests hang indefinitely. The only "escape" is navigating away (which triggers the unmount guard).

4. **Attachments are unbounded** — no file size, type, or count validation on the client. Files are base64-encoded as dataUrl and sent in the JSON body. Large attachments could produce massive payloads.

5. **Zero production observability** — errors are shown to the user and nowhere else. No Sentry, no analytics, no performance tracking.

6. **CORS is elegantly solved** — same-origin proxy eliminates all CORS concerns. The proxy is transparent and adds no transformation or auth.

7. **CopilotKit is completely independent** — it shares no state, no types, and no API calls with the ai-chat feature. They are parallel implementations of chat UI.

8. **Security posture is minimal but adequate for the scope** — no CSRF tokens (mitigated by JSON content-type CORS protection), no auth (delegated to backend), no encryption (not needed for ephemeral in-memory state), XSS handled by React's default escaping.
