# Artifact 2: Data, Access, and Atomicity — ai-chat

Feature path: `web/nextjs/src/features/ai-chat/`
Extraction date: 2026-06-09
Pass: 2 (Data & Access)
Methodology: `AI-Dev-Shop-speckit/skills/reverse-spec/SKILL.md` v2.0.0
Prior artifact consumed: `artifact-1-core-logic.md` (REQ-CHAT-001 through REQ-CHAT-030)

---

## Phase 4: Database-Resident Behavior

### Database Behavior Inventory

This feature has **no direct database access**. All state is held in-memory via Zustand stores (browser process memory). There are no:

- Triggers, stored procedures, views, or materialized views
- Generated columns, check constraints, or foreign key cascades
- Row-level security policies
- Sequences or custom DB-level ID generation
- Persistence across page loads (state is ephemeral)

**Status:** `verified_none` — exhaustive inspection of all files under `features/ai-chat/` confirms zero database imports, no ORM usage, no SQL queries, no IndexedDB/localStorage calls. Citation: full directory scan of `api/`, `logic/`, `react/`, `__types__/` — no persistence layer present.

### ID Format Contracts

---

### REQ-CHAT-031: Message ID format is timestamp-string by default

**Layer:** data
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (no external consumers parse this ID)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:219` (implementation — default `createId = (timestamp) => String(timestamp)`)
- `react/hooks/useChat.hooks.ts:105-108` (type — `ChatLogicRuntimeDeps`)
- `__types__/chat.types.ts:28` (type — `id: string`)

**Observed behavior:** Message IDs are generated client-side via `String(Date.now())`. The ID is a stringified Unix timestamp in milliseconds. The generation function is injectable via `runtimeDeps.createId`.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: timestamp from `now()` (default `Date.now`)
- Output: string ID — default format is stringified millisecond timestamp (e.g., `"1717945200000"`)
- Side effects: not_applicable
- Invariants: ID is unique per message within a session (relies on timestamp resolution); ID format is not parsed or decomposed by any consumer in this codebase; IDs are not sent to the backend (only used for local React keys and copy tracking)
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: two messages submitted within the same millisecond would collide — mitigated by single-threaded JS event loop and async request serialization
- Auth requirement: not_applicable

**Open questions:**
- Are message IDs used for anything on the backend? Current evidence: IDs are NOT included in the API request payload (`chatApi.ts:129-139` sends `message`, `model`, `messages[]`, `attachments`, `dataset_id` — no `id` field). Status: `verified_none` for backend consumption.

**Migration notes:**
- Injectable `createId` allows UUID or ULID replacement without changing any consumers.
- Timestamp-based IDs are NOT lexicographically unique across concurrent sessions (irrelevant since store is per-browser-tab).

---

### REQ-CHAT-032: ChatMessage shape contract

**Layer:** data
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (shared contract consumed by multiple screens)
**Risk tags:** —

**Source evidence:**
- `__types__/chat.types.ts:27-33` (type definition)

**Observed behavior:** Every message in the transcript conforms to: `{ id: string, role: "user" | "assistant", content: string, createdAt: number, chartSpec?: ChartSpec | null }`.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: not_applicable (type contract)
- Output: not_applicable (type contract)
- Side effects: not_applicable
- Invariants: `role` is a closed union ("user" | "assistant"); `createdAt` is a Unix millisecond timestamp; `chartSpec` is optional and only set on assistant messages via `createAssistantChatMessage`; `id` is a non-empty string
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-033: ChatModelOption shape contract

**Layer:** data
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (drives model selector UI and API payload)
**Risk tags:** —

**Source evidence:**
- `__types__/chat.types.ts:35-38` (type definition)
- `__types__/api.types.ts:17` (backend response shape — `models?: Array<{ id: string; label: string }>`)

**Observed behavior:** Model options are `{ id: string, label: string }`. The `id` is sent to the backend as the `model` field in chat requests. The `label` is displayed in the UI model selector.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: populated from backend GET `/llm/gemini-models` response or from `FALLBACK_CHAT_MODELS` constant
- Output: consumed by UI model selector and sent as `model` field in POST requests
- Side effects: not_applicable
- Invariants: `id` is opaque to the frontend (passed through as-is to backend); frontend does not validate model IDs
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### REQ-CHAT-034: In-memory state is ephemeral (no persistence)

**Layer:** data
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (user expectation: chat history is lost on refresh)
**Risk tags:** —

**Source evidence:**
- `react/state/zustand/aiChatStore.ts:88-89` (implementation — `create<AiChatState>` with no persist middleware)
- `logic/chatStore.logic.ts:16-29` (implementation — `createInitialChatStoreCoreState` always returns empty defaults)

**Observed behavior:** The Zustand store uses `create()` with no persistence middleware (no `persist`, no `localStorage`, no `sessionStorage`, no IndexedDB). Every page load starts from `createInitialChatStoreCoreState({})` which produces empty arrays and null values.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: page load / SPA navigation that remounts the store
- Output: all chat state resets to defaults (messages=[], inputHistory=[], etc.)
- Side effects: verified_none (no write to any persistence layer; cite: no `persist` import in store file, no localStorage/sessionStorage/IndexedDB usage anywhere in feature)
- Invariants: chat transcript, input history, model selection, and feedback are all lost on page refresh
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- If persistence is added later, the store shape is already serializable (no functions, no circular refs in state slice). Only the `messages[].chartSpec` field contains nested objects that would need schema versioning.

---

## Phase 5: Access-Control Matrix

### Authentication Model

---

### REQ-CHAT-035: No frontend-enforced authentication

**Layer:** access-control
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (security boundary is NOT enforced here)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.ts:129-139` (implementation — no Authorization header, no token injection)
- `app/api/ai/[...path]/route.ts:16-23` (implementation — proxy forwards all request headers as-is, no auth injection)
- `react/hooks/useChat.hooks.ts` (no auth check before submit)
- `react/orchestrators/chatOrchestrator.ts` (no auth guard)

**Observed behavior:** The frontend chat feature adds no authentication headers, tokens, or session cookies to its API calls. The Next.js proxy route (`/api/ai/[...path]`) forwards incoming request headers without adding or validating auth. There is no login gate, session check, or role check anywhere in the ai-chat feature code.
**Normative contract:** unknown (backend may enforce auth independently, but the frontend neither sends nor validates credentials)
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: any HTTP request from the browser
- Output: request is forwarded to backend without auth modification
- Side effects: verified_none (no token refresh, no redirect to login)
- Invariants: the frontend never gates chat functionality behind auth; browser cookies (if any) are forwarded automatically by fetch's default credential behavior, but no explicit `credentials: "include"` is set
- Error cases: if the backend rejects with 401/403, this would surface as `CHAT_REQUEST_REJECTED` (status < 500) per REQ-CHAT-013
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: public (from frontend perspective); backend enforcement is unknown

**Open questions:**
- Does the backend enforce API key, session cookie, or other auth? The proxy forwards browser cookies by default (same-origin fetch). If the backend requires auth, it gets it from cookies — but this is NOT explicitly configured or validated on the frontend side.

---

### REQ-CHAT-036: No role-based or tenant-scoped restrictions in frontend

**Layer:** access-control
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (frontend has no RBAC or tenancy concept)
**Risk tags:** —

**Source evidence:**
- Full scan of `features/ai-chat/` — zero references to `role`, `user`, `tenant`, `permission`, `auth`, or `session` (excluding test/type-only mentions)
- `__types__/chat.types.ts` — no user/role fields in any type

**Observed behavior:** The feature has no concept of user identity, roles, tenants, or permissions. All state is anonymous. Messages have no `userId` field. The dataset context (`activeDatasetId`) is screen-provided but not user-scoped.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: not_applicable
- Output: not_applicable
- Side effects: not_applicable
- Invariants: no user identity concept exists in this feature; any auth boundary is the backend's responsibility
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### Access-Control Matrix

| Entrypoint | Actor/Role | Ownership Rule | Tenant/Data Scope | Allowed Actions | Denied Behavior | Evidence |
|------------|-----------|----------------|-------------------|-----------------|-----------------|----------|
| POST `/chat` | any (anonymous) | none | none | send message | backend-enforced (unknown) | `chatApi.ts:94-105` |
| POST `/chat-research` | any (anonymous) | none | dataset_id (screen-injected, not user-scoped) | send message with dataset context | backend-enforced (unknown) | `chatApi.ts:75-86` |
| GET `/llm/gemini-models` | any (anonymous) | none | none | fetch model list | backend-enforced (unknown) | `chatApi.ts:188-265` |

**Summary:** All endpoints are treated as public from the frontend's perspective. No auth enforcement exists in the frontend layer.

---

## Phase 6: Transaction and Atomicity Contracts

### Transaction Boundary Map

This is a frontend-only feature with no database transactions. "Atomicity" here refers to in-memory state mutation consistency.

---

### REQ-CHAT-037: Submit is a non-atomic multi-step operation

**Layer:** transaction
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (state consistency during async operations)
**Risk tags:** temporal_coupling

**Source evidence:**
- `react/hooks/useChat.hooks.ts:291-324` (implementation — `submit`)
- `react/hooks/useChat.hooks.ts:245-289` (implementation — `runSubmission`)
- `integration/req-001.submit-order.integration.test.ts:8` (test — verifies ordering)

**Observed behavior:** The submit pipeline is NOT an atomic operation. It is a sequence of synchronous state mutations followed by an async API call, followed by conditional state mutations:

1. `addInputToHistory(trimmed)` — synchronous
2. `buildChatHistoryWindow(...)` — synchronous (pure computation)
3. `addMessage(userMessage)` — synchronous
4. Store `lastSubmissionRef.current` — synchronous (ref write)
5. `resetValue()` / `resetHistoryCursor()` / `clearAttachments()` — synchronous UI reset
6. `setScreenFeedback(null)` — synchronous (clear previous error)
7. `setSending(true)` — synchronous
8. `await api.sendMessage(...)` — **async boundary** (network)
9. `addMessage(assistantMessage)` — synchronous (conditional on success + mounted)
10. `onMessageReceived(payload)` — synchronous (conditional on success)
11. `setSending(false)` — synchronous (always, if mounted)

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: validated non-empty trimmed user input + current state
- Output: state mutations applied in strict temporal order
- Side effects: HTTP POST to backend (step 8)
- Invariants: steps 1-7 complete synchronously before the async boundary; steps 9-11 are mount-guarded; partial failure at step 8 leaves user message in transcript but no assistant response (acceptable UX: user sees their message + error feedback)
- Error cases: if API throws → `setScreenFeedback(error)` + `setSending(false)` (no assistant message added); if API returns null → `setScreenFeedback(INVALID_RESPONSE_FEEDBACK)` + `setSending(false)`
- Transaction boundary: no database transaction; the "transaction" is the synchronous batch of Zustand `set()` calls before the async boundary
- Concurrency: see REQ-CHAT-038
- Auth requirement: not_applicable

**Migration notes:**
- The ordering invariant (test-verified) means any reimplementation must preserve this exact mutation sequence. Steps 1-7 form a "prepare" phase; step 8 is the "execute" phase; steps 9-11 are the "finalize" phase.

---

### REQ-CHAT-038: No concurrent submission guard

**Layer:** concurrency
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (race condition surface)
**Risk tags:** concurrency_contract

**Source evidence:**
- `react/hooks/useChat.hooks.ts:291-324` (implementation — `submit` has no isSending check)
- `react/hooks/useChat.hooks.ts:248` (implementation — `setSending(true)` is not checked before proceeding)

**Observed behavior:** There is NO explicit guard preventing the user from calling `submit()` while a previous submission is in-flight (`isSending === true`). The `isSending` flag is set but never checked as a precondition in `submit()` or `runSubmission()`. If the UI allows the submit button while `isSending` is true, two concurrent submissions can execute simultaneously.

However, the UI layer (ChatBar component) typically disables the submit button when `isSending === true`, providing a UI-level guard. This is NOT a domain-level invariant — it is a view-layer convention.

**Normative contract:** unknown (no documented contract about concurrent submission prevention at the domain level)
**Rewrite decision:** preserve_actual

**Behavior classification:** intended (UI-guarded, not domain-guarded)
**Preservation decision:** preserve

**Contract:**
- Input: `submit()` called while previous request is in-flight
- Output: second submission runs in parallel — both mutate state independently
- Side effects: two API calls fire; both will try to append messages and toggle `isSending`
- Invariants: the `isSending` flag will be set to `true` by the first, then `true` again by the second (no-op); on completion, the first `finally` sets `false`, then the second `finally` also sets `false` — net effect is correct
- Error cases: race in `lastSubmissionRef.current` — the second submit overwrites the first's submission ref, so `retryLastSubmission` would retry the second (last wins)
- Transaction boundary: not_applicable
- Concurrency: **concurrent submissions are tolerated but not serialized at the domain level**; UI must prevent double-submit
- Auth requirement: not_applicable

**Open questions:**
- Is double-submission prevention intentionally delegated to the UI, or is this a gap? The absence of a domain-level guard suggests it is intentionally UI-controlled (lighter implementation, no unnecessary complexity for the single-threaded browser context).

`[CONCURRENCY CONTRACT]` — Any reimplementation must either: (a) replicate the UI-level submit button disabling when `isSending === true`, OR (b) add a domain-level guard in the submit function itself.

---

### REQ-CHAT-039: Retry reuses stored submission (last-write-wins)

**Layer:** transaction
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (retry correctness)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:311-317` (implementation — `lastSubmissionRef.current = { ... }`)
- `react/hooks/useChat.hooks.ts:326-329` (implementation — `retryLastSubmission`)

**Observed behavior:** `lastSubmissionRef` stores the most recent submission payload. Each new submit overwrites it. `retryLastSubmission()` re-runs `runSubmission(lastSubmissionRef.current)` without re-adding the user message or re-recording history. This is a last-write-wins design: if the user submits multiple times, only the most recent submission can be retried.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `retryLastSubmission()` called when `lastSubmissionRef.current !== null`
- Output: re-executes the API call with stored `{ value, model, history, attachments, datasetId }`
- Side effects: same as a fresh `runSubmission` call (setSending, API call, response handling)
- Invariants: does NOT re-add user message to transcript; does NOT modify `lastSubmissionRef`; uses the snapshot of state at original submission time (model, history, dataset) — not current state
- Error cases: `lastSubmissionRef.current === null` → no-op return
- Transaction boundary: not_applicable
- Concurrency: not_applicable (ref is synchronously read)
- Auth requirement: not_applicable

---

### REQ-CHAT-040: Model fetch is idempotent-safe (no deduplication)

**Layer:** concurrency
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (model fetch is infrequent)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:402-437` (implementation — `refetchModels`)
- `react/hooks/useChat.hooks.ts:487-505` (implementation — bootstrap effect)

**Observed behavior:** `refetchModels` does not deduplicate concurrent calls. If called twice in rapid succession, two HTTP requests fire. However, the bootstrap effect uses `initialModelBootstrapRequestedRef` to prevent the *initial* double-fetch. Manual `refetchModels()` calls have no deduplication.

The result is idempotent in outcome (last response wins via `setModelOptions`/`setSelectedModelId`) but not in execution (multiple network calls).

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `refetchModels()` called (manually or from bootstrap)
- Output: network GET + state update; concurrent calls produce redundant requests
- Side effects: HTTP GET to `/llm/gemini-models`; `setModelsLoading(true/false)` toggle
- Invariants: `isModelsLoading` is set to true at start and false at end (in finally); last response wins for model options; bootstrap ref prevents only the initial double-fire
- Error cases: any failure → fallback models applied (REQ-CHAT-005)
- Transaction boundary: not_applicable
- Concurrency: no request deduplication; idempotent outcome (last-write-wins on state)
- Auth requirement: not_applicable

---

## Phase 6b: Concurrency and In-Memory State

### Runtime Serialization Assumptions

---

### REQ-CHAT-041: Single-threaded event loop assumption

**Layer:** concurrency
**Status:** confirmed
**Confidence:** observed
**Criticality:** high
**Criticality reason:** fragile_consumer (implicit serialization assumption underlying all state logic)
**Risk tags:** concurrency_contract

**Source evidence:**
- `react/hooks/useChat.hooks.ts:291-324` (implementation — synchronous state mutations before `await`)
- `react/state/zustand/aiChatStore.ts:88` (implementation — `create()` is not thread-safe)
- `logic/chatStore.logic.ts:37-41` (implementation — `appendMessage` creates new array via spread)

**Observed behavior:** The entire feature relies on the JavaScript single-threaded event loop for implicit serialization:
1. Zustand's `set()` is synchronous and non-reentrant within a single microtask
2. The submit pipeline (steps 1-7) executes synchronously — no interleaving possible between steps
3. After the `await` boundary, only one microtask resolution runs at a time
4. React's state batching (React 18+) ensures multiple `set()` calls in one synchronous frame produce a single re-render

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: concurrent JavaScript operations (callbacks, promise resolutions)
- Output: all state mutations serialize naturally via event loop
- Side effects: not_applicable
- Invariants: no operation relies on explicit locking, mutex, or compare-and-swap; any target runtime with true parallelism (e.g., Web Workers, multi-threaded server) would need explicit synchronization for store access
- Error cases: not_applicable in single-threaded runtime
- Transaction boundary: synchronous mutations within a single event loop turn are implicitly atomic
- Concurrency: **relies on JS event loop single-threading** — `[CONCURRENCY CONTRACT]`
- Auth requirement: not_applicable

**Migration notes:**
- If porting to a multi-threaded runtime (Rust, Go, Java), all Zustand-equivalent store mutations need explicit locking or actor-based serialization.
- If using Web Workers for chat logic, the store must remain on the main thread or use message-passing serialization.

---

### REQ-CHAT-042: isMountedRef lifecycle guard pattern

**Layer:** concurrency
**Status:** confirmed
**Confidence:** tested
**Criticality:** medium
**Criticality reason:** standard CRUD (prevents React memory leak warnings and zombie state updates)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:225` (implementation — `const isMountedRef = useRef(true)`)
- `react/hooks/useChat.hooks.ts:229-234` (implementation — effect sets true/false)
- `react/hooks/useChat.hooks.ts:262,281,284` (implementation — guard checks in `runSubmission`)
- `react/hooks/useChat.hooks.ts:416` (implementation — guard in `refetchModels`)
- `integration/req-007.abort-unmount.integration.test.ts:16` (test)

**Observed behavior:** A `useRef(true)` tracks component mount status. On unmount, the cleanup function sets it to `false`. All post-await state mutations in `runSubmission` and `refetchModels` check `isMountedRef.current` before proceeding. If false, the function returns early without modifying state.

Guard points in `runSubmission`:
1. After `await api.sendMessage(...)` — line 262: `if (!isMountedRef.current) return`
2. In catch block — line 281: `if (!isMountedRef.current) return`
3. In finally block — line 284: `if (!isMountedRef.current) return`

Guard point in `refetchModels`:
1. After `await api.fetchModels({})` — line 416: `if (!isMountedRef.current) return`
2. In finally block (implicit via early return)

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: component unmount during in-flight async operation
- Output: all post-async state mutations are skipped
- Side effects: HTTP request still completes (not cancelled) but response is discarded
- Invariants: `isMountedRef` transitions: `true` on mount → `false` on unmount; once false, never becomes true again for that hook instance; guard is checked at every post-await mutation site
- Error cases: race between unmount and API response — response arrives, guard fires, no state update occurs
- Transaction boundary: not_applicable
- Concurrency: this is a lifecycle guard, not a concurrency lock
- Auth requirement: not_applicable

**Open questions:**
- The `sendChatMessage` API call is NOT aborted on unmount (no AbortController for chat requests, only for model fetch). The HTTP request completes and the response is discarded. Is this intentional? (See Pass 1 open question #2.)

---

### REQ-CHAT-043: AbortController usage limited to model fetch

**Layer:** concurrency
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (optimization, not correctness)
**Risk tags:** —

**Source evidence:**
- `api/chatApi.ts:193-195` (implementation — AbortController created for `fetchChatModels`)
- `api/chatApi.ts:206-211` (implementation — signal passed to fetch)
- `api/chatApi.ts:114-178` (implementation — `sendChatMessageToEndpoint` has no AbortController)

**Observed behavior:** `fetchChatModels` creates an AbortController with a configurable timeout (default 5000ms). If the request exceeds the timeout, `controller.abort()` fires and the fetch rejects with `AbortError`. The chat message send functions (`sendChatMessage`, `sendChatMessageDirect`) do NOT use AbortController — they have no timeout and no cancellation mechanism.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: model fetch initiated → timeout fires after `timeoutMs`
- Output: fetch aborted → `MODEL_FETCH_TIMEOUT` error result
- Side effects: `clearTimeout(timeoutId)` in finally (cleanup regardless of outcome)
- Invariants: timeout is always cleaned up; abort signal is scoped to a single request lifecycle
- Error cases: timeout → abort → error result (not thrown)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

**Migration notes:**
- Chat message requests have no timeout or cancellation. In a production environment with slow backends, this could leave `isSending === true` indefinitely (until the TCP timeout). The only recovery is the unmount guard discarding the response.

---

### REQ-CHAT-044: Navigation-away behavior during in-flight request

**Layer:** concurrency
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (UX correctness during navigation)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:229-234` (implementation — cleanup sets `isMountedRef.current = false`)
- `react/hooks/useChat.hooks.ts:262,281,284` (implementation — guard checks)

**Observed behavior:** When the user navigates away (SPA route change or hard navigation):
1. React unmounts the chat component tree
2. `useEffect` cleanup fires → `isMountedRef.current = false`
3. In-flight HTTP requests continue to completion on the network
4. When the response arrives, all guard checks fail → no state mutations
5. For chat messages: response is silently discarded (no abort)
6. For model fetch: the AbortController may or may not fire depending on timing (timeout race)

The global Zustand store (`useAiChatStore`) persists in memory across SPA navigations but `isSending` may remain `true` if the response guard fires. On re-mount (navigating back), a fresh `useChatLogic` instance is created with a new `isMountedRef = true`, but the store's `isSending` state from the previous mount is NOT cleaned up.

**Normative contract:** unknown (potential stale `isSending === true` after navigate-away)
**Rewrite decision:** human_decision_required

**Behavior classification:** unclear
**Preservation decision:** human_decision_required

**Contract:**
- Input: user navigates away while `isSending === true`
- Output: HTTP request completes but response is discarded; `isSending` remains `true` in global store
- Side effects: stale `isSending` flag in global store until next successful submit or page refresh
- Invariants: the `isMountedRef` guard prevents zombie state mutations but does NOT clean up the `isSending` flag it previously set
- Error cases: user returns to chat → `isSending === true` → submit button may appear disabled with no active request
- Transaction boundary: not_applicable
- Concurrency: this is a lifecycle edge case, not a concurrency bug
- Auth requirement: not_applicable

**Open questions:**
- Is stale `isSending === true` after navigation intentional? The LandingPage uses an isolated store (different Zustand instance) so this only affects the global `useAiChatStore`. On the AgenticResearchPage which uses the global store, navigating away mid-request could leave `isSending` stuck. This may be mitigated by the page always re-mounting the full chat surface (which would re-fetch models, triggering `setModelsLoading` but NOT resetting `isSending`).

`[NEEDS CLARIFICATION]` — Stale `isSending` after navigate-away: is this a known gap or is there an external reset mechanism not visible in this feature's code?

---

### REQ-CHAT-045: Zustand store is a global singleton (shared across all consumers)

**Layer:** concurrency
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (architectural boundary — state sharing scope)
**Risk tags:** concurrency_contract

**Source evidence:**
- `react/state/zustand/aiChatStore.ts:88` (implementation — module-level `create()`)
- `ui/screens/LandingPage/chat/state/zustand/landingChatStore.ts:31` (implementation — separate `create()` instance)
- `react/state/adapters/aiChatState.adapter.ts:8` (implementation — directly uses `useAiChatStore`)

**Observed behavior:** `useAiChatStore` is created at module level via `create<AiChatState>(...)`. This produces a single global Zustand store instance shared by ALL components that import it. Multiple screens using `useAiChatStateAdapter` share the same messages, model options, and sending state.

The LandingPage uses a SEPARATE isolated store (`useLandingChatStore`) to avoid cross-contamination. This is an explicit architectural choice: screens that need isolation create their own store; screens that share state use the global one.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: any component calling `useAiChatStore` or `useAiChatStateAdapter`
- Output: shared state reference — mutations from one consumer are visible to all others
- Side effects: not_applicable (store creation is module-level, happens once at import time)
- Invariants: global store is shared; isolated stores are per-screen; the `UseChatStatePort` abstraction allows swapping between shared/isolated without changing hook logic
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: multiple React components may dispatch actions to the same store simultaneously (within one event loop turn via batching); Zustand handles this correctly via synchronous `set()`
- Auth requirement: not_applicable

`[CONCURRENCY CONTRACT]` — Global singleton store means all consumers of `useAiChatStateAdapter` share state. Any reimplementation must preserve the "one global + optional isolated per-screen" topology.

---

### REQ-CHAT-046: Shallow equality prevents unnecessary re-renders

**Layer:** concurrency
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (performance optimization)
**Risk tags:** —

**Source evidence:**
- `react/state/adapters/aiChatState.adapter.ts:1` (implementation — `import { useShallow } from "zustand/react/shallow"`)
- `react/state/adapters/aiChatState.adapter.ts:9-20` (implementation — state selector with `useShallow`)
- `react/state/adapters/aiChatState.adapter.ts:22-34` (implementation — actions selector with `useShallow`)

**Observed behavior:** The state adapter uses `useShallow` for both state and actions selectors. This means the adapter only triggers a re-render when a selected field's shallow reference changes — not on every store update. For arrays like `messages`, this means a new array reference (from spread in `appendMessage`) correctly triggers an update, but unrelated state changes (e.g., `isSending` toggle) do not re-render components that only consume `messages`.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: Zustand store state change
- Output: re-render only if selected slice reference changed (shallow equality)
- Side effects: not_applicable (React optimization)
- Invariants: all state mutations must produce new object/array references for changed fields (immutability requirement); spread operator in pure logic functions guarantees this
- Error cases: mutating arrays in-place would break shallow comparison (not observed in codebase — all logic functions return new references)
- Transaction boundary: not_applicable
- Concurrency: not_applicable
- Auth requirement: not_applicable

---

### Temporal Coupling Inventory

---

### REQ-CHAT-047: Model fetch must complete before first submission can include model ID

**Layer:** domain
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (functional correctness — model in request)
**Risk tags:** temporal_coupling

**Source evidence:**
- `react/hooks/useChat.hooks.ts:487-505` (implementation — bootstrap effect)
- `react/hooks/useChat.hooks.ts:313` (implementation — `model: state.selectedModelId` captured at submit time)

**Observed behavior:** The bootstrap effect fires `refetchModels()` on first mount when `modelOptions.length === 0`. Until that fetch completes, `selectedModelId` is `null`. If the user submits before models load, the request payload includes `model: null`. The backend must handle `model: null` gracefully.

This is a temporal coupling: the model fetch must complete before the user's submission includes a meaningful model ID. The feature does NOT block submission on model loading — it allows `model: null` submissions.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: user submits before model fetch completes
- Output: API request includes `model: null`
- Side effects: backend receives null model (must handle gracefully)
- Invariants: submission is never blocked by model loading state; model ID is captured at submit-time from current state
- Error cases: not_applicable (feature handles null model as valid)
- Transaction boundary: not_applicable
- Concurrency: model fetch is async and independent of submission
- Auth requirement: not_applicable

`[TEMPORAL COUPLING]` — Model selection depends on prior completion of model fetch. The frontend tolerates the race (sends null), but the backend must accept `model: null`.

---

### REQ-CHAT-048: Bootstrap model fetch is once-only per mount lifecycle

**Layer:** concurrency
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (prevents fetch storm)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:471` (implementation — `const initialModelBootstrapRequestedRef = useRef(false)`)
- `react/hooks/useChat.hooks.ts:487-505` (implementation — bootstrap effect with ref guard)

**Observed behavior:** The `useChatIntegration` hook maintains `initialModelBootstrapRequestedRef`. The bootstrap effect:
1. If `modelOptions.length > 0` → resets ref to false (allows future re-bootstrap if models are cleared)
2. If `isModelsLoading` → skips (another fetch is already running)
3. If `initialModelBootstrapRequestedRef.current` → skips (already requested this lifecycle)
4. Otherwise → sets ref to true, calls `refetchModelsRef.current()`

This prevents duplicate model fetches during React strict mode double-mount and during normal effect re-runs.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: component mount or state change triggering effect re-evaluation
- Output: at most one model fetch per component lifecycle (until models are populated)
- Side effects: one HTTP GET at most
- Invariants: ref guard ensures single fetch; `isModelsLoading` guard prevents concurrent fetches; models populated → ref reset allows future refetch if models are externally cleared
- Error cases: fetch failure → fallback models applied → `modelOptions.length > 0` → ref resets → no infinite retry loop
- Transaction boundary: not_applicable
- Concurrency: ref-based guard is synchronous and race-free in single-threaded JS
- Auth requirement: not_applicable

---

### REQ-CHAT-049: refetchModelsRef indirection prevents stale closure

**Layer:** concurrency
**Status:** confirmed
**Confidence:** observed
**Criticality:** low
**Criticality reason:** internal_only (React closure correctness pattern)
**Risk tags:** —

**Source evidence:**
- `react/hooks/useChat.hooks.ts:472` (implementation — `const refetchModelsRef = useRef(actions.refetchModels)`)
- `react/hooks/useChat.hooks.ts:474-481` (implementation — effect keeps ref current)
- `react/hooks/useChat.hooks.ts:504` (implementation — `refetchModelsRef.current()`)

**Observed behavior:** The bootstrap effect calls `refetchModelsRef.current()` instead of `actions.refetchModels` directly. A separate effect updates `refetchModelsRef.current` whenever `actions.refetchModels` changes. This prevents a stale closure bug: the bootstrap effect's dependency array (`[state.isModelsLoading, state.modelOptions.length]`) does not include `actions.refetchModels`, which would cause infinite re-runs.
**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: `actions.refetchModels` reference changes (due to dependency array changes in `useChatLogic`)
- Output: ref is updated synchronously; next bootstrap effect invocation uses latest function
- Side effects: not_applicable
- Invariants: ref always points to the latest `refetchModels`; bootstrap effect never captures a stale version
- Error cases: not_applicable
- Transaction boundary: not_applicable
- Concurrency: not_applicable (React render cycle guarantees)
- Auth requirement: not_applicable

---

### REQ-CHAT-050: Proxy route is a transparent passthrough (no transformation)

**Layer:** api
**Status:** confirmed
**Confidence:** observed
**Criticality:** medium
**Criticality reason:** standard CRUD (infrastructure contract)
**Risk tags:** —

**Source evidence:**
- `app/api/ai/[...path]/route.ts:26-56` (implementation — `proxyRequest`)
- `app/api/ai/[...path]/route.ts:9-13` (implementation — hop-by-hop header stripping)

**Observed behavior:** The Next.js API route at `/api/ai/[...path]` is a transparent reverse proxy:
1. Strips hop-by-hop headers (`connection`, `content-length`, `host`)
2. Forwards all remaining headers (including cookies)
3. Forwards the request body as-is (ArrayBuffer)
4. Forwards query parameters
5. Returns upstream response status, headers, and body without modification
6. Uses `cache: "no-store"` and `redirect: "manual"`
7. Handles all HTTP methods (GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD)

No auth injection, no body transformation, no rate limiting, no caching.

**Normative contract:** matches observed
**Rewrite decision:** preserve_actual

**Behavior classification:** intended
**Preservation decision:** preserve

**Contract:**
- Input: any HTTP request to `/api/ai/*`
- Output: same request forwarded to `{AI_API_URL}/{path}{query}` and response returned as-is
- Side effects: network request to backend
- Invariants: transparent passthrough — no header injection, no body modification, no auth logic; purpose is CORS avoidance only (same-origin proxy)
- Error cases: upstream failure → status code forwarded as-is
- Transaction boundary: not_applicable
- Concurrency: Next.js handles concurrent requests via its built-in server
- Auth requirement: none at proxy level (any browser-attached cookies are forwarded implicitly)

---

## In-Memory State Inventory

| State Location | Scope | Shared? | Reset On | Purpose |
|----------------|-------|---------|----------|---------|
| `useAiChatStore` (Zustand) | Global (module-level singleton) | Yes — all pages using `useAiChatStateAdapter` | Page refresh only | Chat transcript, model options, sending state |
| `useLandingChatStore` (Zustand) | Global (separate singleton) | No — LandingPage only | Page refresh only | Isolated chat for landing page |
| `useChatUiState` (React useState) | Component instance | No — per mount | Component unmount | Input value, tooltip, attachments |
| `isMountedRef` (React useRef) | Component instance | No — per mount | Component unmount | Lifecycle guard |
| `lastSubmissionRef` (React useRef) | Component instance | No — per mount | Component unmount | Retry payload |
| `draftValueRef` (React useRef) | Component instance | No — per mount | Component unmount | History navigation draft |
| `initialModelBootstrapRequestedRef` (React useRef) | Component instance | No — per mount | Component unmount | Fetch deduplication |
| `refetchModelsRef` (React useRef) | Component instance | No — per mount | Component unmount | Stale closure prevention |

---

## Open Questions Discovered This Pass

1. **Stale `isSending` after navigate-away:** When a user navigates away from a page using the global store while a request is in-flight, `isSending` remains `true` in the global store. Is there an external mechanism (not visible in this feature) that resets it? Or is this accepted as a known edge case? Priority: important.

2. **Backend auth enforcement:** The frontend sends no auth headers. Does the backend rely on cookies forwarded by the proxy? Is the chat endpoint truly public or session-gated? Priority: nice-to-have (does not block frontend spec, but matters for end-to-end security).

3. **Chat message cancellation (from Pass 1):** `sendChatMessage` has no AbortController. For long-running AI responses, there is no way for the user to cancel. Is this intentional? Priority: nice-to-have.

---

## Amendments to Pass 1 Findings

### Amendment to REQ-CHAT-012 (Unmount guard)

REQ-CHAT-012 documented the unmount guard for `runSubmission`. This pass extends coverage: the same `isMountedRef` pattern is also used in `refetchModels` (line 416). The guard is universal across all async operations in `useChatLogic`, not just submission.

### Amendment to REQ-CHAT-002 (Submit ordering)

REQ-CHAT-002 documented the ordering invariant. This pass clarifies the atomicity boundary: steps 1-7 are synchronously atomic (single event loop turn); step 8 (await) introduces the async boundary where interleaving can occur; steps 9-11 are individually mount-guarded.

### Amendment to REQ-CHAT-010 (Model fetch timeout)

REQ-CHAT-010 documented the timeout mechanism. This pass clarifies the AbortController lifecycle: `clearTimeout` is called in a `finally`-equivalent position (after try/catch, line 263), ensuring no timer leak regardless of outcome.

---

## Risk Tags/Markers Raised This Pass

| Marker | REQ ID | Severity | Description |
|--------|--------|----------|-------------|
| `[CONCURRENCY CONTRACT]` | REQ-CHAT-038 | Important | No domain-level guard against concurrent submissions; UI must prevent |
| `[CONCURRENCY CONTRACT]` | REQ-CHAT-041 | Important | Entire feature relies on JS single-threaded event loop for state serialization |
| `[CONCURRENCY CONTRACT]` | REQ-CHAT-045 | Important | Global singleton store shared across pages using same adapter |
| `[TEMPORAL COUPLING]` | REQ-CHAT-047 | Advisory | Model selection depends on prior model fetch completion |
| `[NEEDS CLARIFICATION]` | REQ-CHAT-044 | Important | Stale `isSending` after navigate-away — gap or accepted behavior? |

---

## Entrypoints Discovered or Removed

### Discovered
- `proxyRequest` in `app/api/ai/[...path]/route.ts` — transparent proxy function (infrastructure, not feature logic)

### Removed
None.

---

## Summary

- **Requirements extracted this pass:** 20 (REQ-CHAT-031 through REQ-CHAT-050)
- **Cumulative total:** 50 (30 from Pass 1 + 20 from Pass 2)
- **Confidence breakdown (this pass):** observed: 18, tested: 2
- **Blocking markers:** 1 (`[NEEDS CLARIFICATION]` on stale isSending)
- **Important markers:** 3 (`[CONCURRENCY CONTRACT]`)
- **Advisory markers:** 1 (`[TEMPORAL COUPLING]`)

### Key Findings

1. **No database, no persistence, no auth** — the feature is entirely in-memory, ephemeral, and unauthenticated from the frontend perspective.
2. **Concurrency is managed by the JS event loop** — no explicit locks or serialization. The store is synchronous and mutations are atomic within a single event loop turn.
3. **The unmount guard (isMountedRef) is the primary lifecycle safety mechanism** — but it does NOT clean up store state that was set before the async boundary (specifically `isSending`).
4. **The proxy is transparent** — no auth injection, no transformation, CORS-avoidance only.
5. **Dual-store topology** — global singleton for shared pages + isolated store for LandingPage. The `UseChatStatePort` abstraction enables this without duplicating hook logic.
