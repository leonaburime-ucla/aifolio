# Implementation Outline: Multi-Framework Frontend Core Extraction

- Spec: SPEC-001 v1.0.0 (hash: sha256:5ac6301bec0e51192f30229309d6327185c39c3b87025927f84cee8660d9299d)
- ADR: ADR-001
- Status: PRODUCED
- Trigger result: Boundary Cross, Contract Change, Brownfield Dependency, Reverse-Spec Or Migration, Parallelization Ambiguity
- Date: 2026-06-09T18:30:00Z
- Author: Software Architect

## Trigger Decision Matrix

| Trigger | Applies? | Evidence | Source Trace |
|---|---:|---|---|
| Boundary Cross | yes | Feature crosses `web/packages/contracts`, `web/packages/frontend-core`, and `web/nextjs` — three packages with explicit dependency rules. | ADR-001 Module Boundaries |
| Contract Change | yes | 23+ new public/exported contracts defined in shared packages (types + functions). All consumed outside owning module. | ADR-001 API/Event Contract Summary, api.spec.md |
| System Wiring | no | No queues, webhooks, jobs, or service-to-service wiring. Direct TypeScript imports only. | — |
| Data And Persistence | no | No schema, table, or persistence changes. Pure type/logic extraction. | — |
| Brownfield Dependency | yes | Existing 14+ AG-UI consumers of ChartSpec. 57 production cross-feature imports. Re-exports must preserve existing paths. | ANALYSIS-aifolio-2026-06-09, behavior.spec.md §2.3 |
| Reverse-Spec Or Migration | yes | Source `logic/` and `__types__/` mapped to target package modules. Phased migration with verification gates. | behavior.spec.md §2.1, §2.2 |
| Critical Cross-Boundary Invariant | no | No cross-module transaction, double-spend, or consistency invariant. Type compatibility is enforced by TypeScript compiler. | — |
| Parallelization Ambiguity | yes | Entity extractions can parallel but logic extraction depends on entity extraction completing first. Task ordering needs structure. | ADR-001 Parallel Delivery Plan |

## Module Map

| Module/Domain | Owns | Responsibility | Public Contracts | Dependencies | Notes |
|---|---|---|---|---|---|
| `@aifolio/contracts` | Domain entity types + Zod schemas | Single source of truth for all domain type definitions and runtime validation | C-001 through C-014 | Zod only | Zero framework deps. Zero runtime deps beyond Zod. |
| `@aifolio/frontend-core` | Pure logic functions | Framework-agnostic business logic consumed by all UI apps | C-015 through C-028 | `@aifolio/contracts` only | No framework deps. No state management deps. |
| `web/nextjs` (features/ai-chat) | React-specific chat UI | React hooks, Zustand store, components that consume shared packages | N/A (framework-specific, not cross-module contracts) | `@aifolio/contracts`, `@aifolio/frontend-core` | FSD features layer. No entities layer. |

## File Map

### `@aifolio/contracts` package

| File Path | Module | Creates / Changes | Public Contracts Housed | Responsibility | Why This Separation Exists | Notes |
|---|---|---|---|---|---|---|
| `web/packages/contracts/package.json` | contracts | creates | — | Package definition, exports map, Zod dep | Package identity and dependency boundary | npm workspace member |
| `web/packages/contracts/tsconfig.json` | contracts | creates | — | TypeScript compilation and declaration output | Declaration files for consumers | Emits `.d.ts` |
| `web/packages/contracts/src/entities/chart/index.ts` | contracts | creates | C-001, C-002, C-003 | ChartSpec type, ChartSpecSchema, ChartActionsPort | Chart is the most-imported entity (14+ consumers). Must extract first per behavior.spec.md §2.1 tier 1. | Source: `features/charts/contracts/chart.types.ts` |
| `web/packages/contracts/src/entities/chat/index.ts` | contracts | creates | C-004 through C-012 | ChatMessage, ChatModelOption, ChatAssistantPayload, ChatHistoryMessage, ChatAttachment, ChatHistoryDirection, ChatState, ChatStateActions, ChatCoreStateActions, ChatStatePort, ChatChartActionsPort, ModelSelectionResult, FallbackSelectionInput, FetchedSelectionInput, ScreenFeedback | All chat domain types in one public API | Source: `features/ai-chat/__types__/chat.types.ts`, `uiFeedback.types.ts`, `logic/modelSelection.types.ts`, `logic/chatSubmission.types.ts`, `logic/chatStore.types.ts`, `logic/chatComposition.types.ts` |
| `web/packages/contracts/src/entities/chat/api.types.ts` | contracts | creates | C-013 | ChatApiResponse, SendChatMessageInput, FetchChatModelsResult, ChatApiError and related API types | Chat API contract shapes (not the fetch implementation) | Source: `features/ai-chat/__types__/api.types.ts` |
| `web/packages/contracts/src/index.ts` | contracts | creates | — | Package root barrel | Single entrypoint re-exporting entity subpaths | Subpath exports in package.json preferred |

### `@aifolio/frontend-core` package

| File Path | Module | Creates / Changes | Public Contracts Housed | Responsibility | Why This Separation Exists | Notes |
|---|---|---|---|---|---|---|
| `web/packages/frontend-core/package.json` | frontend-core | creates | — | Package definition, dep on `@aifolio/contracts` | Package identity and dependency boundary | npm workspace member |
| `web/packages/frontend-core/tsconfig.json` | frontend-core | creates | — | TypeScript compilation and declaration output | Declaration files for consumers | Emits `.d.ts` |
| `web/packages/frontend-core/src/features/model-selection/index.ts` | frontend-core | creates | C-015, C-016, C-017 | resolveFallbackModelSelection, resolveFetchedModelSelection, FALLBACK_CHAT_MODELS | Model selection is a pure computation used by all chat implementations | Source: `features/ai-chat/logic/modelSelection.logic.ts` |
| `web/packages/frontend-core/src/features/chat-submission/index.ts` | frontend-core | creates | C-018, C-019, C-020, C-021, C-022 | normalizeSubmissionValue, buildChatHistoryWindow, createUserChatMessage, createAssistantChatMessage, shouldRestoreDraftValue | Chat submission pipeline — pure input normalization and message construction | Source: `features/ai-chat/logic/chatSubmission.logic.ts` |
| `web/packages/frontend-core/src/features/chat-normalization/index.ts` | frontend-core | creates | C-023, C-024, C-025, C-026 | normalizeChatApiResult, normalizeTextResult, parseJsonPayload, createModelFetchErrorResult | API response normalization — pure parsing of backend payloads into typed structures | Source: `features/ai-chat/logic/chatApiNormalization.logic.ts` |
| `web/packages/frontend-core/src/features/chat-composition/index.ts` | frontend-core | creates | C-027, C-028, C-029 | mapChatStateWithDataset, createOnMessageReceived, composeChatStateActions | State composition helpers — pure functions that wire state/actions together | Source: `features/ai-chat/logic/chatComposition.logic.ts` |
| `web/packages/frontend-core/src/features/chat-store/index.ts` | frontend-core | creates | C-030, C-031, C-032, C-033 | createInitialChatStoreCoreState, appendMessage, appendInputHistory, resolveHistoryCursor | Store mutation logic — pure state transitions usable by any state manager | Source: `features/ai-chat/logic/chatStore.logic.ts` |
| `web/packages/frontend-core/src/features/chat-orchestrator/index.ts` | frontend-core | creates | C-034, C-035 | createChatApiDeps, createChatDeps | Dependency assembly — pure constructors that compose injection bundles | Source: `features/ai-chat/logic/chatOrchestrator.logic.ts` |
| `web/packages/frontend-core/src/index.ts` | frontend-core | creates | — | Package root barrel | Re-exports feature subpaths | Subpath exports in package.json preferred |

### `web/nextjs` re-exports (temporary migration shims)

| File Path | Module | Creates / Changes | Public Contracts Housed | Responsibility | Why This Separation Exists | Notes |
|---|---|---|---|---|---|---|
| `web/nextjs/src/features/charts/contracts/chart.types.ts` | nextjs | changes | — (re-export only) | Re-exports ChartSpec + ChartActionsPort from `@aifolio/contracts` | Preserves existing import paths for 14+ consumers during migration | Removed in Phase 4 |
| `web/nextjs/src/features/ai-chat/__types__/chat.types.ts` | nextjs | changes | — (re-export only) | Re-exports all chat entity types from `@aifolio/contracts` | Preserves existing import paths during migration | Removed in Phase 4 |
| `web/nextjs/src/features/ai-chat/__types__/api.types.ts` | nextjs | changes | — (re-export only) | Re-exports API contract types from `@aifolio/contracts` | Preserves existing import paths during migration | Removed in Phase 4 |
| `web/nextjs/src/features/ai-chat/__types__/uiFeedback.types.ts` | nextjs | changes | — (re-export only) | Re-exports ScreenFeedback from `@aifolio/contracts` | Preserves existing import paths during migration | Removed in Phase 4 |
| `web/nextjs/src/features/ai-chat/logic/*.logic.ts` | nextjs | changes | — (re-export only) | Re-exports logic functions from `@aifolio/frontend-core` | Preserves existing import paths during migration | Removed in Phase 4 |

## Contract Map

### Contracts Package — Entity Types

| Contract ID | File | Owner Module | Kind | Why Needed | Job | Inputs | Outputs | Validation | Errors | Effect Boundary | Complexity | Aggregate-Risk | Spec/ADR Trace | Test Seam |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-001 ChartSpec | `contracts/src/entities/chart/index.ts` | contracts | exported type + Zod schema | 14+ consumers across ag-ui, chat, recharts features | Define the shape of a chart data specification | N/A (type) | N/A (type) | Zod schema validates all required fields: id, title, type (15-member union), xKey, yKeys, data | Schema parse error with field identification | pure (no effects) | O(n) field validation | N/A | AC-01, REQ-01 | `ChartSpecSchema.parse(input)` assertion |
| C-002 ChartActionsPort | `contracts/src/entities/chart/index.ts` | contracts | exported type | Cross-feature chart action injection contract | Define the narrow interface for chart spec mutation | N/A (type) | N/A (type) | N/A (interface) | N/A | pure | trivial | N/A | REQ-01 | Type compatibility check |
| C-003 ChartSpecSchema | `contracts/src/entities/chart/index.ts` | contracts | exported Zod schema | Runtime validation of chart payloads from backend | Validate unknown data against ChartSpec shape at runtime | `unknown` | `ChartSpec` | All required fields present, type union valid, data array non-empty when applicable | ZodError with path | pure | O(n) on data array length | N/A | AC-06, REQ-03 | `expect(() => schema.parse(invalid)).toThrow()` |
| C-004 ChatMessage | `contracts/src/entities/chat/index.ts` | contracts | exported type + Zod schema | Core domain noun for all chat features | Define single chat message shape | N/A (type) | N/A (type) | id: string, role: "user"\|"assistant", content: string, createdAt: number, chartSpec?: ChartSpec\|null | Schema parse error | pure | trivial | N/A | AC-02, REQ-01 | Schema parse assertion |
| C-005 ChatModelOption | `contracts/src/entities/chat/index.ts` | contracts | exported type | Model selection UI and logic | Define selectable AI model shape | N/A | N/A | id: string, label: string | N/A | pure | trivial | N/A | REQ-01 | Type check |
| C-006 ChatAssistantPayload | `contracts/src/entities/chat/index.ts` | contracts | exported type | Backend response normalization target | Define structured assistant response shape | N/A | N/A | message: string, chartSpec: ChartSpec\|ChartSpec[]\|null | N/A | pure | trivial | N/A | REQ-01, AC-03 | Type check |
| C-007 ChatState | `contracts/src/entities/chat/index.ts` | contracts | exported type | State portability contract | Define observable state shape all frameworks must implement | N/A | N/A | All fields with defined initial values per INV-06 | N/A | pure | trivial | N/A | AC-10, REQ-07 | Initial value assertions |
| C-008 ChatStateActions | `contracts/src/entities/chat/index.ts` | contracts | exported type | State mutation contract | Define mutation interface all frameworks must implement | N/A | N/A | All methods match spec function signatures | N/A | pure (type only) | trivial | N/A | REQ-07 | Type compatibility |
| C-009 ModelSelectionResult | `contracts/src/entities/chat/index.ts` | contracts | exported type | Return type for model selection logic | Typed result of model resolution | N/A | N/A | modelOptions: ChatModelOption[], selectedModelId: string\|null | N/A | pure | trivial | N/A | REQ-01 | Type check |
| C-010 ScreenFeedback | `contracts/src/entities/chat/index.ts` | contracts | exported type | Persistent inline feedback contract | Define structured feedback shape for UI rendering | N/A | N/A | kind: union, code: string, message: string | N/A | pure | trivial | N/A | REQ-10 | Type check |
| C-011 ChatHistoryMessage | `contracts/src/entities/chat/index.ts` | contracts | exported type | API payload shape for history window | Define message shape sent to backend | N/A | N/A | role, content, attachments? | N/A | pure | trivial | N/A | REQ-01 | Type check |
| C-012 FallbackSelectionInput / FetchedSelectionInput | `contracts/src/entities/chat/index.ts` | contracts | exported types | Input types for model selection logic | Typed inputs for pure functions | N/A | N/A | Field presence per type definition | N/A | pure | trivial | N/A | REQ-01 | Type check |
| C-013 ChatApiResponse / SendChatMessageInput / FetchChatModelsResult | `contracts/src/entities/chat/api.types.ts` | contracts | exported types | API boundary contract shapes | Define payload shapes between frontend and backend | N/A | N/A | Discriminated union on status field | N/A | pure | trivial | N/A | REQ-01, AC-09 | Type check + discriminant assertion |
| C-014 ChatApiError | `contracts/src/entities/chat/api.types.ts` | contracts | exported type | Structured error from chat API | Define error shape for timeout/failure cases | N/A | N/A | code: "MODEL_FETCH_TIMEOUT"\|"MODEL_FETCH_FAILED", retryable: boolean | N/A | pure | trivial | N/A | REQ-09, REQ-10 | Type check |

### Frontend-Core Package — Logic Functions

| Contract ID | File | Owner Module | Kind | Why Needed | Job | Inputs | Outputs | Validation | Errors | Effect Boundary | Complexity | Aggregate-Risk | Spec/ADR Trace | Test Seam |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-015 resolveFallbackModelSelection | `frontend-core/src/features/model-selection/index.ts` | frontend-core | exported function | All chat UIs need deterministic model fallback | Resolve model list and selected model when fetch fails | `input: FallbackSelectionInput`, `options?: FallbackSelectionOptions` | `ModelSelectionResult` | input.selectedModelId is string\|null | Never throws — returns fallback defaults | pure (no effects) | O(1) | N/A | AC-04, REQ-04 | `expect(fn({selectedModelId: null})).toEqual({...})` |
| C-016 resolveFetchedModelSelection | `frontend-core/src/features/model-selection/index.ts` | frontend-core | exported function | All chat UIs need deterministic model selection from API | Resolve selected model from fetched results with priority chain | `input: FetchedSelectionInput` | `ModelSelectionResult` | input.result.models must be array | Never throws — returns null selectedModelId when list empty | pure | O(1) | N/A | AC-05, REQ-04 | `expect(fn({...})).toEqual({...})` |
| C-017 FALLBACK_CHAT_MODELS | `frontend-core/src/features/model-selection/index.ts` | frontend-core | exported constant | Default model list when API unavailable | Provide stable fallback model options | N/A | `ChatModelOption[]` | Frozen array, order stable per DR-005 | N/A | pure | trivial | N/A | REQ-04, DR-005 | `expect(FALLBACK_CHAT_MODELS[0].id).toBe("gemini-3-flash-preview")` |
| C-018 normalizeSubmissionValue | `frontend-core/src/features/chat-submission/index.ts` | frontend-core | exported function | All chat UIs need input trim + empty guard | Trim whitespace, return null for empty input | `input: NormalizeSubmissionInput` | `string \| null` | input.value must be string | Never throws | pure | O(n) string length | N/A | AC-01, DR-001, REQ-01 | `expect(fn({value: "  "})).toBeNull()` |
| C-019 buildChatHistoryWindow | `frontend-core/src/features/chat-submission/index.ts` | frontend-core | exported function | All chat UIs need bounded history for API payload | Build window of recent messages for backend | `input: BuildChatHistoryWindowInput`, `options?: BuildChatHistoryWindowOptions` | `ChatHistoryMessage[]` | windowSize defaults to 10 | Never throws — empty array for empty input | pure | O(n) messages length | N/A | DR-002, REQ-01 | `expect(fn({messages: [...10], userContent: "x"}).length).toBe(10)` |
| C-020 createUserChatMessage | `frontend-core/src/features/chat-submission/index.ts` | frontend-core | exported function | All chat UIs construct user messages identically | Create typed user message | `input: CreateChatMessageInput` | `ChatMessage` | id, content, createdAt required | Never throws | pure | O(1) | N/A | REQ-01, AC-01 | `expect(fn({...}).role).toBe("user")` |
| C-021 createAssistantChatMessage | `frontend-core/src/features/chat-submission/index.ts` | frontend-core | exported function | All chat UIs construct assistant messages identically | Create typed assistant message with null chartSpec | `input: CreateChatMessageInput` | `ChatMessage` | id, content, createdAt required | Never throws | pure | O(1) | N/A | REQ-01 | `expect(fn({...}).role).toBe("assistant")` |
| C-022 shouldRestoreDraftValue | `frontend-core/src/features/chat-submission/index.ts` | frontend-core | exported function | History cursor navigation needs draft restore decision | Determine if draft input should be restored on cursor move | `input: ShouldRestoreDraftValueInput` | `boolean` | direction, historyCursor, nextValue required | Never throws | pure | O(1) | N/A | DR-003 | `expect(fn({direction:"down", historyCursor:1, nextValue:""})).toBe(true)` |
| C-023 normalizeChatApiResult | `frontend-core/src/features/chat-normalization/index.ts` | frontend-core | exported function | All chat UIs normalize backend responses identically | Convert raw API result into structured ChatAssistantPayload | `result: ChatApiResponse["result"]` | `ChatAssistantPayload \| null` | Result may be string, object, array, or nullish | Returns null for unusable payloads (never throws) per DR-004 | pure | O(n) on content part count | N/A | DR-004, REQ-01 | `expect(fn(undefined)).toBeNull()` |
| C-024 normalizeTextResult | `frontend-core/src/features/chat-normalization/index.ts` | frontend-core | exported function | Text responses may contain embedded JSON | Parse embedded JSON or wrap as plain message | `text: string` | `ChatAssistantPayload` | text must be string | Never throws | pure | O(n) string length | N/A | REQ-01 | `expect(fn("hello").message).toBe("hello")` |
| C-025 parseJsonPayload | `frontend-core/src/features/chat-normalization/index.ts` | frontend-core | exported function | Backend may return JSON-in-string responses | Extract structured payload from JSON string | `raw: string` | `ChatAssistantPayload \| null` | Checks for `{` prefix and `}` suffix | Returns null on parse failure (never throws) per DR-004 | pure | O(n) string length | N/A | DR-004 | `expect(fn("not json")).toBeNull()` |
| C-026 createModelFetchErrorResult | `frontend-core/src/features/chat-normalization/index.ts` | frontend-core | exported function | Standardize model fetch error construction | Build typed error result for model fetch failures | `input: {code, retryable, message}` | `FetchChatModelsErrorResult` | code must be valid ChatApiError code | Never throws | pure | O(1) | N/A | REQ-09, REQ-10 | `expect(fn({...}).status).toBe("error")` |
| C-027 mapChatStateWithDataset | `frontend-core/src/features/chat-composition/index.ts` | frontend-core | exported function | Screen-level dataset injection into chat state | Merge activeDatasetId into state without mutating | `input: MapChatStateWithDatasetInput` | `ChatState` | state and activeDatasetId required | Never throws | pure | O(1) spread | N/A | REQ-06, AC-06 | `expect(fn({state, activeDatasetId: "x"}).activeDatasetId).toBe("x")` |
| C-028 createOnMessageReceived | `frontend-core/src/features/chat-composition/index.ts` | frontend-core | exported function | Chart spec fan-out from assistant payload | Create handler that routes chartSpec to addChartSpec | `input: CreateOnMessageReceivedInput` | `(payload: ChatAssistantPayload) => void` | input.addChartSpec must be function | Never throws internally | side effect: calls addChartSpec (injected) | O(n) on chartSpec array length | N/A | AC-03, REQ-03 | `verify addChartSpec called per spec in order` |
| C-029 composeChatStateActions | `frontend-core/src/features/chat-composition/index.ts` | frontend-core | exported function | Wire core actions + chart action into full ChatStateActions | Compose complete action bundle from parts | `input: ComposeChatStateActionsInput` | `ChatStateActions` | coreActions and addChartSpec required | Never throws | pure (returns new object) | O(1) | N/A | REQ-01 | `expect(fn({...}).addChartSpec).toBeDefined()` |
| C-030 createInitialChatStoreCoreState | `frontend-core/src/features/chat-store/index.ts` | frontend-core | exported function | All state managers need identical initial state | Produce default ChatStoreCoreState | `input: Record<string, never>` | `ChatStoreCoreState` | Empty object input for signature consistency | Never throws | pure | O(1) | N/A | AC-10, INV-06 | `expect(fn({}).isSending).toBe(false)` |
| C-031 appendMessage | `frontend-core/src/features/chat-store/index.ts` | frontend-core | exported function | Immutable message append for any state manager | Return new array with message appended | `input: AppendMessageInput` | `ChatMessage[]` | messages array and message required | Never throws | pure | O(n) array copy | N/A | REQ-01 | `expect(fn({messages:[], message:m}).length).toBe(1)` |
| C-032 appendInputHistory | `frontend-core/src/features/chat-store/index.ts` | frontend-core | exported function | Immutable input history append + cursor reset | Return new history array and null cursor | `input: AppendInputHistoryInput` | `Pick<ChatStoreCoreState, "inputHistory"\|"historyCursor">` | inputHistory array and value required | Never throws | pure | O(n) array copy | N/A | REQ-01 | `expect(fn({...}).historyCursor).toBeNull()` |
| C-033 resolveHistoryCursor | `frontend-core/src/features/chat-store/index.ts` | frontend-core | exported function | History navigation for any state manager | Compute next cursor position and resolved value | `input: ResolveHistoryCursorInput` | `HistoryCursorResult` | inputHistory, historyCursor, direction required | Never throws — returns bounds-safe result per DR-003 | pure | O(1) | N/A | DR-003 | `expect(fn({inputHistory:[], ...}).nextCursor).toBeNull()` |
| C-034 createChatApiDeps | `frontend-core/src/features/chat-orchestrator/index.ts` | frontend-core | exported function | Assemble API dependency bundle for injection | Construct ChatApiDeps from individual functions | `input: CreateChatApiDepsInput` | `ChatApiDeps` | sendMessage and fetchModels required | Never throws | pure | O(1) | N/A | REQ-01 | `expect(fn({...}).sendMessage).toBeDefined()` |
| C-035 createChatDeps | `frontend-core/src/features/chat-orchestrator/index.ts` | frontend-core | exported function | Assemble full chat dependency bundle | Construct ChatDeps from state, actions, api, logic | `input: CreateChatDepsInput` | `ChatDeps` | All four fields required | Never throws | pure | O(1) | N/A | REQ-01 | `expect(fn({...}).logic).toBeDefined()` |

## Wiring Map

| Flow ID | Source | Transport/Call Type | Target | Payload/Contract | Ordering/Retry/Idempotency | Failure Handling | Trace |
|---|---|---|---|---|---|---|---|
| W-001 | `web/nextjs` features | TypeScript import | `@aifolio/contracts` | Type imports + schema imports | Build-time resolution via workspace | TypeScript compile error if package missing | AC-08, EC-02 |
| W-002 | `web/nextjs` features | TypeScript import | `@aifolio/frontend-core` | Function imports | Build-time resolution via workspace; depends on W-001 completing first | TypeScript compile error if package missing | AC-07, AC-08 |
| W-003 | `@aifolio/frontend-core` | TypeScript import | `@aifolio/contracts` | Type imports only | Build-order: contracts builds before frontend-core | TypeScript compile error | PBR-02, PBR-04 |
| W-004 | Re-export shims in `web/nextjs` | TypeScript re-export | `@aifolio/contracts` / `@aifolio/frontend-core` | Full public API surface forwarded | Must re-export exact same surface per INV-05 | Compile error if surface mismatch | INV-05, AC-11 |

## Data And Side-Effect Boundaries

| Boundary | Owner | Reads | Writes | Side Effects | Consistency / Transaction Rule | Migration / Dual-Write Path |
|---|---|---|---|---|---|---|
| Chat Zustand store | `web/nextjs` features/ai-chat | React hooks in nextjs | React hooks in nextjs via ChatStateActions | None (in-memory only) | Single-threaded JS — no transaction needed | Store implementation stays in nextjs. Only mutation logic (appendMessage, resolveHistoryCursor) extracted as pure functions. |

## Observability And Operational Expectations

N/A — This is a structural refactoring of shared packages. No production backend paths, external I/O, async jobs, or alerting surfaces are introduced or changed. Existing Next.js app observability is preserved unchanged.

## Critical Invariants

| Invariant ID | Scope | Rule | Reason | Enforcement Surface | Test Expectation | Trace |
|---|---|---|---|---|---|---|
| INV-01 | `@aifolio/contracts` | Must never import from react, vue, svelte, @angular/*, next, nuxt | Framework-agnostic guarantee | ESLint no-restricted-imports in package | Lint pass in CI | SPEC-001 INV-01 |
| INV-02 | `@aifolio/frontend-core` | Must never import from framework or state management packages | Framework-agnostic guarantee | ESLint no-restricted-imports in package | Lint pass in CI | SPEC-001 INV-02 |
| INV-03 | `@aifolio/contracts` | Every exported type has a corresponding Zod schema | Runtime validation always available | Contract test for every type | `expect(Schema.parse(validData)).toBeDefined()` per type | SPEC-001 INV-03 |
| INV-04 | `@aifolio/frontend-core` | All exported functions are deterministic (same input → same output) | Cross-framework behavioral parity | Unit tests with fixed inputs | Pure function assertion | SPEC-001 INV-07 |
| INV-05 | Re-export shims | Must re-export exact same public API surface as before extraction | Non-breaking migration | TypeScript compiler (type mismatch = error) | Existing tests pass unchanged | SPEC-001 INV-05, AC-08 |
| INV-06 | `createInitialChatStoreCoreState` | `[internal-invariant]` Every field in ChatStoreCoreState must have a defined initial value — no undefined | State portability requires predictable initialization | Unit test asserting every field !== undefined | `Object.values(fn({})).forEach(v => expect(v).not.toBeUndefined())` | SPEC-001 INV-06 |

## Brownfield / Migration Mapping

| Source Behavior / Contract | Target Module / Contract | Preserve / Change | Characterization Evidence | Migration Safety Note |
|---|---|---|---|---|
| `features/charts/contracts/chart.types.ts` → ChartSpec, ChartActionsPort | `@aifolio/contracts/entities/chart` C-001, C-002 | preserve exactly | 14+ import sites in AG-UI; existing Next.js tests | Re-export at original path during migration. Remove only after all 14+ consumers updated. |
| `features/ai-chat/__types__/chat.types.ts` → ChatMessage, ChatModelOption, ChatState, ChatStateActions, etc. | `@aifolio/contracts/entities/chat` C-004 through C-012 | preserve exactly | ai-chat feature tests; React hook consumers | Re-export at original path. API surface frozen per INV-05. |
| `features/ai-chat/__types__/api.types.ts` → ChatApiResponse, SendChatMessageInput, etc. | `@aifolio/contracts/entities/chat/api.types` C-013, C-014 | preserve exactly | chatApi.adapter.ts consumers | Re-export at original path. |
| `features/ai-chat/__types__/uiFeedback.types.ts` → ScreenFeedback | `@aifolio/contracts/entities/chat` C-010 | preserve exactly | UIFeedback.tsx, store consumers | Re-export at original path. |
| `features/ai-chat/logic/modelSelection.logic.ts` → resolveFallbackModelSelection, resolveFetchedModelSelection, FALLBACK_CHAT_MODELS | `@aifolio/frontend-core/features/model-selection` C-015, C-016, C-017 | preserve exactly | ai-chat spec AC-04, AC-05 | Re-export at original path. |
| `features/ai-chat/logic/chatSubmission.logic.ts` → normalizeSubmissionValue, buildChatHistoryWindow, createUserChatMessage, createAssistantChatMessage, shouldRestoreDraftValue | `@aifolio/frontend-core/features/chat-submission` C-018 through C-022 | preserve exactly | ai-chat spec AC-01, AC-02, DR-001, DR-002 | Re-export at original path. |
| `features/ai-chat/logic/chatApiNormalization.logic.ts` → normalizeChatApiResult, normalizeTextResult, parseJsonPayload, createModelFetchErrorResult | `@aifolio/frontend-core/features/chat-normalization` C-023 through C-026 | preserve exactly | ai-chat spec DR-004 | Re-export at original path. |
| `features/ai-chat/logic/chatComposition.logic.ts` → mapChatStateWithDataset, createOnMessageReceived, composeChatStateActions | `@aifolio/frontend-core/features/chat-composition` C-027 through C-029 | preserve exactly | ai-chat spec AC-03, AC-06 | Re-export at original path. |
| `features/ai-chat/logic/chatStore.logic.ts` → createInitialChatStoreCoreState, appendMessage, appendInputHistory, resolveHistoryCursor | `@aifolio/frontend-core/features/chat-store` C-030 through C-033 | preserve exactly | ai-chat spec AC-10, DR-003 | Re-export at original path. |
| `features/ai-chat/logic/chatOrchestrator.logic.ts` → createChatApiDeps, createChatDeps | `@aifolio/frontend-core/features/chat-orchestrator` C-034, C-035 | preserve exactly | ai-chat orchestrator spec | Re-export at original path. |

## Test Expectations

- **Contract tests (P1):** Every Zod schema in `@aifolio/contracts` — validates known-good data, rejects known-bad data, identifies error fields. Covers C-001 through C-014.
- **Unit tests (P1):** Every exported function in `@aifolio/frontend-core` — deterministic input/output assertions. Covers C-015 through C-035. Run in Node without any framework installed (AC-07, AC-13).
- **Integration tests (P1):** After re-exports wired, existing Next.js test suite passes unchanged (AC-08).
- **Invariant tests (P1):** INV-01 and INV-02 enforced by lint rules in CI. INV-03 by schema-per-type test. INV-06 by initial state field presence test.
- **Characterization tests (P1):** Not needed for separate Phase 0 — existing Next.js tests serve as characterization tests for current behavior. If any feature logic lacks tests before extraction, add them first (per architecture-migration skill Phase 0 rule).
- **Explicitly N/A:** E2E/browser tests — extraction is structural, no UI behavior changes. Performance tests — no runtime path changes.

## Downstream Handoff Notes

- **Coordinator task-generation constraints:** Phase 1 (contracts) must complete before Phase 2 (frontend-core). Within Phase 1, all entity extractions can parallel. Within Phase 2, all feature logic extractions can parallel (they depend on Phase 1 output, not each other). Phase 3 (Next.js import migration) is sequential after Phase 2. Phase 4 (re-export removal) sequential after Phase 3. Phase 5 (boundary enforcement) sequential after Phase 4.
- **TDD focus:** Contract schema tests for all entities (C-001 through C-014). Pure function tests for all logic (C-015 through C-035). Initial state invariant test (INV-06). Framework-independence test (AC-07, AC-13). All before Programmer starts.
- **Programmer architecture audit focus:** Verify re-exports match exact public API surface (INV-05). Verify no framework imports in shared packages (INV-01, INV-02). Verify `package.json` exports map matches file map.
- **Open risks or ambiguities:** OQ-02 (AG-UI extraction scope) does not block ai-chat extraction — ag-ui entities are a later phase. OQ-01 (test runner) defaulted to Vitest per behavior.spec.md §3.
