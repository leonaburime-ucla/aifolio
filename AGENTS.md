# Agent Specification (SDD + Orc-BASH + DDD)

This document translates the latest AI Dev Shop spec kit in `AI-Dev-Shop-speckit/` into an Orc-BASH-aligned, Domain-Driven plan for every AI agent in AIfolio. We combine **Spec-Driven Development (SDD)**, the **Orc-BASH** paradigm (Orchestration, Business Logic, API, State, Hooks), and Bounded Contexts so that each feature ships with deterministic contracts, maximal hook reusability, and predictable error bubbling all the way to the UI. For the Python agent workspace (FastAPI + LangGraph + ML tooling), see `ai-python/AGENT.md`. Reference: [The Orc-BASH Pattern](https://medium.com/@leonaburime/the-orc-bash-pattern-orchestrated-architecture-for-maximum-reusability-5d6b4734c9f6).

## 1. Working Principles

1. **Start with specs.** Every route, workflow, and feature listed in `AI-Dev-Shop-speckit/` maps to an explicit agent capability and data contract before any code exists.
2. **Model layers per Orc-BASH.** Orchestrators know about Business Logic, APIs, State, and Hooks; those four layers never depend on each other—only on shared types.
3. **Codify ubiquitous language.** Define TypeScript types (`types/agent.ts`) for inputs, outputs, events, and error classes so each bounded context (Crypto, Real Estate, etc.) speaks the same language across runtimes.
4. **Instrument everything.** All orchestrators emit diagnostics for LangSmith, Sentry, and PostHog so we can trace failures through layers.

## 2. Orc-BASH for Agents

```
UI ↔ Hooks ↔ State ↔ (Orchestrator → Business Logic → APIs)
```

### 2.1 Orchestration Layer (O)
- **Scope:** `/api/chat`, LangGraph routers, domain orchestrators, and UI-side conversation orchestrators.
- **Responsibilities:**
  - Normalize incoming chat events (`ConversationOrchestrator`), inject auth/profile context, and stream requests to the backend.
  - `IntentRouterOrchestrator` (LangGraph entry) classifies intents, validates prerequisites, and chooses a domain orchestrator via conditional edges.
  - Domain orchestrators (e.g., `CryptoOrchestrator`, `RealEstateOrchestrator`) chain preparation → tool fan-out → synthesis → response → error nodes.
  - Assemble `AgentEnvelope` objects, populate diagnostics/errors, and decide navigation side effects for the UI.
- **Artifacts:** `logic/orchestrators/*.ts`, LangGraph definitions in `logic/graphs/*`, Next.js API handlers in `nextjs/src/app/api/chat`.

### 2.2 Business Logic Layer (B)
- **Scope:** Pure services/functions that implement calculations, scoring, validation, and schema shaping for each module.
- **Characteristics:**
  - Lives under `logic/domain/*` with zero dependencies on API/state/hook files.
  - Provides deterministic methods (`calculatePortfolioDelta`, `scoreNeighborhood`, etc.) reused by orchestrators and tests.
  - Implements fallback rules when tool data is missing and labels error severity.
- **Testing:** SHARP tests per service to guarantee deterministic math and classification thresholds.

### 2.3 API Layer (A)
- **Scope:** Tool adapters and repository helpers for CoinGecko, Supabase, pgvector, News API, etc.
- **Guidelines:**
  - Use free functions or light adapters (`fetchCoinPrices`, `loadUserPortfolio`) under `logic/api/*` so they can be mocked easily.
  - All API functions return typed results, never mutate state, and throw `ToolError` objects with metadata (`toolId`, latency, cache key).
  - Provide caching + retry wrappers via Upstash Redis and HTTP clients.

### 2.4 State Layer (S)
- **Scope:** Zustand stores and server-side state slices powering conversation, feature data, and global error UI.
- **Rules:**
  - State files live under `state-management/*`. They import shared types only; never import hooks or APIs.
  - Stores expose selectors for view models (`useCryptoStore`, `useErrorStore`) consumed by hooks/components.

### 2.5 Hooks Layer (H)
- **Scope:** Reusable React hooks in `ui-hooks/*` that glue Orchestrators to the UI.
- **Pattern:**
  - Hooks receive orchestrator functions/state through dependency injection.
  - They map `AgentEnvelope` streams into component-ready slices (loading flags, error arrays, navigation commands).
  - Hooks subscribe to Zustand selectors but never instantiate stores themselves.

### 2.6 Domain-Driven Mapping (DDD)
- **Bounded Contexts:** Each domain agent (crypto, real estate, etc.) is its own context with vocabulary mirrored across orchestrators, LangGraph nodes, stores, and UI hooks.
- **Aggregates:** Business Logic implements aggregate operations (e.g., `CryptoPortfolio`, `NeighborhoodProfile`) and never leaks persistence concerns uphill.
- **Repositories:** API layer adapters behave like repositories per context; orchestrators orchestrate them but do not mutate their state directly.
- **Context Maps:** Shared types plus `logic/agents/specs.ts` act as the context map so new agents can register their schema, allowed tools, and navigation side-effects.

## 3. Agent Contracts & Shared Types

### 3.1 Domain Agent Registration
```ts
interface DomainAgentSpec {
  id: 'crypto' | 'real-estate' | 'ai-workflows' | string;
  supportedIntents: IntentType[];
  outputSchema: z.ZodTypeAny;      // Shared with UI renderers
  requiredTools: ToolID[];         // Declared for observability & gating
  cacheKeys: string[];             // Used by API layer + state hydration
}
```
Keep specs in `logic/agents/specs.ts`. Orchestrators import these specs to wire LangGraph edges; hooks import them to understand view models and placeholder states.

### 3.2 Agent Envelope
```ts
interface AgentEnvelope {
  message: string;                     // User-facing response
  data: Record<string, unknown>;       // Module view model (typed per spec)
  action?: 'navigate' | 'update' | 'none';
  route?: string;                      // Provided when action === 'navigate'
  diagnostics: AgentDiagnostics;       // traceId, tool costs, latency, cache hits
  error?: AgentError;                  // Populated on partial or total failure
}
```
`AgentEnvelope` is the only thing Hooks see. All lower layers must conform before UI code consumes anything.

### 3.3 Error Object
```ts
interface AgentError {
  level: 'tool' | 'business' | 'orchestrator' | 'state' | 'ui';
  severity: 'info' | 'warning' | 'error';
  code: string;                      // e.g., CRYPTO_PRICE_TIMEOUT
  userMessage: string;               // Safe copy for UI
  developerMessage?: string;         // Logged only (Sentry/LangSmith)
  retryable: boolean;
  causes?: string[];                 // bubble-up stack of prior codes
  context?: Record<string, unknown>; // sanitized metadata
}
```
- API layer throws `level='tool'` errors.
- Business logic may wrap tool errors, append `causes`, and downgrade severity when partial data exists.
- Orchestrators add `traceId`, enforce `action='none'` when severity is `error`, and forward to hooks.
- State/Hook layers may raise `level='state'|'ui'` if schema validations fail before rendering.

## 4. Error Propagation Across Orc-BASH Layers

| Layer | Examples | Handling | Bubble Rule |
| --- | --- | --- | --- |
| **API (Tool)** | CoinGecko timeout, Prisma constraint failure | Retry (max 2), fall back to cache, emit `toolId`, `cacheKey` | Raise `AgentError` → Business Logic decides degradation |
| **Business Logic** | Invalid holdings data, unsupported intent | Attempt reconciliation, return last good snapshot, set `severity='warning'` | Pass `AgentEnvelope` with degraded `data` + warning |
| **Orchestrator** | Intent routing failure, missing domain spec | Route to `FallbackAgent`, set `action='none'`, log critical | UI receives error with no navigation change |
| **State** | Store hydration mismatch, stale cache key | Reset slice, request re-sync, mark `retryable` | Hooks surface inline warnings, optionally auto-retry |
| **Hooks/UI** | Schema mismatch, renderer crash | Validate with Zod before commit, show toast/banner | Highest-level error, instructs user what to do |

## 5. Bubbling Pipeline
1. **Backend orchestrator** returns `AgentEnvelope` over SSE/fetch.
2. **`useAgentStream` hook** parses the envelope:
   - Commits `message`/`data` when no fatal error.
   - When `severity==='error'`, skips `data` mutation but keeps `message` and records error in the `ErrorStore`.
3. **State layer** stores `AgentError` objects with context (domain, route, timestamp) for UI selectors.
4. **Components** subscribe to specific slices & errors, showing inline alerts or fallback skeletons while referencing `error.code` for translators.
5. **Monitoring** automatically receives the same diagnostics: Sentry for orchestrator/UI levels, LangSmith for graph traces, PostHog for feature failure metrics.

## 6. Monitoring Hooks
- **LangSmith:** record `traceId`, tool usage, retries per orchestrator execution.
- **Sentry:** log orchestrator + UI-level errors; attach `developerMessage` and `traceUrl` for post-mortem.
- **PostHog:** emit events when agents degrade to cached data or navigation side effects fail.
- **Custom dashboards:** aggregate `AgentDiagnostics` to spot slow tools or repeated error codes.

## 7. Adding or Updating an Agent (Checklist)
1. **Spec:** Add/modify `DomainAgentSpec` entry with intents, required tools, and schema version.
2. **Contracts:** Define/adjust TypeScript types referenced by Business Logic, API, State, and Hooks.
3. **Business logic:** Implement/extend pure services with SHARP coverage (success + degradation + failure paths).
4. **API layer:** Add tool adapters/repositories plus mocks; ensure they emit structured `AgentError`s.
5. **State:** Create/extend Zustand slices for the module view model and register error selectors.
6. **Hooks:** Build/extend reusable hooks that inject orchestrator commands and map envelopes to UI-ready state.
7. **Orchestrator wiring:** Update LangGraph nodes + `/api/chat` to include the new agent, ensuring diagnostics/errors bubble correctly.
8. **UI reminders:** Document rendering expectations, fallback copy, and toast messages tied to `error.code`.

## Local Python (FastAPI + LangGraph) reminder

Activate the shared venv before running Python servers or installs:

```bash
source ai/.venv/bin/activate
```
Following this SDD + Orc-BASH spec keeps modules decoupled, agents testable, and error signals consistent from deepest tool call to the final UI interaction.
