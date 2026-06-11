# Feature Spec: Multi-Framework Frontend Core Extraction

---

## Header Metadata

| Field | Value |
|-------|-------|
| spec_id | SPEC-001 |
| version | 1.0.0 |
| status | APPROVED |
| content_hash | sha256:5ac6301bec0e51192f30229309d6327185c39c3b87025927f84cee8660d9299d |
| feature_name | FEAT-001-multi-framework-frontend-core |
| last_edited | 2026-06-09T17:00:00Z |
| owner | AIfolio Dev |
| spec_agent | Spec Agent |
| spec_mode | brownfield |

---

## Overview

Extract framework-agnostic frontend logic (contracts, pure functions, validation, normalization) from `web/nextjs` into shared packages so the same application can be implemented in Next.js, Vue, Svelte, and Angular with behavioral parity enforced via shared contracts and tests.

---

## Problem Statement

**Current state:** All frontend logic -- contracts (ChartSpec, ChatMessage, ML training payloads), pure business logic (model selection, training validation, chart formatting), and framework-specific UI code (React hooks, Zustand stores, JSX components) -- lives inside `web/nextjs/src/features/`. Non-React framework implementations cannot reuse any of this logic without duplicating it.

**Desired state:** Framework-agnostic contracts and pure logic live in shared packages consumable by any JavaScript/TypeScript frontend framework. Each framework app (Next.js, Vue, Svelte, Angular) imports the same contracts and logic, implementing only framework-specific rendering, state management binding, and routing.

**Why now:** The codebase already separates `logic/` from `react/` within each feature slice. The extraction path is visible and low-risk. Cross-feature coupling (481 import edges, 57 production) must be resolved before any additional framework apps are built. Technical debt has reached the tipping point where adding a second UI framework without extraction would create permanent duplication.

**Success signal:** A second framework app (Vue, Svelte, or Angular) can render the chat, ML training, and chart features by importing shared contracts and logic packages -- confirmed by shared contract tests passing across all consuming apps.

---

## User Journey

1. **Trigger:** A developer begins implementing a new framework app (e.g., Vue) for the AIfolio frontend.
2. **Steps:**
   1. Developer installs the shared contracts package and the shared frontend-core logic package.
   2. Developer implements framework-specific adapters (Vue composables, Svelte stores, Angular services) that consume the shared logic.
   3. Developer writes framework-specific UI components that conform to the behavioral contracts defined in the shared packages.
   4. Developer runs the shared contract test suite to verify behavioral parity with the reference Next.js implementation.
3. **Outcome:** The new framework app passes all shared contract tests and demonstrates identical business behavior to the Next.js app for all covered features.
4. **Alternate paths:**
   - If a shared contract test fails: the developer inspects which behavioral contract is violated and fixes their framework adapter.
   - If a contract is missing for a needed behavior: the developer proposes a contract addition through the spec change protocol.

---

## Scope

**In scope:**
- Extraction boundary definitions: which modules move from `web/nextjs` into shared packages
- Contract definitions for: ChartSpec, ChatMessage, ChatState, ML training payloads, ML dataset manifests, AG-UI action/tool contracts, model selection results
- Pure logic extraction candidates: model selection, training input validation, chart formatting, chat submission normalization, AG-UI payload normalization
- State portability contracts: observable state shape and transition rules that any framework state manager must satisfy
- UI behavioral parity requirements: observable user-facing behavior each framework app must reproduce
- Migration safety requirements: non-breaking extraction, temporary re-exports, verification gates
- Shared contract test requirements: tests that run against any framework implementation

**Out of scope:**
- Backend Python services (FastAPI, ML trainers, LangGraph agents) -- not changing
- Specific folder layout or FSD layer assignments -- that belongs to the ADR
- Package naming decisions (e.g., `@aifolio/contracts` vs alternatives) -- that belongs to the ADR
- Package manager choice or workspace configuration -- that belongs to the ADR
- Implementation of Vue, Svelte, or Angular apps -- separate features
- UI visual design or styling -- framework-specific concern
- Next.js API route extraction -- separate concern
- Backend route reorganization (FLAW-006, FLAW-007 from analysis) -- separate concern

---

## Requirements

- REQ-01: All shared domain contracts (ChartSpec, ChatMessage, ChatModelOption, ChatAssistantPayload, ChatHistoryMessage, MlDatasetOption, MlDatasetCacheEntry, AG-UI tool action types) must be defined in a shared contracts package with zero framework-specific imports.
- REQ-02: All pure logic functions (model selection, training input validation, chart formatting/normalization, chat submission normalization, AG-UI payload normalization) must be extractable into a shared logic package with zero framework-specific imports.
- REQ-03: The shared contracts package must export TypeScript type definitions and runtime validation schemas (Zod or equivalent) for every domain entity.
- REQ-04: The shared logic package must export pure functions that accept typed inputs and return typed outputs with no side effects and no dependency on React, Vue, Svelte, or Angular APIs.
- REQ-05: The existing Next.js app must continue to function without behavioral regression after extraction, verified by existing tests passing and by the shared contract test suite.
- REQ-06: Each framework app consuming the shared packages must satisfy identical behavioral contracts for: chat message creation, model selection resolution, ML training request validation, chart spec normalization, and AG-UI action payload construction.
- REQ-07: State portability contracts must define the observable state shape (fields, types, initial values) and legal state transitions without prescribing a specific state management library.
- REQ-08: Extraction must be non-breaking: temporary re-export modules must preserve existing import paths in `web/nextjs` until all consumers are migrated.
- REQ-09: Cross-feature coupling (currently 57 production imports between feature slices) must be resolved by routing shared dependencies through the extracted contracts package rather than direct cross-feature imports.
- REQ-10: Every extracted contract and logic function must have at least one framework-agnostic unit test that runs without any framework runtime.

---

## Acceptance Criteria

- AC-01 (REQ-01) [P1]: Given the shared contracts package is built, when a TypeScript consumer imports `ChartSpec`, then the imported type matches the shape defined at `features/charts/contracts/chart.types.ts` with fields: id (string), title (string), type (union of 15 chart types), xKey (string), yKeys (string[]), data (Array<Record<string, number | string>>), and all optional fields (description, xLabel, yLabel, zKey, colorKey, errorKeys, unit, currency, timeframe, source, meta).
- AC-02 (REQ-01) [P1]: Given the shared contracts package is built, when a consumer imports `ChatMessage`, then the imported type has fields: id (string), role ("user" | "assistant"), content (string), createdAt (number), chartSpec (ChartSpec | null, optional).
- AC-03 (REQ-01) [P1]: Given the shared contracts package is built, when a consumer imports `MlDatasetOption`, then the imported type has fields: id (string), label (string), description (string, optional).
- AC-04 (REQ-02) [P1]: Given the shared logic package is built, when `resolveFallbackModelSelection` is called with `{ selectedModelId: null }` and no options, then it returns `{ modelOptions: FALLBACK_CHAT_MODELS, selectedModelId: "gemini-3-flash-preview" }`.
- AC-05 (REQ-02) [P1]: Given the shared logic package is built, when `resolveFetchedModelSelection` is called with `{ selectedModelId: null, result: { models: [...], currentModel: "model-x" } }`, then it returns `{ modelOptions: result.models, selectedModelId: "model-x" }`.
- AC-06 (REQ-03) [P1]: Given the shared contracts package exports a Zod schema for ChartSpec, when an object missing the required `id` field is validated, then validation fails with an error identifying the `id` field.
- AC-07 (REQ-04) [P1]: Given the shared logic package is imported in a Node.js environment with no framework packages installed, when all exported functions are invoked with valid inputs, then they execute without throwing import errors or requiring framework globals.
- AC-08 (REQ-05) [P1]: Given the Next.js app's import paths are updated to use the shared packages (via re-exports or direct imports), when the existing frontend test suite runs, then all previously passing tests continue to pass.
- AC-09 (REQ-06) [P1]: Given a second framework app implements the chat feature using shared contracts and logic, when the shared contract test suite runs against both the Next.js and second framework implementations, then both pass all shared behavioral assertions.
- AC-10 (REQ-07) [P2]: Given the state portability contract defines ChatState with fields (messages, inputHistory, historyCursor, isSending, modelOptions, selectedModelId, isModelsLoading, screenFeedback), when any framework state manager implements this contract, then it must satisfy: initial messages is empty array, initial isSending is false, initial selectedModelId is null.
- AC-11 (REQ-08) [P1]: Given extraction creates a shared package for `ChartSpec`, when existing code at `@/features/charts/contracts/chart.types.ts` is imported by 14+ AG-UI modules, then a re-export file at the original path forwards to the shared package without requiring consumer changes.
- AC-12 (REQ-09) [P2]: Given cross-feature imports are routed through the shared contracts package, when an ESLint or Steiger boundary rule is applied, then zero production imports directly between feature slices remain for types/contracts that have been extracted.
- AC-13 (REQ-10) [P1]: Given the shared contracts and logic packages exist, when `npm test` (or equivalent) is run in each package in isolation (no framework dependencies installed), then all tests pass.
- AC-14 (REQ-06) [P2]: Given the ML training validation logic is extracted, when `validateTrainingInput` is called with an empty dataset ID, then it returns a validation failure result with error code `DATASET_REQUIRED`.
- AC-15 (REQ-07) [P2]: Given the state portability contract for MlDatasetState, when any implementation transitions from `isLoadingDataset: true` to `isLoadingDataset: false`, then either `datasetCache` has a new entry for the requested dataset ID or `error` is non-null.

---

## Invariants

- INV-01: Shared contract packages must never import from any framework-specific package (react, vue, svelte, @angular/*, next, nuxt).
- INV-02: Shared logic packages must never import from any framework-specific package or from any state management library (zustand, pinia, ngrx, svelte/store).
- INV-03: Every exported type in the shared contracts package must have a corresponding runtime validation schema (Zod or equivalent) that produces the same type via inference.
- INV-04: The shared contract test suite must be executable without installing any frontend framework -- only TypeScript, the test runner, and the shared packages.
- INV-05: Re-export modules in `web/nextjs` must re-export the exact same public API surface that existed before extraction -- no narrowing, no renaming, no additional exports.
- INV-06: State portability contracts must define initial values for every field -- no field may have an undefined initial state.
- INV-07: Pure logic functions in the shared package must be deterministic: given identical inputs, they must always produce identical outputs regardless of execution environment.

---

## Edge Cases

- EC-01: What happens when the shared contracts package is imported by a consumer that uses an older TypeScript version (e.g., 4.x vs 5.x)?
  Expected behavior: The package must compile to declaration files compatible with TypeScript >= 5.0. Consumers on older versions are out of scope and documented as unsupported.
- EC-02: What happens when a re-export module in `web/nextjs` is imported but the shared package has not been built yet (e.g., fresh clone without `npm install`)?
  Expected behavior: TypeScript compilation fails with a clear missing-module error. The build system must resolve shared packages before app compilation.
- EC-03: What happens when a framework app implements the state portability contract but omits an optional field from the state shape?
  Expected behavior: Optional fields (those marked nullable in the contract) may be omitted from state implementations. Required fields with defined initial values must be present. The shared contract test suite must validate required field presence.
- EC-04: What happens when two framework apps produce different intermediate representations for the same ChartSpec input?
  Expected behavior: Intermediate representations are framework-specific. Only the final observable output (rendered chart data structure passed to the charting library) must match the contract. Internal state shape differences are acceptable.
- EC-05: What happens when a cross-feature import is not resolvable through the shared contracts package because the dependency is on framework-specific behavior (e.g., a React hook)?
  Expected behavior: Framework-specific cross-feature dependencies remain in the framework app. Only type/contract and pure-logic dependencies are extracted. The boundary rule enforcement must distinguish between extracted (shared) and retained (framework-specific) cross-feature imports.
- EC-06: What happens when a new chart type is added to ChartSpec after extraction?
  Expected behavior: The chart type union is extended in the shared contracts package. All framework apps must update their chart type handling. The shared contract test suite must include a test for the new type. The re-export in `web/nextjs` automatically exposes the new type.

---

## Dependencies

| Dependency | What It Provides | Failure Mode | Fallback |
|------------|------------------|--------------|----------|
| TypeScript compiler | Type checking and declaration generation for shared packages | Shared packages cannot be compiled or consumed | none -- blocks feature |
| Zod (or equivalent runtime validation library) | Runtime validation schemas matching TypeScript types | Contract validation unavailable at runtime | Types-only export without runtime validation (degrades REQ-03) |
| Existing `web/nextjs` test suite | Regression verification after extraction | Cannot confirm non-breaking migration | Manual testing against key user flows |
| Existing feature `logic/` modules | Source of pure functions to extract | Nothing to extract if missing | N/A -- modules exist per codebase analysis |
| Existing feature `__types__/` modules | Source of type contracts to extract | Nothing to extract if missing | N/A -- modules exist per codebase analysis |
| Monorepo workspace tooling | Package resolution between shared packages and apps | Apps cannot import shared packages | N/A -- ADR decides tooling, but some workspace support is required |

---

## Open Questions

- OQ-01: Should the shared contract test suite use a specific test runner (Vitest, Jest) or be runner-agnostic? -- Owner: AIfolio Dev -- Resolve by: 2026-06-16
- OQ-02: Should AG-UI tool action types be extracted given their current tight coupling to CopilotKit React APIs, or should only the payload shapes be extracted while the action registration remains React-specific? -- Owner: AIfolio Dev -- Resolve by: 2026-06-16

---

## Constitution Compliance

| Article | Status | Notes |
|---------|--------|-------|
| I -- Library-First | COMPLIES | Uses Zod for runtime validation (existing well-maintained library). No custom validation framework. |
| II -- Test-First | COMPLIES | TDD Agent will be dispatched before Programmer. Shared contract tests written before implementation extraction. |
| III -- Simplicity Gate | COMPLIES | Every extracted module traces to REQ-01 through REQ-10. No speculative packages. |
| IV -- Anti-Abstraction Gate | COMPLIES | Contracts extracted have 3+ concrete consumers (Next.js, planned Vue, planned Svelte/Angular). ChartSpec alone has 14+ import sites. |
| V -- Integration-First Testing | COMPLIES | Shared contract tests verify integration between packages and consuming apps. Unit tests supplement for pure logic. |
| VI -- Security-by-Default | COMPLIES | Security Agent review required before merge. Extraction does not introduce new attack surface. |
| VII -- Spec Integrity | COMPLIES | All downstream agents must reference SPEC-001 version 1.0.0 and validated content hash. |
| VIII -- Observability | N/A | Extraction is a structural refactor of shared packages. No new runtime error paths or external I/O introduced. Existing observability in the Next.js app is preserved. |

---

## Implementation Readiness Gate

- [x] spec_id assigned and unique (verified against existing `ADS-project-knowledge/reports/pipeline/` folders)
- [x] version set to correct semver
- [x] status set to APPROVED
- [x] content_hash computed using the Speckit canonical hash rule and verified by the provider-local validator
- [x] feature_name matches the FEAT folder name exactly
- [x] Zero `[NEEDS CLARIFICATION]` markers remain in this file
- [x] All Open Questions have an owner and a resolution target date
- [x] All REQ-* items are testable and contain no vague qualifiers
- [x] All REQ-* items have at least one AC
- [x] All AC-* items have a [P1], [P2], or [P3] priority tag
- [x] All AC-* items follow Given/When/Then format
- [x] All Invariants are written as absolute, falsifiable statements
- [x] All Edge Cases have an explicit Expected Behavior
- [x] Dependencies table is complete -- no blank failure mode or fallback cells
- [x] Constitution Compliance table complete -- all 8 articles marked COMPLIES / EXCEPTION / N/A
- [x] Scope: in-scope list present and non-empty
- [x] Problem Statement: "Why now" field is filled
- [x] User Journey: trigger, steps, outcome, and alternate paths are present
- [x] Scope: out-of-scope list present and non-empty
- [x] Full spec-system package present: all `PRESENT` files listed in spec-manifest.md exist
- [x] behavior.spec.md complete (feature has non-trivial ordering/precedence rules for migration sequencing)
- [x] traceability.spec.md complete (marked "pending implementation")
- [x] spec-manifest.md complete -- all 10 logical files listed with `PRESENT` or `OMITTED` and concrete reasons
- [x] spec-dod.md filled and all items PASS or NA with concrete justification
- [x] spec-dod.md Spec Agent sign-off row completed; Coordinator row reserved for Planning Preflight
- [x] Brownfield evidence paths recorded in spec-manifest.md

**Gate result:** PASS

---

## Agent Directives

Always:
- Reference existing code by file path and function name -- do not restate what existing code does
- Treat the existing `logic/` and `__types__/` separation in each feature as the extraction boundary indicator
- Preserve the existing public API surface of all extracted modules exactly

Ask before:
- Adding any new dependency to the shared packages beyond TypeScript and a validation library
- Changing the type signature of any existing exported function during extraction

Never:
- Import React, Vue, Svelte, Angular, or Next.js from shared contract or logic packages
- Modify backend Python services as part of this feature
- Prescribe specific FSD layer assignments or folder structures in the spec (that is ADR scope)
