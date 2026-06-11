# ADR-001: Multi-Framework Frontend Core — FSD + Vertical Slice Extraction

- Status: ACCEPTED
- Date: 2026-06-09T18:00:00Z
- Spec: SPEC-001 v1.0.0 (hash: sha256:5ac6301bec0e51192f30229309d6327185c39c3b87025927f84cee8660d9299d)
- Author: Software Architect Agent / AIfolio Dev

## Constitution Check

| Article | Status | Notes |
|---------|--------|-------|
| I — Library-First | COMPLIES | Uses Zod for runtime validation. TypeScript workspace tooling. No custom framework. |
| II — Test-First | COMPLIES | TDD Agent dispatched before Programmer. Contract tests before extraction. |
| III — Simplicity Gate | COMPLIES | Every package traces to REQ-01 through REQ-10. Two shared packages only. |
| IV — Anti-Abstraction Gate | COMPLIES | Contracts package has 3+ concrete consumers (Next.js + planned Vue/Svelte/Angular). ChartSpec alone has 14+ import sites. |
| V — Integration-First Testing | COMPLIES | Shared contract tests verify integration across packages. Unit tests only for pure logic. |
| VI — Security-by-Default | COMPLIES | No new attack surface. Security Agent review before merge. |
| VII — Spec Integrity | COMPLIES | References SPEC-001 v1.0.0 hash throughout. |
| VIII — Observability | N/A | Structural refactor. No new runtime paths or external I/O. |

## Research Summary

- Research artifact: N/A — no new technology choices. Uses existing TypeScript, Zod, Vitest, npm workspaces.
- Key decision: FSD + vertical slice hybrid — FSD for framework apps (minus entities layer), vertical domain slices for shared packages.

## Planning Preflight Evidence

- Coordinator Planning Preflight: PASS
- Spec hash verified at: 2026-06-09T17:00:00Z
- Red-Team status and artifact: Not dispatched (structural refactor, no new attack surface)
- System Blueprint status and artifact: N/A
- CodeBase Analyzer reports consumed: `ADS-project-knowledge/reports/codebase-analysis/ANALYSIS-aifolio-2026-06-09.md`
- Reverse-spec artifacts consumed: N/A (brownfield extraction, not reverse-spec)
- Validator result or waiver: Spec DoD PASS

## Context

**Problem:** All frontend contracts, pure logic, and UI code lives in `web/nextjs/src/features/`. Non-React frameworks cannot reuse domain logic without duplication. 481 cross-feature import edges (57 production) indicate coupling that will compound with each new framework app.

**System drivers:**
- Multi-framework portability (primary): identical business behavior across React, Vue, Svelte, Angular
- Coupling reduction: 57 production cross-feature imports must route through shared packages
- Incremental migration: system stays working at every step
- Team velocity: clear boundaries reduce cognitive load and merge conflicts

**Constraints:**
- Existing dirty worktree with in-flight changes — do not break current Next.js app
- Backend Python services unchanged — extraction is frontend-only
- Article II mandates tests before implementation

**If we do nothing:** Adding Vue/Svelte/Angular means duplicating every contract and logic function. Cross-feature coupling grows. Bug fixes require N copies.

## Decision

Extract framework-agnostic domain entities and pure logic from `web/nextjs/src/features/` into two shared packages at `web/packages/`. Framework apps use FSD for UI layers only (app, pages, widgets, features, shared) and consume domain entities from the shared contracts package.

**Pattern(s) selected:** Feature-Sliced Design (UI apps) + Vertical Domain Slices (shared packages)

## Default Heuristic Alignment

- Default heuristic: modular monolith at the macro level, vertical slices for feature ownership, and hexagonal boundaries only where external I/O or business-critical logic justify them. Frontend applications use Feature-Sliced Design.
- Alignment: FOLLOWS
- Notes: FSD for UI apps is the documented default. Vertical slicing for shared packages follows the modular-monolith heuristic at the package level. No hexagonal ports needed — these are pure functions and types, not I/O boundaries.

## Rationale

- Multi-framework portability → addressed by extracting domain entities into a framework-agnostic contracts package that any app can import
- Coupling reduction → addressed by routing all shared types through package public APIs, eliminating direct cross-feature imports
- Incremental migration → addressed by phased extraction with re-exports maintaining backward compatibility
- Team velocity → addressed by clear package boundaries and FSD layer rules in each app

## Pattern Evaluation

| Pattern | Fit Band | Adaptability | Evidence Basis | Pros | Cons | Key Tradeoffs | Verdict |
|---------|----------|--------------|----------------|------|------|---------------|---------|
| FSD (UI) + Vertical Slice (packages) | Strong fit | High | prior_art | Domain entities shared cleanly; UI apps stay thin and framework-native; import direction enforced; entities live once | Two levels of convention to document (FSD for apps, slices for packages) | Convention overhead justified by multi-framework reuse | **SELECTED** |
| Full FSD everywhere (including packages) | Weak fit | Medium | analogical | Single methodology; familiar FSD vocabulary | FSD `pages`/`widgets`/`app` layers nonsensical for a headless package; forces UI-only concepts onto pure logic | Misapplied abstraction; entities layer redundant in apps when contracts package exists | Not selected — FSD is for frontend apps, not libraries |
| Hexagonal (ports + adapters) | Viable fit | High | analogical | Clean I/O boundaries; highly testable | Over-engineered for packages with no I/O; port/adapter ceremony for pure functions adds no value | Overkill for this extraction — no external adapters needed | Not selected — no I/O in shared packages justifies it |
| Monorepo with no internal structure | Rejected | Low | prior_art | Zero convention overhead | No import enforcement; coupling returns immediately; FSD benefits lost | No protection against the problem we're solving | Not selected — recreates the coupling problem |

## Quality Attribute Scorecard

| Axis | Score (1-5) | Confidence | Strengths | Weaknesses | Rationale | Activation Source | Mitigation | Review Trigger |
|---|---|---|---|---|---|---|---|---|
| modifiability | 5 | prior_art | Adding a new framework app = new folder consuming existing packages; changing domain logic = one place | Must update re-exports during migration window | FSD + shared packages maximize change locality | always-on | — | New framework with incompatible module system |
| modularity | 5 | measured | Explicit package boundaries; FSD layer isolation; public API enforcement | Two convention systems to learn | Contracts package measured at 14+ distinct consumers for ChartSpec alone | always-on | — | Shared packages exceed 20 export modules |
| scalability | 4 | analogical | Horizontal scaling via additional framework apps; no coupling ceiling | Each new app needs FSD setup | Linear cost per new app, not quadratic | always-on | — | >6 framework apps |
| reliability | 4 | prior_art | Shared contract tests catch regressions across all consumers | Migration window has dual-path risk | Re-exports preserve existing behavior during transition | always-on | — | Contract test failures in CI |
| security | 4 | analogical | No new attack surface; packages are pure types and functions | — | Structural refactor, no auth/network/storage changes | always-on | — | Shared package gains runtime I/O |
| operability | 4 | prior_art | npm workspaces handle package resolution; existing CI extends | Workspace config adds build-order complexity | Standard npm workspace pattern | always-on | — | Migration to pnpm or Turborepo |
| cost | 5 | measured | Zero new infrastructure; zero new runtime dependencies; developer time only | Migration effort across phases | All tooling already in repo or free | always-on | — | — |
| testability | 5 | prior_art | Pure functions trivially testable; contracts have schema validation tests; no mocking needed | — | Framework-agnostic tests run in Node without browser | always-on | — | Logic functions gain side effects |
| portability | 5 | analogical | Entire purpose of the extraction; proven by contract tests across frameworks | — | Shared packages tested independently of any framework | spec REQ-06 | — | Framework with incompatible TypeScript target |

## Overall Strengths

- Domain entities defined once, consumed everywhere — single source of truth
- Framework apps stay thin: routing, state binding, rendering only
- FSD hard invariants (import direction, slice isolation, public APIs) prevent coupling regrowth
- Migration is fully incremental with rollback at every phase

## Overall Weaknesses

- Two convention systems (FSD for apps, vertical slices for packages) require documentation
- Migration window with re-exports adds temporary complexity
- Developers must know which layer they're working in (package vs app)

## Tradeoff Tension

We are trading single-convention simplicity for multi-framework portability. The second convention (vertical slices in packages) is simpler than FSD and exists precisely because FSD's UI-only layers don't apply to headless packages.

## Why This Won

The primary driver is multi-framework portability. Only the selected pattern places domain entities in a shared location while keeping UI-layer FSD in each app. Full FSD everywhere would force entities into each app (defeating portability). Hexagonal adds ceremony for packages with no I/O. The selected pattern is the only one that matches all four system drivers without over-engineering.

## Runner-Up Comparison

- Runner-up: Full FSD everywhere (entities layer in each app)
- Why it lost: Entities are abstract and reusable — placing them per-app contradicts the extraction goal. Apps would either duplicate entity model logic or import from a package anyway, making the app-level entities layer an empty pass-through.

## Consequences

**Positive:**
- Any new framework app imports contracts + frontend-core and only builds UI
- Cross-feature coupling drops to zero for extracted types
- FSD import rules in apps prevent coupling regrowth
- Contract tests verify behavioral parity without running full apps

**Negative / Tradeoffs:**
- Re-export shims during migration add temporary indirection
- Developers must understand package vs app boundary
- npm workspace build-order adds CI complexity

**Risks:**
- Risk: AG-UI types tightly coupled to CopilotKit React APIs → mitigation: extract payload shapes only (per OQ-02), keep action registration in React app
- Risk: Re-exports linger past migration → mitigation: 2-week deadline per behavior.spec.md Section 4, tracked in pipeline state
- Risk: Workspace resolution breaks existing Next.js dev → mitigation: re-exports ensure no import path changes until Phase 4

## Mitigations Required

No axis scored 1 or 2. No mitigations required.

## Migration Safety

| Safety Item | Decision / Evidence | Owner |
|---|---|---|
| Expand/contract shape | Additive: new packages created alongside existing code. Re-exports maintain old paths. No removal until all consumers migrated. | AIfolio Dev |
| Dual-write or read-routing plan | Not needed — no data migration. Types exist in both locations via re-export during transition. | — |
| Backfill plan | Not needed — no data to backfill. Import paths updated file-by-file. | — |
| Reconciliation checks | TypeScript compiler verifies type compatibility. Shared contract tests verify behavioral equivalence. | CI |
| Observability proving phase health | Next.js build + test suite pass after each phase. No runtime behavior change to observe. | CI |
| Rollback test | Each phase is revertible: delete new package folders, restore original files from git. Re-exports mean old paths never break. | AIfolio Dev |
| Cutover approval and timing | Each phase requires verification gate pass (behavior.spec.md G-01 through G-05) before next phase starts. | AIfolio Dev |
| Point of no return | Phase 4 (re-export removal) is the point of no return for old import paths. All prior phases are fully reversible. | AIfolio Dev |
| Post-cutover verification | After Phase 5: ESLint/Steiger boundary rules pass. Zero cross-feature type imports. All shared tests green. | CI |

## Re-evaluation Triggers

- Calendar trigger: 6 months after Phase 5 completes — assess whether vertical slice convention in packages needs revision
- Scale trigger: If shared packages exceed 30 export modules, evaluate splitting contracts into per-domain sub-packages
- Topology trigger: If a non-TypeScript consumer needs the contracts (e.g., Dart, Kotlin), evaluate code generation from schemas
- Dependency trigger: If Zod is abandoned or a breaking major version ships, evaluate validation library migration

## Module / Service Boundaries

```
web/
  packages/
    contracts/                    # Domain entities — types + Zod schemas
      src/
        entities/
          chart/                  # ChartSpec, ChartActionsPort
            index.ts
          chat/                   # ChatMessage, ChatModelOption, ChatState, etc.
            index.ts
          ml/                     # MlDatasetOption, MlDatasetCacheEntry, MlDatasetState
            index.ts
          ag-ui/                  # AG-UI tool/action payload types
            index.ts
      package.json
      tsconfig.json

    frontend-core/                # Pure logic — vertical domain slices
      src/
        features/
          model-selection/        # resolveFallbackModelSelection, resolveFetchedModelSelection
            index.ts
          chat-submission/        # normalizeSubmissionValue, buildChatHistoryWindow, createUserChatMessage, createAssistantChatMessage
            index.ts
          ml-validation/          # validateTrainingInput
            index.ts
          chart-formatting/       # formatChartData, eChartOptionsBuilder
            index.ts
        shared/
          constants/              # FALLBACK_CHAT_MODELS, shared config values
            index.ts
      package.json
      tsconfig.json

  nextjs/                         # React/Next.js UI app — FSD layers (no entities layer)
    src/
      app/                        # Next.js routing, providers, layout
      pages/                      # Route-level page compositions
      widgets/                    # Self-contained UI blocks (chat sidebar, chart workspace, nav)
      features/                   # User-facing actions (send-chat-message, train-model, etc.)
        ai-chat/
          ui/                     # ChatBar, ChatSidebar, UIFeedback components
          model/                  # Zustand store, React state adapters
          api/                    # React hooks calling frontend-core logic
          index.ts
      shared/                     # UI kit, framework helpers, API client config
        ui/
        lib/
        config/

  vue/                            # Future — same FSD structure, Vue composables instead of hooks
  svelte/                         # Future — same FSD structure, Svelte stores instead of hooks
  angular/                        # Future — same FSD structure, Angular services instead of hooks
```

**Boundary rules:**

| Rule | Enforcement |
|---|---|
| `contracts` has zero runtime deps beyond Zod | package.json audit in CI |
| `frontend-core` depends only on `contracts` | package.json + ESLint no-restricted-imports |
| Framework apps may depend on `contracts` + `frontend-core` | workspace dep graph |
| No circular deps between packages | tsc build-order |
| Framework apps do NOT have an `entities/` layer | Steiger/lint rule — entities live in `contracts` package |
| FSD hard invariants enforced in each app | Steiger lint (no-cross-imports, no-public-api-sidestep) |
| Same-layer slice isolation in apps | Steiger lint |
| Public API per slice (`index.ts`) in apps | Steiger lint |

## API / Event Contract Summary

Defined in full in `api.spec.md`. Key interfaces:

- `@aifolio/contracts/entities/chart` — ChartSpec, ChartActionsPort, ChartSpecSchema
- `@aifolio/contracts/entities/chat` — ChatMessage, ChatModelOption, ChatState, ChatStateActions, ChatAssistantPayload, ModelSelectionResult
- `@aifolio/contracts/entities/ml` — MlDatasetOption, MlDatasetCacheEntry, MlDatasetState
- `@aifolio/contracts/entities/ag-ui` — AG-UI tool/action payload types
- `@aifolio/frontend-core/features/model-selection` — resolveFallbackModelSelection, resolveFetchedModelSelection, FALLBACK_CHAT_MODELS
- `@aifolio/frontend-core/features/chat-submission` — normalizeSubmissionValue, buildChatHistoryWindow, createUserChatMessage, createAssistantChatMessage
- `@aifolio/frontend-core/features/ml-validation` — validateTrainingInput
- `@aifolio/frontend-core/features/chart-formatting` — formatChartData, eChartOptionsBuilder

## Enforcement

- ESLint `no-restricted-imports`: block framework-specific imports in shared packages
- Steiger: enforce FSD hard invariants in framework apps, including "no entities layer" convention
- TypeScript project references: enforce build-order and prevent circular deps
- CI gate: shared package tests must pass independently (no framework deps installed)
- PR template: extraction PRs must reference the phase and verification gate from behavior.spec.md

## Directory Structure Decision

The directory layout above is the canonical target. Key conventions:

1. **Shared packages use vertical domain slices** — `entities/<domain>/` in contracts, `features/<capability>/` in frontend-core
2. **Framework apps use FSD minus entities** — layers: `app`, `pages`, `widgets`, `features`, `shared`
3. **Entity model/api segments satisfied by packages** — apps only carry entity `ui/` if needed (rare; usually handled via widgets)
4. **Next.js routing collision resolved via "flat FSD in src/"** — Next.js `app/` directory is both FSD app layer and Next routing; FSD pages/widgets/features/shared live as siblings under `src/`

## Complexity Justification

No Constitution Check exceptions. Table empty.

## Related Decisions

- Supersedes: None
- Relates to: None (first ADR for this project)

## Parallel Delivery Plan

After TDD certifies tests, these extraction slices can be worked in parallel:

| Slice | Package | Depends On | Can Parallel With |
|---|---|---|---|
| chart entities (ChartSpec, ChartActionsPort) | contracts | nothing | all other entity extractions |
| chat entities (ChatMessage, ChatModelOption, ChatState, etc.) | contracts | nothing | all other entity extractions |
| ml entities (MlDatasetOption, MlDatasetCacheEntry) | contracts | nothing | all other entity extractions |
| ag-ui entities (tool/action types) | contracts | nothing | all other entity extractions |
| model-selection logic | frontend-core | chat entities in contracts | chat-submission logic |
| chat-submission logic | frontend-core | chat entities in contracts | model-selection logic |
| ml-validation logic | frontend-core | ml entities in contracts | model-selection, chat-submission |
| chart-formatting logic | frontend-core | chart entities in contracts | all other logic extractions |

Phase 3 (Next.js import migration) and Phase 4 (re-export removal) are sequential and cannot parallel.

## Implementation Outline Trigger Check

Triggers evaluated per `skills/implementation-outline/SKILL.md`:

- Broad package moves across multiple directories: **YES**
- Cross-framework adapters required: **YES** (future apps consume packages differently)
- Public contract definitions: **YES** (all shared package exports)
- Migration phases with rollback points: **YES**

**Result:** Implementation Outline is REQUIRED. Will be produced as the next artifact.
