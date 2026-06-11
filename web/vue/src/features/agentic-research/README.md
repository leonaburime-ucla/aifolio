# Feature: Agentic Research

Structured using Feature-Sliced Design (FSD).

## Directory Layout

```
features/agentic-research/
├── api/
│   ├── index.ts                    # Barrel export
│   └── agenticResearchApi.ts       # Fetch adapters wrapping @aifolio/frontend-core
├── model/
│   ├── index.ts                    # Barrel export
│   └── useAgenticResearch.ts       # Composable: owns reactive state + orchestrates api/
└── components/
    └── AgenticResearchWorkspace.vue # Thin UI shell consuming the composable
```

## FSD Layers

| Layer | What lives here | Depends on |
|-------|----------------|------------|
| `api/` | Fetch wrappers that call `@aifolio/frontend-core` with the correct base URL | `@aifolio/frontend-core`, `@aifolio/contracts` |
| `model/` | Feature-scoped composables that own refs, computed, and orchestration logic | `api/` |
| `components/` | Vue SFCs — template + thin wiring to the composable | `model/` |

## Composables vs model/ — When to Use Which

In Vue, a **composable** is any `useX()` function that uses the Composition API (`ref`, `computed`, `watch`) and returns reactive state. It's the standard pattern for reusable stateful logic.

In FSD, the question is **where** to put them:

| Location | Scope | Examples |
|----------|-------|----------|
| `src/composables/` | **App-wide shared utilities** — cross-feature, no domain logic | `useApi`, `useChartStore`, `useMlDataStore` |
| `features/<name>/model/` | **Feature-scoped logic** — owns the feature's state, calls its api layer, exposes derived data | `useAgenticResearch`, `useTrainingOrchestrator` |

The rule: if the composable is **about** a specific feature (owns its state, calls its endpoints), it goes in that feature's `model/`. If it's a generic utility consumed by multiple features, it goes in top-level `composables/`.

Both are composables. The difference is organizational — FSD's `model/` layer makes the dependency graph explicit: `components/ → model/ → api/`. You never import `api/` directly from a component.

## Tests

```bash
# Unit tests (vitest) — tests the composable with mocked fetch
npm run test:unit

# E2E tests (playwright) — hits live app at localhost:3000
npm run test:e2e
```

Unit tests live at `web/vue/__tests__/unit/useAgenticResearch.spec.ts`.
E2E tests live at `web/vue/__tests__/e2e/agentic-research.spec.ts`.

## How the Composable Wires to frontend-core

The old SFC had inline `fetch()` calls duplicating logic already in `@aifolio/frontend-core/agentic-research`. The extraction:

1. `api/agenticResearchApi.ts` wraps `fetchAgenticDatasetManifest`, `fetchAgenticSklearnTools`, `fetchAgenticDatasetRows` — injecting the Nuxt proxy base URL (`/api/ai`)
2. `model/useAgenticResearch.ts` calls the api layer, owns the refs, and exposes `init()`, `onDatasetChange()`, `onDatasetWatch()`, computed `toolGroups`
3. The SFC destructures the composable return and wires it to the template

This means both the Vue and Next.js apps share the same validated logic from `frontend-core` — the only difference is the thin framework-specific adapter layer.
