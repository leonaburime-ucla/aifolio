# Progress Ledger: FEAT-001-multi-framework-frontend-core

## Current Objective
Duplicate cleanup pass complete for chat, ML training, recharts, agentic-research, AG-UI, and AI API config. Reusable logic and shared contracts now live in `web/packages/`; Next feature folders retain React, browser, Next API, CopilotKit, DOM, and runtime adapters only. Next test type drift and runtime regressions from the extraction have been repaired.

## Completed Phases

### Phase 1: Contracts Package (complete)
- Created `web/packages/contracts/` with entity types + Zod schemas
- Entities: chart (ChartSpec, ChartActionsPort, ChartSpecSchema), chat (28+ types + 6 schemas + api.types)
- 23 contract schema tests passing

### Phase 2: Frontend-Core Package (complete)
- Created `web/packages/frontend-core/` with pure logic functions
- Features: model-selection, chat-submission, chat-normalization, chat-composition, chat-store, chat-orchestrator
- 60 pure function tests passing

### Phase 3: Next.js Brownfield Wiring (complete)
- Added `@aifolio/contracts` and `@aifolio/frontend-core` as `file:` deps in Next.js
- Configured tsconfig path aliases for both packages
- Replaced 11 files with re-export shims (6 logic + 5 type files)
- All 89 Next.js ai-chat tests pass through shims unchanged

### Phase 4: Remove Re-Export Shims for Chat (complete)
- Updated all 14+ ChartSpec consumers + ai-chat React layer to import from packages directly

### Phase 5: ESLint Chat Boundary Rules (complete)
- Added `no-restricted-imports` for `@/features/ai-chat/logic/*` and related paths

### Phase 6: Recharts + Agentic Research + ML Extraction (complete)
- Created `@aifolio/frontend-core/recharts` (2 files)
- Created `@aifolio/frontend-core/agentic-research` (7 files)
- Created `@aifolio/frontend-core/ml-training` (12 files)
- 40+ app-side import rewrites

### Phase 7: Chat/API Extraction (complete — assessed as already done)
- Existing `./chat` slice in frontend-core covers all extractable chat logic
- Remaining chat API code is inherently coupled to React hooks/state

### Phase 8: AG-UI Logic Extraction (complete)
- Created `@aifolio/frontend-core/ag-ui` (14 files)
- Config: tool name constants, route aliases, ML framework metadata
- Model: workspace tab resolution, copilot payload parsing, tool result formatting, tools catalog, context helpers, frontend tool handlers, ML form patch resolution, ML tools flow
- Lib: message persistence utilities
- 16 production consumers rewired to import from package directly
- ML display formatting also extracted (`displayFormat.ts`)

### Phase 9: Cleanup & Guardrails (complete)
- ESLint `no-restricted-imports` rules added for 12 AG-UI shim paths + 4 ML utility paths
- Circular dependency check: 0 cycles (both frontend-core and ag-ui-chat feature)
- TypeScript: compiles clean (only 2 pre-existing echarts type errors, unrelated)
- App-side `__types__` and `__specs__` directories removed from `web/nextjs/src/features`
- Specs moved to `ADS-project-knowledge/specs/001-multi-framework-frontend-core/nextjs-feature-specs/`
- Stale package-local handoff artifacts removed from `web/packages/frontend-core`

### Duplicate Cleanup Follow-Up (complete)
- Removed remaining reusable Next duplicates for ML distillation/modal helpers, ML bridge patch helpers, ML training orchestrators, AG-UI model defaults, AG-UI frontend tool action creation, and AG-UI pure frontend tool logic
- Added/updated reusable contracts for chat UI, ML training bridges, ML dataset state, training runs state, and framework training/distillation flows
- Added reusable package helpers for ML form bridge patches, training orchestration, distillation view models, training modal helpers, AG-UI model defaults, AG-UI Copilot frontend tool actions, and ML AG-UI randomization helpers
- Moved reusable chat and agentic-research API clients into `@aifolio/frontend-core`; Next `agenticResearchApi.ts` and `chatApi.ts` now only provide app-specific base URL/debug runtime wiring
- Added missing public-function docs, `@complexity`, and `@overallScore` metadata across chat API clients, ML training helpers, and recharts helpers
- Remaining `web/nextjs/src/features/ml/logic` file: `trainingRuntime.logic.ts` only; it stays because it wires `react-hot-toast`, `navigator.clipboard`, and browser scheduling
- Remaining `web/nextjs/src/features/ag-ui-chat/config` file: `copilotRuntime.config.ts` only; it stays because it wires Next runtime URLs, env config, and app agent defaults
- Remaining `web/nextjs/src/features/ag-ui-chat` files are React/CopilotKit/Next/browser adapters or thin facades over ML-owned browser adapters, not reusable framework-agnostic logic
- Extracted reusable AI API base URL resolution to `@aifolio/frontend-core/config/aiApi`; Next `src/core/config/aiApi.ts` now only injects `process.env` and browser detection.
- Fixed Next test type drift after shared contract extraction: ChartSpec fixtures, ML training result literals, optimizer UI mocks, bridge mocks, ECharts series access, toast mocks, and dataset parser/API expectations.

### External Audit (complete — partial)
- Gemini 3.1 Pro: PASS_WITH_NOTES

## Test Evidence
- `@aifolio/contracts`: 23/23 pass
- `@aifolio/frontend-core`: 75/75 pass
- Next.js full Vitest suite: 492/492 pass across 144 files
- Total verified in this pass: 590 tests green across contracts, frontend-core, and Next

## Latest Verification
- `npm run typecheck` in `web/packages/contracts`: pass
- `npm test` in `web/packages/contracts`: pass, 23 tests
- `npm run typecheck` in `web/packages/frontend-core`: pass
- `npm test` in `web/packages/frontend-core`: pass, 75 tests
- `npx tsc --noEmit --pretty false` in `web/nextjs`: pass
- `npm test` in `web/nextjs`: pass, 492 tests across 144 files
- `find web/nextjs/src/features -type d \( -name '__types__' -o -name '__specs__' \)`: no results
- `find web/nextjs/src/features -type d -empty`: no results

## Architecture Summary

```
@aifolio/contracts          — types + Zod schemas (shared)
@aifolio/frontend-core      — pure logic (5 slice exports)
  ./chat                    — chat submission, normalization, composition, store
  ./recharts                — chart options, formatting
  ./agentic-research        — dataset/chart/tool logic
  ./ml-training             — training validation, sweep, optimizer, display
  ./ag-ui                   — workspace, payload, tools, persistence
web/nextjs                  — React app (consumes packages)
```

## Known Risks
- Circular type import `chat/index.ts` ↔ `chat/api.types.ts` (safe for type-only)
- tsconfig `paths` duplicates what package.json `exports` provide
- `web/nextjs/src/__tests__/ui/components/Datatable/DataTable.unit.test.tsx` is very slow in the full suite (~38s observed during the latest run)
- Moved ADS specs and older graph/report artifacts can still mention old feature-local paths as historical references

## Failure Clusters
No active failure cluster after the latest verification.
