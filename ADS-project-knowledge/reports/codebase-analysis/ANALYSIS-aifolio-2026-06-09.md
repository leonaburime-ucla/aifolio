# Codebase Analysis: AIfolio

- Analysis ID: ANALYSIS-aifolio
- Date: 2026-06-09T00:20:22Z
- Analyst: CodeBase Analyzer Agent
- Parts: 1 of 1
- Graphify graph: `ADS-project-knowledge/reports/graphify-out/AIfolio/graph.json`

## Executive Summary

- Language/Framework: Next.js 16 + React 19 + TypeScript frontend; FastAPI + Python ML/agent backend.
- Apparent Pattern Intent: Frontend feature-oriented architecture moving toward Feature-Sliced Design; backend layered/modular service architecture with emerging ports/adapters.
- Graphify Result: 6,675 nodes and 11,387 edges from 702 structurally extracted files.
- Files Sampled: 28 focused files/commands plus Graphify graph summaries; not an exhaustive read.
- Severity Counts: Critical: 0 | High: 3 | Medium: 5 | Low: 1
- Current State Classification: Frontend Layered/Feature-Oriented (degraded); Backend Layered (partially healthy, with route aggregation hotspots).
- Recommended Target: FSD-lite with vertical domain slices for the frontend; keep backend as modular monolith with clearer application/service boundaries.

## Sampling Notice

Files sampled:
- Graphify: `graphify update . --force --no-cluster`, plus structured graph summaries by source path/import edges.
- Frontend: `web/nextjs/package.json`, `web/nextjs/eslint.config.mjs`, `web/nextjs/src/app/**/{page,layout,route}.tsx?`, selected `src/ui/*`, selected `src/features/*`, especially AG-UI, chart, chat, ML, and agentic-research hotspots.
- Backend: `backend/requirements*.txt`, `backend/pytest.ini`, `backend/server/app.py`, `backend/server/routes/core.py`, `backend/server/routes/ml_framework.py`, `backend/application/agui/service.py`, `backend/ml/core/request_prep.py`.

Files excluded:
- Full source tree was not read line-by-line.
- Tests were counted/sampled by structure and configuration only; no test suite was run.
- `AI-Dev-Shop-speckit/`, dependency folders, build outputs, and generated caches were not analyzed as project source.
- Graphify semantic extraction was not run; graph evidence is AST/structure only.

Confidence levels by finding category:
- Architecture structure: High
- Dependency direction: High for frontend, Medium for backend
- Test coverage signal: Medium
- Security surface: Low
- Code quality indicators: Medium

Note: Confidence reflects sample coverage, not model certainty. A High-confidence finding means the sample was broad enough to support the conclusion. A Low-confidence finding is a hypothesis requiring human verification.

## Findings

### FLAW-001
- Severity: High
- Category: Frontend Architecture Boundary Violation
- Location: `web/nextjs/src/features/*`

The frontend has significant same-layer feature coupling. Static import scanning found 481 absolute `features -> features` imports and 57 production cross-feature imports between different feature slices. FSD forbids feature slices importing other feature slices directly.

Evidence:
- `ag-ui-chat -> ml`: 14 imports, e.g. `features/ag-ui-chat/__types__/logic/copilotFrontendToolsFlow.types.ts -> @/features/ml/__types__/ai/agUi/mlTrainingTooling.types`
- `ag-ui-chat -> agentic-research`: 6 imports, e.g. `features/ag-ui-chat/api/hooks/useAgUiCopilotReadableContext.hooks.ts -> @/features/agentic-research/react/state/adapters/agenticResearchState.adapter`
- `ag-ui-chat -> recharts`: 4 imports, e.g. `features/ag-ui-chat/api/hooks/useCopilotFrontendTools.hooks.ts -> @/features/recharts/react/ai/state/adapters/chartActions.adapter`
- `agentic-research -> ag-ui-chat`: `features/agentic-research/ai/tools/chartTools.ts -> @/features/ag-ui-chat/logic/copilotAssistantPayload.util`

Impact: A direct folder move to FSD would preserve hidden feature coupling and likely fail once import boundaries are enforced.

---

### FLAW-002
- Severity: High
- Category: Missing Public API Contracts
- Location: `web/nextjs/src/features/*`

No current feature slice has a root `index.ts` or `index.tsx`. Consumers import deep internal paths such as `react/state/zustand`, `react/views/components`, `logic`, and `__types__`.

Evidence: root index scan found zero public API files for `ag-ui-chat`, `agentic-research`, `ai-chat`, `charts`, `ml`, `ml-model-ui`, and `recharts`.

Impact: FSD migration cannot be safely enforced until cross-slice contracts are explicit. Without public APIs, every move is a fragile global import rewrite.

---

### FLAW-003
- Severity: High
- Category: God Orchestrator / Composition Boundary
- Location: `web/nextjs/src/features/ag-ui-chat/api/hooks/useCopilotFrontendTools.hooks.ts`

AG-UI frontend tool registration is the largest frontend import hotspot in the graph: 69 extracted import edges. It registers Copilot actions while reaching into chart, agentic-research, AG-UI workspace, and ML internals.

Evidence:
- Imports chart adapters from `@/features/recharts/...`
- Imports agentic-research adapters from `@/features/agentic-research/...`
- Imports ML framework metadata/types from `@/features/ml/...`
- Owns routing side effects through `next/navigation`

Impact: In FSD, this should not remain a normal `features/ag-ui-chat` module. It is page/widget-level orchestration that composes multiple lower-level capabilities.

---

### FLAW-004
- Severity: Medium
- Category: Layer Vocabulary Drift
- Location: `web/nextjs/src/ui`, `web/nextjs/src/lib`, `web/nextjs/src/core`

The current `ui` layer mixes at least four FSD concepts:
- `ui/components/*`: shared UI primitives.
- `ui/screens/*`: page layer candidates.
- `ui/patterns/Nav/Navbar.tsx`: widget/app-shell candidate.
- `ui/providers/*`: app/provider candidate.

Evidence: `Navbar.tsx` imports AG-UI model state directly, and `ui/screens/*` composes feature screens and chat sidebars.

Impact: Moving `ui` wholesale would create incorrect FSD layers. It needs to be split by responsibility.

---

### FLAW-005
- Severity: Medium
- Category: Route Shell Inconsistency
- Location: `web/nextjs/src/app`

Several Next app routes are thin shells already, but some still contain route-level UI composition or backend proxy implementation.

Evidence:
- Thin routes: `app/page.tsx`, `app/chat/page.tsx`, `app/agentic-research/page.tsx`, `app/ml/{pytorch,tensorflow}/page.tsx`
- Non-thin routes: `app/ag-ui/page.tsx` (79 lines), `app/ml/knowledge-distillation/page.tsx` (40 lines, client state/UI), `app/api/ai/[...path]/route.ts` (122 lines proxy implementation)

Impact: FSD migration should first move page composition into `src/pages/*` and leave route files as controllers.

---

### FLAW-006
- Severity: Medium
- Category: Backend Route Aggregation
- Location: `backend/server/routes/core.py`

The FastAPI app has a clean entry point (`server/app.py`) and delegates into routers, but `server/routes/core.py` aggregates chat, AG-UI, LLM, sample data, ML data, sklearn tools, Gemini models, and LangSmith trace routes in one module.

Impact: This is manageable today because many route functions delegate to services, but it will become a coordination hotspot as domains grow. Splitting by bounded context would better match the frontend/domain direction.

---

### FLAW-007
- Severity: Medium
- Category: Backend ML Complexity Hotspots
- Location: `backend/ml`, `backend/shared/tools/sklearn_tools`

Graphify identified backend hotspots:
- `backend/shared/tools/sklearn_tools/__init__.py`: 124 graph nodes, 37 imports.
- `backend/ml/frameworks/pytorch/trainer.py`: 21 nodes, 35 imports.
- `backend/ml/frameworks/tensorflow/trainer.py`: 21 nodes, 29 imports.
- `backend/application/agui/service.py`: 39 nodes, 15 imports.

The backend ML structure is better layered than the frontend, with shared request preparation in `backend/ml/core/request_prep.py`, framework handlers, trainers, and routes. The risk is not immediate disorder; it is growth pressure around framework-specific training, AG-UI streaming, and sklearn tool catalogs.

Impact: If new ML frameworks or training modes are added, introduce narrower ports/adapters before expanding these hotspots.

---

### FLAW-008
- Severity: Medium
- Category: Missing Architecture Enforcement
- Location: `web/nextjs/eslint.config.mjs`, `web/nextjs/package.json`

Frontend linting uses the Next defaults only. There is no Steiger, no FSD boundary lint, and no `no-restricted-imports` rule protecting layer direction or public APIs.

Impact: Even after migration, the repo will regress unless import boundaries are enforced mechanically.

---

### FLAW-009
- Severity: Low
- Category: Graph Noise / Analysis Hygiene
- Location: `web/sample-data`, specs, project docs

The Graphify graph includes useful source files but also many nodes from sample JSON and spec/documentation files. This is not a code flaw, but future graph queries should scope to `web/nextjs/src` or `backend` for architecture questions.

Impact: Broad natural-language graph queries can return noisy results. Structured path/import summaries were more reliable for this audit.

## What Was Not Analyzed

- No build, lint, typecheck, or test commands were run.
- No security audit was performed.
- No frontend component behavior was verified in a browser.
- No backend runtime behavior was exercised.
- No migration plan was generated in this pass.

## Recommended Next Step

Generate a focused architecture migration plan for the frontend:

1. Add public APIs to current feature slices before moving files.
2. Extract `entities/chart` from shared `ChartSpec` usage.
3. Split `src/ui` into `shared/ui`, `pages`, `widgets`, and app providers.
4. Reclassify AG-UI workspace/tool orchestration as `widgets/ag-ui-workspace` or `pages/ag-ui`, not a normal feature.
5. Add Steiger or ESLint import-boundary rules before large folder moves.

Backend should not be reorganized as part of the frontend FSD migration. Treat backend improvements as a separate modularization effort around `server/routes/core.py`, `application/agui`, and `backend/ml` framework ports.
