# Handoff: FSD Restructuring + Extraction Continuation

**Date:** 2026-06-09  
**Branch:** main (all changes unstaged/uncommitted)  
**Last Agent:** Claude Opus 4.6 (Coordinator + Programmer)  
**Pipeline Stage:** Phase 5 complete, Phase 2-9 extraction pending

---

## What Was Done This Session

### 1. Audit Fixes (from prior handoff)
- **High #1:** Fixed conditional React hook call in `useChatSurface.orchestrator.ts:68` — `useActiveDatasetId()` now called unconditionally
- **High #2:** Removed `.ts` extensions from imports in contracts + frontend-core barrel exports
- **Medium #4:** Added `ChartSpecSchema` Zod validation at parse boundary in `chat-normalization/model/normalization.ts`
- **Medium #5:** Fixed test fixtures in `req-003.chart-fanout.unit.test.ts` to use correct `ChartSpec` shape
- **Test fix:** Repointed `ab-003` test from deleted `logic/` dir to `../packages/frontend-core/src/chat`

### 2. FSD Restructuring (Phase 0 + 1)
- Split contracts into FSD segments: `entities/chart/model/{types,schema}.ts`, `entities/chat/model/{types,schema}.ts`, `entities/chat/api/types.ts`
- Restructured frontend-core: consolidated 6 separate `chat-*` directories into single `chat/` vertical slice with `model/` segment
- Renamed `src/features/chat/` → `src/chat/` (packages are libraries, not FSD apps — no layer prefixes)

### 3. ESLint Boundary Rules (Phase 5 from prior plan)
- Added `no-restricted-imports` to `web/nextjs/eslint.config.mjs` enforcing old shim paths are banned

### 4. AI-Dev-Shop-speckit Updated
- Pulled latest from origin (`75d012f..7a1c676`)

### 5. Swarm Consensus Debate: Package Architecture
- **Result:** Unanimous Option D (all 3 models: Claude Opus 4.6, Gemini 3.1 Pro, GPT-5.5)
- **Decision:** Two packages, FSD-influenced internal structure, no FSD layers inside packages
- Context packet: `ADS-project-knowledge/.local-artifacts/swarm-consensus/context/CTX-fsd-package-structure-2026-06-09.md`

---

## Current Package Structure

```
web/packages/contracts/src/
  entities/
    chart/
      model/types.ts          (ChartSpec, ChartActionsPort)
      model/schema.ts         (ChartSpecSchema — Zod)
      index.ts                (barrel)
    chat/
      model/types.ts          (30+ domain types)
      model/schema.ts         (6 Zod schemas)
      api/types.ts            (transport types)
      index.ts                (barrel)
  index.ts

web/packages/frontend-core/src/
  chat/
    model/
      composition.ts          (mapChatStateWithDataset, composeChatStateActions, createOnMessageReceived)
      normalization.ts        (normalizeChatApiResult, parseJsonPayload, validateChartSpec)
      orchestrator.ts         (createChatApiDeps, createChatDeps)
      store.ts                (createInitialChatStoreCoreState, appendMessage, resolveHistoryCursor)
      submission.ts           (normalizeSubmissionValue, buildChatHistoryWindow, create*ChatMessage)
      model-selection.ts      (FALLBACK_CHAT_MODELS, resolveFallbackModelSelection, resolveFetchedModelSelection)
    index.ts                  (barrel re-exporting all model/* files)
  shared/index.ts             (empty stub)
  index.ts
```

---

## Architecture Decisions (Settled)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package count | 2 (contracts + frontend-core) | Types separate from logic for tree-shaking, independent consumption |
| FSD inside packages | Segments only (model/, api/, lib/, config/) — no layers | FSD says don't apply layers to libraries; app owns layer assignment |
| Slice naming | `src/<domain>/` not `src/features/<domain>/` | No layer prefix — packages don't declare which FSD layer they are |
| Import direction | apps → frontend-core → contracts (never reverse) | Enforceable via ESLint |
| Public API | Each slice has `index.ts` barrel | No deep imports from outside the slice |
| CopilotKit dep | Split: keep framework fn in app, extract pure logic | features layer must not depend on UI libs |
| papaparse/xlsx | Peer deps in frontend-core | Loose coupling; consumer provides version |
| echarts | Peer dep in frontend-core | Same reasoning |
| Entity cross-ref (chat→chart) | Keep as-is | Domain-correct @x reference |
| ML ai/agUi files | Extract pure ones, keep DOM-bound in app | Clean split on framework boundary |
| Test migration | Unit tests move to package, integration tests stay in app | Co-locate tests with source |

---

## What Remains (Phases 2-9)

### Phase 2: Extract Entity Types to `@aifolio/contracts`
- Create `entities/ml-training/`, `entities/agentic-research/`, `entities/ag-ui/`, `entities/recharts/` in contracts
- Move framework-agnostic type definitions from Next.js `__types__/` directories
- Add `package.json` exports entries, update tsconfig paths, rewrite consumer imports

### Phase 3: Extract `shared/` Layer to `frontend-core`
- Move `src/core/config/aiApi.ts` → `shared/config/aiApi.ts`
- Move `src/features/ml/utils/displayFormat.util.ts` → `shared/lib/displayFormat.ts`
- Refactor `aiApi.ts` into a factory: `createBaseUrlResolver(env, isServer)` — pure packages shouldn't read `process.env`

### Phase 4: Extract `recharts/` Slice
- `src/recharts/model/echartsOptions.ts` + `chartFormatting.ts`
- Add `echarts` as peer dep (already done in package.json)

### Phase 5: Extract `agentic-research/` Slice
- 5 logic files → `model/`
- `datatable.util.ts` → `lib/`
- `papaparse` + `xlsx` as peer deps (already in package.json)

### Phase 6: Extract `ml-training/` Slice (largest batch)
- 6 logic → `model/`
- 4 utils → `lib/`
- 2 validators → `lib/`
- 2 configs → `config/`

### Phase 7: Extract `chat/api/` Segment
- Move `chatApi.ts` + `chatApi.adapter.ts` into `frontend-core/src/chat/api/`
- Depends on `shared/config/aiApi` (Phase 3) and `chat/model/normalization` (already there)

### Phase 8: Extract `ag-ui/` Slice
- 7 pure logic files → `model/`
- Split `copilotAssistantPayload.util.ts`: pure functions to package, CopilotKit wrapper stays in app
- `copilotFrontendToolActions.logic.ts` + `copilotFrontendToolsFlow.logic.ts` STAY (React/window deps)

### Phase 9: Cleanup
- Clean barrel exports (explicit named exports, no wildcards — per FSD public API rules)
- Add ESLint rules for all new boundaries
- Check for circular deps
- Update `package.json` exports for all new slices

---

## Files Changed This Session (Unstaged)

Key modifications:
- `web/packages/contracts/src/entities/chart/model/{types,schema}.ts` (new — split from index)
- `web/packages/contracts/src/entities/chat/model/{types,schema}.ts` (new — split from index)
- `web/packages/contracts/src/entities/chat/api/types.ts` (moved from api.types.ts)
- `web/packages/frontend-core/src/chat/` (consolidated from 6 dirs)
- `web/packages/frontend-core/package.json` (exports updated)
- `web/nextjs/eslint.config.mjs` (no-restricted-imports added)
- `web/nextjs/src/features/ai-chat/react/compositions/useChatSurface.orchestrator.ts` (hook fix)
- ~16 Next.js files with import path updates

---

## Test Status

- **frontend-core:** 5 files, 60 tests — ALL PASS
- **Next.js:** 145 files, 494 tests — ALL PASS
- **TypeScript:** Zero errors in contracts/frontend-core (pre-existing errors in AG-UI/Agentic Research test files remain)

---

## Environment Notes

- `/private/tmp` ramdisk fills to 0 bytes during long sessions
- Codex CLI: use `codex exec -m gpt-5.5 -s read-only` with stdin for non-interactive dispatch (no `-p` flag, no `--reasoning-effort`)
- Gemini CLI: `gemini --model gemini-3.1-pro-preview -p "..."` works with piped stdin
