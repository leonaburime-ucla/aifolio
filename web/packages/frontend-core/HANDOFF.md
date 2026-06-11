# Frontend-Core Test Restructuring — Handoff

## What Was Done

### 1. "Stays in app" comments added to all API files
All files remaining in `web/nextjs/src/features/*/api/` now have a comment above the first import explaining WHY they stay in the Next.js app (deployment-specific wiring via `getAiApiBaseUrl`/`process.env`):
- `features/ag-ui-chat/api/agUiModelApi.ts`
- `features/ag-ui-chat/config/copilotRuntime.config.ts`
- `features/ml/api/mlDataApi.ts`, `pytorchApi.ts`, `tensorflowApi.ts`
- `features/agentic-research/api/agenticResearchApi.ts`, `agenticResearchApi.adapter.ts`
- `features/ai-chat/api/chatApi.ts`, `chatApi.adapter.ts`

### 2. Tests restructured from flat → mirrored
`frontend-core/__tests__/` was a flat directory with 10 files. Now mirrors `src/`:
```
__tests__/
  ag-ui/
    lib/
      messagePersistence.test.ts
      messagePersistenceFull.test.ts   ← NEW (written, not yet verified)
    model/
      mlTrainingToolAdapter.test.ts
      workspace.test.ts
      toolsCatalog.test.ts
      toolResultPresentation.test.ts
      copilotFrontendToolActions.test.ts
      frontendTools.test.ts
      context.test.ts                  ← NEW (written, not yet verified)
      mlToolsFlow.test.ts              ← NEW (written, not yet verified)
      frontendToolsHandlers.test.ts    ← NEW (written, not yet verified)
  agentic-research/
    lib/
      datatable.test.ts
    model/
      chart.test.ts, chartLogic.test.ts, chartStore.test.ts, chartTools.test.ts
      dataset.test.ts, datasetTools.test.ts, manifest.test.ts, tools.test.ts
      + 7 unit tests (default-dataset-selection, reorder-chart-remainder, etc.)
  chat/
    fixtures/
      chatLogicDeps.fixture.ts
    model/
      composition.test.ts, normalization.test.ts, store.test.ts, submission.test.ts
      model-selection.test.ts, model-selection-req.test.ts
      chatSubmission.test.ts, chatStore.test.ts, chatApiNormalization.test.ts
      fallback-model-order.test.ts, history-window-boundary.test.ts
      history-cursor-totality.test.ts, chart-fanout.test.ts
  config/
    aiApi.test.ts
  ml-training/
    config/
      datasetTrainingDefaults.test.ts, trainingModeExplainers.test.ts
    lib/
      bayesianOptimizer.test.ts, displayFormat.test.ts, trainingRuns.test.ts
      trainingUiShared.test.ts, trainingSweep.test.ts
    model/
      formBridgePatch.test.ts, trainingOrchestrator.test.ts
      trainingModals.test.ts, trainingHookDecisions.test.ts
      trainingRunsSection.test.ts, trainingInputValidation.test.ts
      pytorchFormBridgePatch.test.ts, tensorflowFormBridgePatch.test.ts
      distillationView.test.ts
  recharts/
    model/
      chartFormatting.test.ts, echartsOptions.test.ts
```

### 3. 47 pure tests moved from `nextjs/src/__tests__/features/` → `frontend-core/__tests__/`
These all import from `@aifolio/frontend-core/*` with no `@/` (app-local) imports.

One test (`trainingShared.logic.unit.test.ts`) was moved back — it imports `@testing-library/react`.

### 4. All 55 existing tests pass (228 assertions)
Run: `npx vitest run` from `web/packages/frontend-core`

---

## What Remains

### Coverage gaps (target: >95% except types)
Current overall: **78.67%**. Low-coverage source files:

| File | Stmts | What it needs |
|------|-------|---------------|
| `src/ag-ui/lib/messagePersistence.ts` | 23% | Test written (`messagePersistenceFull.test.ts`) but not verified passing |
| `src/ag-ui/model/context.ts` | 8% | Test written (`context.test.ts`) but not verified passing |
| `src/ag-ui/model/frontendTools.ts` | 34% | Test written (`frontendToolsHandlers.test.ts`) but not verified passing |
| `src/ag-ui/model/mlToolsFlow.ts` | 0% | Test written (`mlToolsFlow.test.ts`) but not verified passing |
| `src/agentic-research/model/apiClient.ts` | 1% | Needs test — mock fetch, test all 4 functions |
| `src/chat/model/apiClient.ts` | 3% | Needs test — mock fetch, test sendChatMessage/sendChatMessageDirect/fetchChatModels |
| `src/chat/model/orchestrator.ts` | 12% | Needs test — test createChatApiDeps/createChatDeps |
| `src/ml-training/lib/trainingRuns.ts` | 41% | Partially tested, needs more branches |
| `src/shared/index.ts` | 0% | Empty file (`export {}`) — can exclude from coverage |
| `src/index.ts` | 0% | Re-export barrel — can exclude from coverage |

### Tests NOT yet verified
The 4 new test files I wrote may have import/type issues. Run:
```bash
cd web/packages/frontend-core
npx vitest run
```

### Remaining app-coupled tests (stay in nextjs)
8 files in `nextjs/src/__tests__/features/` that import both `@aifolio/frontend-core` AND `@/` paths. These correctly stay in the Next.js test suite:
- `ag-ui-chat/react/views/components/CopilotAssistantMessage.unit.test.tsx`
- `ai-chat/integration/err-002.fetch-models-fallback.integration.test.ts`
- `ai-chat/logic/chatOrchestrator.logic.unit.test.ts`
- `ai-chat/react/state/zustand/aiChatStore.unit.test.ts`
- `ai-chat/react/hooks/useChat.hooks.unit.test.tsx`
- `agentic-research/react/ai/adapters/useAgenticResearchAiSurface.unit.test.tsx`
- `ml/react/orchestrators/tensorflowTraining.orchestrator.unit.test.ts`
- `ml/react/orchestrators/pytorchTraining.orchestrator.unit.test.ts`

### Coverage config suggestion
Add to `vitest.config.ts`:
```ts
coverage: {
  include: ['src/**/*.ts'],
  exclude: ['src/index.ts', 'src/shared/index.ts', 'src/**/index.ts'],
  thresholds: { statements: 95 }
}
```

### README
Created `web/README.md` explaining the package dependency graph and why `contracts` is separate from `frontend-core`.

---

## Quick Commands

```bash
# Run all frontend-core tests
cd web/packages/frontend-core && npx vitest run

# Run with coverage
npx vitest run --coverage

# Check a specific test file
npx vitest run __tests__/ag-ui/model/context.test.ts
```
