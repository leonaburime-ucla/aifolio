# Handoff: Multi-Framework Frontend Core Extraction

**Date:** 2026-06-09  
**Branch:** main (all changes unstaged/uncommitted)  
**Last Agent:** Claude Opus 4.6  
**Pipeline Stage:** Programmer → Phase 4 COMPLETE, Phase 5 + audit fixes pending

---

## What Was Done

### Phase 4: Remove Re-export Shims, Direct Imports

All consumers in `web/nextjs/` now import directly from shared packages instead of through local re-export shims.

**Import path migrations completed:**

| Old Path | New Path | Files Updated |
|----------|----------|---------------|
| `@/features/charts/contracts/chart.types` | `@aifolio/contracts/entities/chart` | ~30 (source + tests) |
| `@/features/ai-chat/__types__/chat.types` (contracts types) | `@aifolio/contracts/entities/chat` | ~25 |
| `@/features/ai-chat/__types__/api.types` | `@aifolio/contracts/entities/chat/api` | 1 (chatApi.ts) |
| `@/features/ai-chat/__types__/uiFeedback.types` | `@aifolio/contracts/entities/chat` | 4 |
| `@/features/ai-chat/logic/chatStore.logic` | `@aifolio/frontend-core/features/chat-store` | ~4 |
| `@/features/ai-chat/logic/chatSubmission.logic` | `@aifolio/frontend-core/features/chat-submission` | ~5 |
| `@/features/ai-chat/logic/modelSelection.logic` | `@aifolio/frontend-core/features/model-selection` | ~5 |
| `@/features/ai-chat/logic/chatComposition.logic` | `@aifolio/frontend-core/features/chat-composition` | 1 |
| `@/features/ai-chat/logic/chatOrchestrator.logic` | `@aifolio/frontend-core/features/chat-orchestrator` | 1 |
| `@/features/ai-chat/logic/chatApiNormalization.logic` | `@aifolio/frontend-core/features/chat-normalization` | 1 |

**Files deleted (dead shims):**
- `src/features/ai-chat/__types__/api.types.ts`
- `src/features/ai-chat/__types__/uiFeedback.types.ts`
- `src/features/ai-chat/__types__/logic/` (entire directory)
- `src/features/ai-chat/logic/` (entire directory — 6 files)
- `src/features/charts/contracts/chart.types.ts`

**File kept (contains UI-only types that cannot move to contracts):**
- `src/features/ai-chat/__types__/chat.types.ts` — now contains ONLY: `ChatUiState`, `ChatActions`, `ChatIntegration`, `UseChatStatePort`, `UseChatChartActionsPort`

**Test rewritten:**
- `src/__tests__/features/ai-chat/integration/req-005.contract-location.integration.test.ts` — now asserts new package structure instead of old shim existence

**Test status:** The specific contract-location test passed 4/4 inline. Full suite (494 tests) could not be confirmed due to `/private/tmp` ramdisk at 0 bytes — **first action for next session should be verifying the full suite.**

---

## What Remains

### Immediate (fix before merge)

1. **Verify full test suite passes** — run `npx vitest run` and confirm 145 files / 494 tests all green

2. **Fix High #1: React hook-order violation**
   - File: `web/nextjs/src/features/ai-chat/react/compositions/useChatSurface.orchestrator.ts:68`
   - Problem: `useActiveDatasetId()` is called conditionally (`activeDatasetId === undefined`)
   - Fix: Always call the hook unconditionally, then use the value conditionally:
     ```typescript
     const hookDatasetId = useActiveDatasetId();
     const resolvedActiveDatasetId = activeDatasetId === undefined ? hookDatasetId : activeDatasetId;
     ```

3. **Fix High #2: `.ts` extension imports break tsc in consumer context**
   - Files: `web/packages/contracts/src/entities/chat/index.ts:1-2`, `web/packages/contracts/src/entities/chat/api.types.ts:1-2`
   - Problem: `import { z } from "zod"; import { ChartSpecSchema } from "../chart/index.ts"` — the `.ts` suffix requires `allowImportingTsExtensions` which Next.js tsconfig doesn't enable
   - Fix: Remove `.ts` extensions from imports within the contracts package (use `"../chart/index"` and `"./api.types"` and `"./index"`)

4. **Fix Medium #4: Runtime contract validation incomplete**
   - File: `web/packages/frontend-core/src/features/chat-normalization/index.ts`
   - Problem: JSON is cast to `ChartSpec` without Zod validation at line ~47 and `chartSpec` forwarded without schema check at ~98
   - Fix: Add `ChatAssistantPayloadSchema.safeParse()` or `ChartSpecSchema.safeParse()` at the parse boundary

5. **Fix Medium #5: Test uses invalid ChartSpec shape**
   - File: `src/__tests__/features/ai-chat/unit/req-003.chart-fanout.unit.test.ts`
   - Problem: Test fixtures create `{ chartType, title, data }` objects cast as `ChartSpec`, missing required `id`, `type`, `xKey`, `yKeys`
   - Fix: Update test fixtures to match actual `ChartSpec` schema

### Phase 5 (after fixes)

6. **ESLint `no-restricted-imports` rules** — enforce INV-01 and INV-02:
   - Disallow `@/features/ai-chat/logic/` imports
   - Disallow `@/features/ai-chat/__types__/api.types` imports  
   - Disallow `@/features/ai-chat/__types__/logic/` imports
   - Disallow `@/features/charts/contracts/chart.types` imports

### Programmer Agent Output (retroactive)

7. **Architecture Audit** — brief pass over dependency arrows, confirm no cycles
8. **Pre-Completion Checklist** — all items from the programmer agent spec
9. **Progress-ledger update** — update `progress-ledger.md` with Phase 4 completion

---

## Architecture Context

```
web/
├── packages/
│   ├── contracts/          # @aifolio/contracts — framework-agnostic types + Zod schemas
│   │   └── src/entities/
│   │       ├── chart/index.ts      (ChartSpec, ChartActionsPort, ChartSpecSchema)
│   │       └── chat/
│   │           ├── index.ts        (30+ types, 6 Zod schemas, ChatDeps/ChatApiDeps/ChatLogicDeps)
│   │           └── api.types.ts    (transport-layer types: Send/Fetch inputs/options/results)
│   └── frontend-core/     # @aifolio/frontend-core — framework-agnostic pure logic
│       └── src/features/
│           ├── chat-composition/     (mapChatStateWithDataset, composeChatStateActions, createOnMessageReceived)
│           ├── chat-normalization/   (normalizeChatApiResult, parseJsonPayload, createModelFetchErrorResult)
│           ├── chat-orchestrator/    (createChatApiDeps, createChatDeps)
│           ├── chat-store/           (createInitialChatStoreCoreState, appendMessage, resolveHistoryCursor)
│           ├── chat-submission/      (normalizeSubmissionValue, buildChatHistoryWindow, create*ChatMessage, shouldRestoreDraftValue)
│           └── model-selection/      (FALLBACK_CHAT_MODELS, resolveFallbackModelSelection, resolveFetchedModelSelection)
└── nextjs/                 # React/Next.js app — consumes shared packages
    ├── package.json        (has `file:` deps to ../packages/*)
    ├── tsconfig.json       (has `paths` for @aifolio/* resolution)
    └── src/features/ai-chat/
        ├── __types__/chat.types.ts   (UI-ONLY: ChatUiState, ChatActions, ChatIntegration, UseChatStatePort, UseChatChartActionsPort)
        ├── api/                      (chatApi.ts, chatApi.adapter.ts — HTTP transport)
        └── react/                    (hooks, orchestrators, state adapters — React-specific)
```

**Key decisions:**
- `file:` references (not `workspace:*`) because npm, not pnpm
- `allowImportingTsExtensions` + `noEmit` in package tsconfigs (no build step yet — #3 in audit)
- tsconfig `paths` for resolution (not `references`, which conflicts with `noEmit`)
- UI-only types stay in the app, not in contracts (they reference React state patterns)

---

## Question from User (pending)

User asked: "Is `chatApi.ts` in the right place? Seems like it's reusable."

Answer provided: Yes, it's reusable. The function signatures already accept `runtimeDeps` with `resolveBaseUrl`, so the only coupling is the default `getAiApiBaseUrl()` import. Recommendation: move to `@aifolio/frontend-core` with base URL injected, keep the default in the Next.js adapter layer. But not urgent — do it when a second framework consumer needs it.

---

## Environment Notes

- `/private/tmp` ramdisk fills to 0 bytes during long sessions — bash output gets lost
- Workaround: redirect output to project dir, or run shorter targeted test commands
- The `codex exec --full-auto` transport is unreliable on this repo (zero-byte output after 5+ min). Use separate Codex sessions manually.
