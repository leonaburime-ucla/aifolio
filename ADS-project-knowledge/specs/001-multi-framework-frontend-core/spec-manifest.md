# Spec Manifest: Multi-Framework Frontend Core Extraction

---

## Header Metadata

| Field | Value |
|-------|-------|
| spec_id | SPEC-001 |
| feature_name | FEAT-001-multi-framework-frontend-core |
| version | 1.0.0 |
| last_edited | 2026-06-09T17:00:00Z |
| spec_naming | standard |
| spec_root | ADS-project-knowledge/specs/001-multi-framework-frontend-core/ |
| spec_entrypoint | feature.spec.md |
| spec_readiness_artifact | spec-dod.md |

---

## Package Applicability Matrix

| Logical File | Status (`PRESENT|OMITTED`) | Actual Filename | Why Present / Why Omitted |
|---|---|---|---|
| `feature.spec.md` | PRESENT | `feature.spec.md` | Canonical primary requirements spec |
| `api.spec.md` | PRESENT | `api.spec.md` | Extracted packages expose programmatic APIs (TypeScript module exports, function contracts, type definitions) consumed by multiple framework apps |
| `state.spec.md` | PRESENT | `state.spec.md` | State portability contracts define observable state shape and transitions that all framework state managers must satisfy |
| `orchestrator.spec.md` | OMITTED | -- | Orchestration remains in framework-specific apps (React hooks, Vue composables). No shared orchestrator is extracted. Backend orchestration is out of scope. |
| `ui.spec.md` | PRESENT | `ui.spec.md` | Behavioral parity requirements define observable UI behavior each framework app must satisfy |
| `errors.spec.md` | OMITTED | -- | No new error codes defined. Existing backend error envelope format is preserved unchanged. Frontend validation errors use shared logic that returns typed results, not error codes. Cite existing format at `backend/shared/errors/` rather than respeccing. |
| `behavior.spec.md` | PRESENT | `behavior.spec.md` | Migration sequencing, extraction ordering, verification gates, and rollback rules require deterministic behavior specification |
| `traceability.spec.md` | PRESENT | `traceability.spec.md` | Seeds REQ/AC/INV/EC coverage mapping before TDD |
| `spec-manifest.md` | PRESENT | `spec-manifest.md` | Required package index for downstream stages |
| `spec-dod.md` | PRESENT | `spec-dod.md` | Readiness gate and quality proof |

---

## Stage Read Set

| Stage | Must Read |
|---|---|
| `architect` | `feature.spec.md`, `api.spec.md`, `state.spec.md`, `ui.spec.md`, `behavior.spec.md`, `traceability.spec.md`, `spec-dod.md` |
| `tdd` | `feature.spec.md`, `api.spec.md`, `state.spec.md`, `ui.spec.md`, `behavior.spec.md`, `traceability.spec.md`, `spec-dod.md`, ADR, tasks |
| `programmer` | `feature.spec.md`, `api.spec.md`, `state.spec.md`, `ui.spec.md`, `behavior.spec.md`, `traceability.spec.md`, ADR, certified tests |

---

## Brownfield / Reverse-Spec References

| Evidence / Touchpoint | Type | Why It Matters |
|---|---|---|
| `ADS-project-knowledge/reports/codebase-analysis/ANALYSIS-aifolio-2026-06-09.md` | codebase-analysis | Documents 481 cross-feature imports, missing public APIs (FLAW-002), god orchestrator (FLAW-003), and layer vocabulary drift (FLAW-004) that drive extraction requirements |
| `ADS-project-knowledge/reports/graphify-out/nextjs/graph.json` | codebase-analysis | AST-level dependency graph for frontend; enables mechanical verification of import boundaries post-extraction |
| `ADS-project-knowledge/.local-artifacts/handoff/20260609T161653Z-handoff.md` | source touchpoint | Contains extraction map, architecture direction, candidate module inventory, and migration phase sequence |
| `web/nextjs/src/features/charts/contracts/chart.types.ts` | source touchpoint | Existing ChartSpec and ChartActionsPort -- primary extraction candidate for contracts package |
| `web/nextjs/src/features/ai-chat/__types__/chat.types.ts` | source touchpoint | Chat domain types (ChatMessage, ChatState, ChatActions, etc.) -- extraction candidates |
| `web/nextjs/src/features/ai-chat/logic/modelSelection.logic.ts` | source touchpoint | Pure model selection logic -- extraction candidate for frontend-core |
| `web/nextjs/src/features/ml/__types__/mlData.types.ts` | source touchpoint | ML dataset types -- extraction candidates |
| `web/nextjs/src/features/ml/logic/trainingInputValidation.logic.ts` | source touchpoint | Training validation logic -- extraction candidate for frontend-core |
| `web/nextjs/src/features/recharts/logic/chartFormatting.logic.ts` | source touchpoint | Chart formatting logic -- extraction candidate for frontend-core |
| `web/nextjs/src/features/ag-ui-chat/logic/copilotAssistantPayload.util.ts` | source touchpoint | AG-UI payload normalization -- extraction candidate for frontend-core |

---

## Validation Notes

- Validator last run: 2026-06-09T17:00:00Z
- Validator result: PASS
- Validator manual waiver: N/A
- Canonical hash verified at: 2026-06-09T17:00:00Z
- Notes: Hash computed and updated by validator. sha256:5ac6301bec0e51192f30229309d6327185c39c3b87025927f84cee8660d9299d
