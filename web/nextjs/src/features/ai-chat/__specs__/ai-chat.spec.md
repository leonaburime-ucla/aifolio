# Feature Spec: AI Chat

Spec ID: `ai-chat`
Version: `1.8.0`
Status: `draft`
Last updated: `2026-03-02`

## Scope

In scope:
- Reusable chat feature behavior independent of page/screen context.
- Message submit pipeline (input normalization, history shaping, assistant response handling).
- Model option loading with deterministic fallback behavior.
- Chart payload fan-out from assistant responses into chart action ports.
- Type contracts under `features/ai-chat/__types__/typescript`.

Out of scope:
- Route/page composition and cross-feature orchestration (owned by `screens/*`).
- Backend implementation details for `/chat`, `/chat-research`, `/llm/gemini-models`.
- Chart rendering internals (`recharts`, `echarts`) and AG-UI Copilot runtime behavior.
- Streaming partial assistant tokens.
- Message edit/delete/regenerate operations.
- Conversation persistence across browser sessions.

## Architecture Boundaries

- AB-001: `features/ai-chat` MUST NOT import domain state/orchestrators from other feature folders.
- AB-002: Screen-level context injection (dataset, route mode, destination stores) MUST be done in `screens/*` modules.
- AB-003: `typescript/logic/*` MUST remain framework-agnostic and side-effect-light except explicit I/O adapters.

## Requirement Set

- REQ-001: Submit behavior MUST append the user message before starting API send and set `isSending=true` during flight.
- REQ-002: Submit behavior MUST always set `isSending=false` after completion (success, null response, or failure).
- REQ-003: Assistant payload with `chartSpec` MUST call `addChartSpec` once per chart in deterministic order.
- REQ-004: Model-loading behavior MUST select in this order: existing selected model, API `currentModel`, first API model, fallback first model.
- REQ-005: AI chat contracts (chat/api/chart + logic input/output types) MUST live under `features/ai-chat/__types__/typescript`.
- REQ-006: Core feature orchestrator (`chatOrchestrator`) MUST default `activeDatasetId` to `null` and remain page-agnostic.
- REQ-007: In-flight requests MUST be abortable on unmount/navigation, and aborted requests MUST NOT mutate state afterward.
- REQ-008: Attachment handling MUST define deterministic acceptance rules (allowed type policy, max size, max count) and reject invalid files with structured non-crashing outcomes.
- REQ-009: Timeout behavior MUST be explicit per endpoint; timeout outcomes MUST map to defined error contracts.

## Deterministic Rules

- DR-001: Input normalization trims whitespace; empty post-trim input MUST short-circuit with no state mutation.
- DR-002: History payload window MUST include at most 10 recent entries including the current user message.
- DR-003: History cursor navigation MUST stay within bounds and never throw.
- DR-004: Invalid assistant payload normalization MUST yield `null` (not throw).
- DR-005: Fallback model list order MUST remain stable unless explicitly changed by spec version bump.

## Acceptance Scenarios

- AC-001 (REQ-001): Given non-empty input, when submit starts, then user message appears in transcript before API send begins.
- AC-002 (REQ-002): Given any API outcome (ok/null/error), `isSending` is false at completion and attachments are cleared.
- AC-003 (REQ-003): Given assistant payload with `chartSpec=[A,B,C]`, `addChartSpec` is called in order A,B,C exactly once each.
- AC-004 (REQ-004): Given no selected model and models response `{currentModel:null, models:[M1,M2]}`, selected model becomes `M1`.
- AC-005 (REQ-005): Given ai-chat module imports contract types, they resolve only from `@/features/ai-chat/__types__/typescript/*`.
- AC-006 (REQ-006): Given default chat orchestrator initialization, `activeDatasetId` is `null` unless injected by screen composition.
- AC-007 (REQ-007): Given unmount/navigation during request, no post-abort state update occurs.
- AC-008 (REQ-008): Given invalid attachment input, chat UI remains usable and invalid files are rejected predictably.
- AC-009 (REQ-009): Given endpoint timeout, UI receives deterministic timeout behavior per error contract.

## Open Clarifications

- None.
