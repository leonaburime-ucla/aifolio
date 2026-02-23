# AI Chat Feature Spec

Spec ID: `ai-chat`
Version: `1.2.0`
Status: Draft
Last updated: `2026-02-23`

## Overview

Defines the Orc-BASH contract for chat orchestration in the AI feature. This version introduces explicit adapter boundaries so orchestrators do not import concrete Zustand stores.

## In Scope

- Chat orchestrators (`chat`, `landing`, `agentic-research`)
- Adapter ports for chat state and chart write actions
- Message and model loading behaviors

## Requirements

### ORCH-1 Adapter Boundary
- ORCH-1.1 Orchestrators MUST consume chat state through adapter ports.
- ORCH-1.2 Orchestrators MUST NOT import `state/zustand/*` files directly.
- ORCH-1.3 Orchestrators MUST NOT call `Store.getState()` directly.

### ORCH-2 Chart Sync Contract
- ORCH-2.1 Assistant chart payloads MUST be forwarded via a chart actions port.
- ORCH-2.2 Array chart payloads MUST be added in-order.

### ORCH-3 Dataset Context Contract
- ORCH-3.1 Research chat orchestrators MUST derive `activeDatasetId` from the Agentic Research state port.
- ORCH-3.2 Landing chat orchestrator MUST set `activeDatasetId` to `null`.

### ORCH-4 API Contract
- ORCH-4.1 Orchestrators inject API dependencies (`sendMessage`, `fetchModels`) through `ChatApiDeps`.
- ORCH-4.2 API calls are delegated to feature API modules, not embedded in orchestrator logic.

## Acceptance Criteria

- AC-1 `chatOrchestrator`, `landingChatOrchestrator`, `agenticResearchChatOrchestrator` compile without direct Zustand store imports.
- AC-2 Existing chat flow behavior remains unchanged (messages/models/chart sync).
- AC-3 Adapter ports are colocated under feature state adapter directories.
