# Copilot Chat Feature Spec

Spec ID: `copilot-chat`
Version: `1.0.0`
Status: Draft
Last updated: `2026-02-23`

## Overview

Defines Orc-BASH contracts for AG-UI/legacy Copilot chat behavior used on `/ag-ui`.

## In Scope

- Sidebar mode switching (`legacy` vs `ag-ui`)
- Frontend tool-call orchestration wiring
- Runtime adapter/provider boundaries
- Assistant message rendering variants

## Requirements

### CP-UI-1 Sidebar Contract
- CP-UI-1.1 Sidebar MUST support `legacy` and `ag-ui` modes.
- CP-UI-1.2 `ag-ui` mode MUST render frontend tools bridge.
- CP-UI-1.3 Assistant renderer MUST switch by selected mode.

### CP-ORCH-1 Frontend Tools Contract
- CP-ORCH-1.1 Frontend tools orchestration MUST be isolated in feature orchestrators.
- CP-ORCH-1.2 Tool call UI effects MUST be deterministic for repeated runs.

### CP-STATE-1 Persistence Contract
- CP-STATE-1.1 Message persistence state MUST remain feature-scoped.
- CP-STATE-1.2 Runtime/provider setup MUST be outside UI view components.

## Acceptance Criteria

- AC-1 `/ag-ui` page works with Copilot sidebar mode `ag-ui`.
- AC-2 Sidebar can be switched to legacy mode without runtime errors.
- AC-3 Frontend tools wiring stays decoupled from page components.
