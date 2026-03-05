# Feature Spec: AG-UI Tool Routing

- Spec ID: `ag-ui-tool-routing`
- FEAT: `004`
- Version: `1.0.0`
- Status: `draft`
- Last Edited (UTC): `2026-03-04T00:00:00Z`
- Content Hash (sha256): `TBD`

## Problem Statement

`/ag-ui` tool calls currently mix route aliases, tab switching, and page navigation behavior without a full requirement package. We need deterministic contracts so tool calls can reliably control `/`, `/agentic-research`, `/ml/pytorch`, and `/ml/tensorflow` and remain testable.

## Scope

In scope:
- Frontend tool-call contracts for route navigation and AG-UI tab switching.
- Deterministic alias resolution for `/`, `/agentic-research`, `/ml/pytorch`, `/ml/tensorflow`.
- Separation of pure tool logic from React registration surfaces.
- Testability constraints for orchestration and React bindings.

Out of scope:
- Backend model inference logic.
- Redesign of Copilot transport protocol.
- Changes to non-AG-UI feature domains except import path updates required by restructure.

## Requirements

- REQ-001: `navigate_to_page` MUST resolve aliases and permit only allowed routes.
- REQ-002: Alias resolution MUST map user synonyms to canonical routes for home, agentic research, PyTorch, and TensorFlow pages.
- REQ-003: Invalid `navigate_to_page` inputs MUST return a structured error payload with `code="INVALID_ROUTE"` and `allowedRoutes`.
- REQ-004: `switch_ag_ui_tab` MUST accept only `charts | agentic-research | pytorch | tensorflow` and return structured `INVALID_TAB` for invalid values.
- REQ-005: React tool registration components MUST delegate validation/transformation to pure logic functions so handlers are unit-testable.
- REQ-006: `/ag-ui` page composition MUST expose tool-call controls for navigation + tab switching without runtime coupling to backend implementation details.
- REQ-007: AG-UI chat message history MUST persist across route changes and return navigation (for example leaving `/ag-ui` and coming back).

## Deterministic Rules

- DR-001: Route alias input normalization is `trim().toLowerCase()` before lookup.
- DR-002: If an alias is not found and input starts with `/`, input is treated as direct route candidate; otherwise result is empty.
- DR-003: Allowed route validation uses canonical values from route alias config and is deterministic by set membership.
- DR-004: Tab resolution normalization is deterministic and case-insensitive.

## Acceptance Criteria

- AC-001 (REQ-001/REQ-002): `navigate_to_page("agentic research")` returns `{status:"ok", resolvedRoute:"/agentic-research"}`.
- AC-002 (REQ-001/REQ-002): `navigate_to_page("pytorch")` returns `{status:"ok", resolvedRoute:"/ml/pytorch"}` and `navigate_to_page("tensorflow")` returns `{status:"ok", resolvedRoute:"/ml/tensorflow"}`.
- AC-003 (REQ-003): invalid route input returns `{status:"error", code:"INVALID_ROUTE", allowedRoutes:[...]}`.
- AC-004 (REQ-004): valid tabs return `{status:"ok", tab:<canonical>}`; invalid tabs return `{status:"error", code:"INVALID_TAB", allowedTabs:[...]}`.
- AC-005 (REQ-005): Pure logic files can be tested without React runtime.
- AC-006 (REQ-006): `/ag-ui` still mounts both global frontend tools and AG-UI tab switch tool.
- AC-007 (REQ-007): Given existing AG-UI chat messages, when user navigates away and returns to `/ag-ui`, prior conversation remains visible.

## Constitution Compliance

| Article | Status | Notes |
|---|---|---|
| I — Library-First | COMPLIES | Uses existing React/CopilotKit/Next APIs; no duplicate library replacement. |
| II — Test-First | COMPLIES | Spec defines deterministic requirements for TDD coverage prior to behavior changes. |
| III — Simplicity Gate | COMPLIES | Scope constrained to tool routing + structure alignment only. |
| IV — Anti-Abstraction Gate | COMPLIES | No new speculative abstraction requirements. |
| V — Integration-First Testing | COMPLIES | Requires integration coverage for `/ag-ui` tool registration and unit coverage for pure logic. |
| VI — Security-by-Default | N/A | No auth/security surface changed in this scope. |
| VII — Spec Integrity | COMPLIES | Requirements are explicit, deterministic, and traceable to files/tests. |
| VIII — Observability | COMPLIES | Existing tool-call structured response contracts are preserved and expanded in tests. |

## Open Clarifications

- None.
