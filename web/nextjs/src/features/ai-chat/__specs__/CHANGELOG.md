# AI Chat Specs Changelog

## 1.8.0 - 2026-03-02

- De-duplicated behavior vs feature specs; behavior file now focuses on flow semantics/invariants.
- Converted `*.spec.ts` files into machine-readable requirement-reference indexes (not duplicated prose requirements).
- Strengthened traceability with concrete target test file names.
- Added explicit edge-case requirements for abort/unmount behavior, attachment policy, and timeout handling.
- Expanded out-of-scope to explicitly exclude streaming/edit/delete/regenerate/persistence.

## 1.7.0 - 2026-03-02

- Normalized ai-chat specs to a single version baseline (`1.7.0`) across core spec files.
- Added architecture boundaries for screen composition vs feature ownership.
- Added explicit behavior and error contract specs for test-first implementation.
- Updated traceability to include boundary/wiring-focused test requirements.

## 1.6.0 - 2026-03-02

- Moved Agentic Research page-specific chat orchestrator out of `features/ai-chat` into `screens/AgenticResearchPage`.
- Kept `features/ai-chat` focused on reusable chat primitives and generic orchestration.

## 1.5.0 - 2026-03-02

- Moved landing-page-specific chat wiring out of `features/ai-chat` into `screens/LandingPage`.
- Updated ai-chat spec scope to keep only reusable chat feature contracts.

## 1.4.0 - 2026-03-02

- Reorganized implementation layout for portability:
  - `typescript/` for framework-agnostic TypeScript.
  - `typescript/react/{orchestrators,hooks,state,views}` for React-bound code.
  - `types/` mirrors the same shape for type-contract organization.
  - `__tests__/` mirrors the same shape for test placement.

## 1.3.0 - 2026-03-02

- Added full ai-chat spec pack:
  - `ai-chat.spec.md`
  - `ai-chat.api.spec.ts`
  - `ai-chat.state.spec.ts`
  - `ai-chat.orchestrator.spec.ts`
  - `ai-chat.ui.spec.ts`
  - `ai-chat.traceability.spec.md`
  - `spec-manifest.md`
- Normalized ai-chat type contracts into `features/ai-chat/__types__/typescript/`.

## 1.2.0 - 2026-02-23

- Added feature-local spec pack under `features/ai/__specs__`.
- Added adapter-boundary requirements for orchestrators.
- Added explicit state and chart action port contracts.

## 1.1.0 - 2026-01-20

- Initial sidebar chat requirements documented in root-level spec.
