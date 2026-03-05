# AI Chat Behavior Spec

Version: `1.8.0`
Last updated: `2026-03-02`

This file defines behavior flow semantics only. Requirement ownership remains in `ai-chat.spec.md`.

## Behavior Flows

- BH-001 (REQ-001, REQ-002): Submit flow ordering is deterministic:
  `normalize -> append input history -> append user message -> set sending true -> send API -> append assistant message (if any) -> onMessageReceived -> set sending false -> clear attachments`.
- BH-002 (REQ-003): Chart fan-out preserves assistant payload order and cardinality.
- BH-003 (REQ-007): Abort flow is terminal for state mutation on the aborted request path.
- BH-004 (REQ-008): Attachment acceptance/rejection is deterministic for the same file set.

## Behavioral Invariants

- INV-001: No behavior path may leave `isSending=true` after completion.
- INV-002: Empty normalized submission has zero write side-effects.
- INV-003: History cursor operations are total (no throw) for all cursor/index states.
