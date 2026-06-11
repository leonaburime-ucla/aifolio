# Intentional Changes — ai-chat

Extraction date: 2026-06-09
Source: `web/nextjs/src/features/ai-chat/`

---

## Status

**No intentional-change decisions required for this module.**

---

## Justification

Across all 100 extracted requirements (REQ-CHAT-001 through REQ-CHAT-100):

- **Zero requirements classified as `known_bug`** — no behaviors were identified where the current implementation produces incorrect results that should be fixed in a rewrite.

- **Zero requirements classified as `accidental`** — no behaviors were found that exist due to implementation accident rather than design intent. All observed behaviors either have explicit test evidence confirming intentionality or follow consistent architectural patterns that indicate deliberate design.

- **Zero requirements classified as `deprecated`** — no behaviors are marked for removal. The CopilotKit integration (REQ-CHAT-062) exists as an alternative path but is not deprecated; it is actively used.

- **All 100 requirements have `preservation_decision: preserve`** (98 with `preserve_actual`, 2 with `human_decision_required`).

---

## Items Requiring Human Decision (not changes, but clarification)

Two requirements have `preservation_decision: human_decision_required` rather than an intentional change classification:

| REQ ID | Current Behavior | Question | Priority |
|--------|-----------------|----------|----------|
| REQ-CHAT-044 | Stale `isSending` after navigate-away from global store page | Is this accepted behavior or a gap requiring a cleanup mechanism? | Important |
| REQ-CHAT-065 | No client-side file size/type validation on attachments | Is validation intentionally server-side only, or is frontend validation needed? | Important |

These are flagged as `[NEEDS CLARIFICATION]` in the review digest. They are NOT intentional changes — they are ambiguous behaviors where the rewrite decision depends on human input about intent. Until clarified, the default is `preserve_actual`.
