# Extraction Manifest — ai-chat

## Source Information

- Source: `web/nextjs/src/features/ai-chat/`
- Extraction date: 2026-06-09
- Passes completed: 5 (inventory + 4 extraction passes)
- Extraction agent: Claude Opus 4.6
- Pass artifacts produced: `inventory.md`, `artifact-1-core-logic.md`, `artifact-2-data-access.md`, `artifact-3-boundaries.md`, `artifact-4-external.md`
- Branch: main
- Runtime environment inspected: none (static analysis only)
- Test command run: none (tests read for evidence, not executed)

---

## Requirement Totals

- **Total requirements:** 100 (REQ-CHAT-001 through REQ-CHAT-100)

### Confidence Distribution

| Confidence Level | Count | Percentage |
|-----------------|-------|-----------|
| tested | 29 | 29% |
| observed | 71 | 71% |
| inferred | 0 | 0% |
| documented-only | 0 | 0% |

### Confidence by Pass

| Pass | Requirements | Tested | Observed |
|------|-------------|--------|----------|
| Pass 1 (Core Logic) | 30 | 19 | 11 |
| Pass 2 (Data & Access) | 20 | 2 | 18 |
| Pass 3 (Boundaries) | 23 | 6 | 17 |
| Pass 4 (External) | 27 | 2 | 25 |

---

## Coverage Status

- **Entrypoints with at least 1 requirement:** 100%
- All inventory entrypoints (types, hooks, compositions, logic, API endpoints, view components) have been covered by one or more requirements across the four passes.

---

## Human-Attention Markers Summary

### Blocking (must resolve before Software Architect proceeds)

| Marker | REQ ID | Description |
|--------|--------|-------------|
| `[NEEDS CLARIFICATION]` | REQ-CHAT-044 | Stale `isSending` after navigate-away |
| `[NEEDS CLARIFICATION]` | REQ-CHAT-065 | No file size or type validation on attachments |

### Important (should resolve before implementation)

| Marker | REQ ID | Description |
|--------|--------|-------------|
| `[CONCURRENCY CONTRACT]` | REQ-CHAT-038 | No domain-level guard against concurrent submissions |
| `[CONCURRENCY CONTRACT]` | REQ-CHAT-041 | Feature relies on JS single-threaded event loop |
| `[CONCURRENCY CONTRACT]` | REQ-CHAT-045 | Global singleton store shared across pages |

### Advisory (resolve during implementation or defer)

| Marker | REQ ID | Description |
|--------|--------|-------------|
| `[TEMPORAL COUPLING]` | REQ-CHAT-047 | Model selection depends on prior model fetch |
| `[ENVIRONMENTAL CONTRACT]` | REQ-CHAT-061 | Deployment must set AI_API_URL env var |
| `[ENVIRONMENTAL CONTRACT]` | REQ-CHAT-063 | Feature requires standard browser Web APIs |
| `[ENVIRONMENTAL CONTRACT]` | REQ-CHAT-079 | Chat sidebar must not be server-side rendered |
