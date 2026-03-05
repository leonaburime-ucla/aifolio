# AI Chat Error Spec

Version: `1.8.0`
Last updated: `2026-03-02`

## Error Handling Contracts

- ERR-001: API adapter parse failures return `null`, never throw into UI hooks.
- ERR-002: Model fetch failure falls back to deterministic fallback model list.
- ERR-003: Submit pipeline always leaves `isSending=false` even if API throws.
- ERR-004: Clipboard write failure clears copied indicator and does not crash UI.
- ERR-005: Endpoint timeout returns a deterministic timeout result shape (`retryable=true`) and must not crash UI.
- ERR-006: Aborted request returns deterministic no-op completion behavior; no stale-state writes are allowed.
- ERR-007: Invalid attachments return deterministic rejection outcomes without breaking submit flow.
