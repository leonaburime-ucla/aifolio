# Review Digest: ai-chat Reverse-Spec

## BLOCKING (must resolve before proceeding)

### 1. [NEEDS CLARIFICATION] -- Stale isSending after navigate-away
- REQ: REQ-CHAT-044
- Issue: When user navigates away mid-request, isMountedRef guard prevents state mutations but does NOT reset isSending=true in the global store. User returns to find send button permanently disabled.
- Decision needed from: human (feature owner)
- Options: A) Accept as known edge case (low frequency) B) Add cleanup logic in unmount effect to reset isSending C) Add isSending staleness check on re-mount

### 2. [NEEDS CLARIFICATION] -- No file attachment validation
- REQ: REQ-CHAT-065
- Issue: No client-side limits on file size, type, or count. Users can attach arbitrarily large files (100MB+) encoded as base64, producing ~133MB JSON payloads.
- Decision needed from: human (product/backend owner)
- Options: A) Backend validates and rejects (document the contract) B) Add frontend guards (max size, type allowlist, count cap) C) Both

## ADVISORY (should address, non-blocking)

### Concurrency Contracts
- **REQ-CHAT-038** [CONCURRENCY CONTRACT]: No domain-level guard against concurrent submissions; relies solely on UI button disable.
- **REQ-CHAT-041** [CONCURRENCY CONTRACT]: All state serialization depends on JS single-threaded event loop; multi-threaded port needs explicit locks.
- **REQ-CHAT-045** [CONCURRENCY CONTRACT]: Global singleton store shared across pages; reimplementation must preserve shared-vs-isolated topology.

### Temporal Coupling
- **REQ-CHAT-037** [TEMPORAL COUPLING]: Submit pipeline steps 1-7 must execute synchronously before async boundary (ordering is test-verified).
- **REQ-CHAT-047** [TEMPORAL COUPLING]: Model fetch must complete before submission includes model ID; feature sends model:null if not ready.

### Environmental Contracts
- **REQ-CHAT-061** [ENVIRONMENTAL CONTRACT]: Deployment must set AI_API_URL for non-local; missing both env vars defaults to 127.0.0.1:8000.
- **REQ-CHAT-063** [ENVIRONMENTAL CONTRACT]: Requires browser APIs (fetch, AbortController, FileReader, clipboard, rAF); all injectable.
- **REQ-CHAT-079** [ENVIRONMENTAL CONTRACT]: Chat sidebar MUST NOT be server-side rendered (hooks require browser APIs).

## INFORMATIONAL

- **Coverage:** 29% tested (29 of 100 requirements have test evidence) -- below 60% threshold; mitigated by comprehensive DI enabling test creation without mocking, and all high-criticality architectural invariants (REQ-026/027) are tested.
- **Total requirements:** 100 across 4 passes (30 core logic, 20 data/access, 23 boundaries, 27 external/consumers).
- **High-criticality count:** 14 requirements marked high (all in fragile_consumer or high_traffic categories).
- **Exit criteria status:** 2 blocking clarifications outstanding; 8 advisory items documented; no contradictions between artifacts detected.
