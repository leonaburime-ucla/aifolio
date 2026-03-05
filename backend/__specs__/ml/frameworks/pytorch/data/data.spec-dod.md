# Spec Definition of Done (DoD) Checklist: pytorch-data

| Field | Value |
|-------|-------|
| spec_id | PT-DATA-001 |
| feature_name | pytorch-data |
| version | 0.5 |
| filled_by | Spec Agent (Direct) |
| filled_date | 2026-02-25T00:00:00Z |
| reviewed_by | — |
| reviewed_date | — |

---

## How to Use This Checklist

- `PASS` — fully satisfied.
- `FAIL` — not satisfied; spec must be updated before handoff. See Blocking Issues section.
- `NA` — genuinely does not apply; justification required.
- All items must have a status. Blank = FAIL.

---

## Section A: Spec Package Completeness

| # | Item | Status | Notes |
|---|------|--------|-------|
| A-01 | `feature.spec.md` is present in the feature folder | PASS | `data.feature.spec.md` present |
| A-02 | `feature.spec.md` is non-empty — all placeholder values replaced with real content | PASS | 9 REQs, 9 ACs, invariants, edge cases, dependencies all populated |
| A-03 | `api.contract.spec` present (or NA with justification) | NA | Python ML library; no language-specific API layer |
| A-04 | `state.contract.spec` present (or NA with justification) | NA | Python library; no state management layer |
| A-05 | `orchestrator.contract.spec` present (or NA with justification) | NA | Python library; no orchestrator layer |
| A-06 | `ui.contract.spec` present (or NA with justification) | NA | Python library; no UI |
| A-07 | `errors.contract.spec` present (or NA with justification) | NA | `prepare_tensors` has no custom error codes; all errors propagate from dependencies (documented in spec-manifest.md) |
| A-08 | `behavior.spec.md` present (or NA with justification) | NA | Linear pipeline with no branching; omission justified in spec-manifest.md |
| A-09 | `traceability.spec.md` present and all REQ/AC rows populated | PASS | Covered by shared `ai/__specs__/traceability/pytorch-backend-traceability.spec.md` (PYTORCH-TRACE-001 v1.1); PT-DATA-001 rows present with test file mappings |
| A-10 | `requirements.md` (legacy checklist) present and filled | NA | Legacy format not used; spec-manifest.md serves this role |

---

## Section B: feature.spec.md Quality

| # | Item | Status | Notes |
|---|------|--------|-------|
| B-01 | `spec_id` assigned and unique | PASS | PT-DATA-001; no duplicate found under `__specs__/ml/` |
| B-02 | `version` is valid semver | FAIL | Version is `0.5`; must be `0.5.0` per semver format |
| B-03 | `status` is APPROVED | FAIL | No `status` field in spec metadata; field must be added and set to APPROVED after human review |
| B-04 | `content_hash` computed and recorded | PASS | `sha256:91809e01e92adfc329ad4b757ff2e7f197d9f4f24e8187881cceea5a09511309` present |
| B-05 | `feature_name` matches FEAT folder name exactly | FAIL | FEAT number not yet registered in `AI-Dev-Shop-speckit/reports/pipeline/`; brownfield spec predates pipeline registration |
| B-06 | `last_edited` is valid ISO-8601 UTC | PASS | `2026-02-25T00:00:00Z` |
| B-07 | `owner` set to named human or team | FAIL | No `owner` field in spec metadata |
| B-08 | Overview section present (1–3 sentences) | PASS | Goals section describes `prepare_tensors` purpose clearly |
| B-09 | Problem Statement present (Current / Desired / Success) | FAIL | Goals section present but missing formal "Current state / Desired state / Success signal" structure |
| B-10 | Scope: In-scope list present and non-empty | PASS | `prepare_tensors` function, return tuple, preprocessing pipeline |
| B-11 | Scope: Out-of-scope list present and non-empty | PASS | File-loading logic, model construction explicitly excluded |
| B-12 | Zero `[NEEDS CLARIFICATION]` markers | PASS | None found |
| B-13 | All Open Questions have owner and resolution target date | NA | No open questions |
| B-14 | Requirements section has at least one REQ-* item | PASS | 9 REQ-D* items (REQ-D01 through REQ-D09) |
| B-15 | All REQ-* observable and testable — no vague qualifiers | PASS | No banned language found; all requirements specify exact behavior |
| B-16 | All REQ-* independently verifiable | PASS | Each requirement can be tested in isolation |
| B-17 | Acceptance Criteria section has at least one AC-* | PASS | 9 AC-D* items |
| B-18 | Every REQ-* has at least one AC-* | PASS | 1-to-1 mapping confirmed |
| B-19 | All AC-* follow Given/When/Then format | PASS | All ACs use Given/When/Then |
| B-20 | All AC-* have [P1], [P2], or [P3] priority tag | PASS | Priority tags confirmed on all ACs |
| B-21 | All P1 AC items independently testable | PASS | P1 ACs require only input data and `prepare_tensors`; no upstream dependency |
| B-22 | No AC requires knowledge of implementation to evaluate | PASS | All ACs specify observable outputs (tensor shapes, dtypes, artifact properties) |
| B-23 | Invariants section has at least one INV-* | PASS | Invariants present (preprocessing fit on train only, etc.) |
| B-24 | All INV-* written as absolute statements | PASS | "must always" / "must never" language used |
| B-25 | Edge Cases section has at least one EC-* | PASS | EC items present (single unique label, all-NaN columns, empty dataset) |
| B-26 | All EC-* are concrete scenarios | PASS | Each EC specifies exact input condition |
| B-27 | All EC-* have explicit Expected Behavior | PASS | Each EC documents the expected output or error |
| B-28 | Dependencies table complete — no blank cells | PASS | Dependencies listed with failure modes and fallback behavior |
| B-29 | Constitution Compliance table complete | FAIL | No Constitution Compliance table; spec predates this requirement |
| B-30 | EXCEPTION rows have notes | NA | B-29 fails; no EXCEPTION entries to evaluate |
| B-31 | Implementation Readiness Gate in feature.spec.md shows PASS | FAIL | No Implementation Readiness Gate section in spec |

---

## Section C: Contract Quality

| # | Item | Status | Notes |
|---|------|--------|-------|
| C-01 | All language-specific contract files use interfaces/types | NA | Python library; no language-specific contract files required |
| C-02 | All public interfaces have JSDoc comments | NA | Python library |
| C-03 | Optional fields explicitly marked with `?` | NA | Python library |
| C-04 | Nullable fields typed as `T \| null` | NA | Python library |
| C-05 | No `any` types | NA | Python library |
| C-06 | Const objects use `as const` | NA | Python library |
| C-07 | api.contract.spec: All endpoints in API_ENDPOINTS registry | NA | Python library |
| C-08 | api.contract.spec: All error codes in API_ERROR_HTTP_STATUS | NA | Python library |
| C-09 | api.contract.spec: Auth requirements present | NA | Python library |
| C-10 | state.contract.spec: INITIAL_FEATURE_STATE covers all fields | NA | Python library |
| C-11 | state.contract.spec: STATE_TRANSITIONS covers all action types | NA | Python library |
| C-12 | state.contract.spec: STATE_INVARIANTS falsifiable | NA | Python library |
| C-13 | orchestrator.contract.spec: Async outputs return OrchestratorResult<T> | NA | Python library |
| C-14 | orchestrator.contract.spec: ORCHESTRATOR_INVARIANTS falsifiable | NA | Python library |
| C-15 | ui.contract.spec: All components have Props interface | NA | Python library |
| C-16 | ui.contract.spec: DISPLAY_CONDITIONS covers all interactive components | NA | Python library |
| C-17 | ui.contract.spec: ACCESSIBILITY_REQUIREMENTS covers all components | NA | Python library |
| C-18 | errors.contract.spec: All error codes have full entries | NA | Python library |
| C-19 | errors.contract.spec: No error code missing from coverage requirements | NA | Python library |

---

## Section D: Behavior Rules Quality

| # | Item | Status | Notes |
|---|------|--------|-------|
| D-01 | Precedence rules cover every multi-source field | NA | No behavior.spec.md; linear pipeline has no precedence conflicts |
| D-02 | Precedence rules are ordered | NA | See D-01 |
| D-03 | Default Values table covers all non-obvious defaults | NA | No behavior.spec.md; no configurable defaults in `prepare_tensors` |
| D-04 | "Why" column in Default Values has rationale | NA | See D-03 |
| D-05 | Limits and Bounds table covers all numeric constraints | NA | No behavior.spec.md; constraints documented in feature spec |
| D-06 | Enforcement column specifies where constraints are checked | NA | See D-05 |
| D-07 | Deduplication rules define "duplicate" precisely | NA | No deduplication logic in this module |
| D-08 | Tie-break logic is deterministic | NA | No tie-break logic in this module |
| D-09 | Edge Case Handling covers all boundary values | NA | No behavior.spec.md; edge cases in feature spec |
| D-10 | Every behavior rule has a traceability row | NA | No behavior.spec.md |

---

## Section E: Traceability Quality

| # | Item | Status | Notes |
|---|------|--------|-------|
| E-01 | traceability.spec.md is present | PASS | Shared `pytorch-backend-traceability.spec.md` covers PT-DATA-001 |
| E-02 | Every REQ-* appears in traceability Section 1 | PASS | All REQ-D* mapped in shared traceability file |
| E-03 | Every AC-* appears in traceability Section 1 | PASS | All AC-D* mapped |
| E-04 | Every INV-* appears in traceability Section 2 | PASS | INV items mapped |
| E-05 | Every EC-* appears in traceability Section 3 | PASS | EC items mapped |
| E-06 | Every error code in errors.contract.spec in traceability Section 4 | NA | No errors.contract.spec for this module |
| E-07 | "Pending" rows acceptable at spec stage | PASS | Test IDs marked "planned" are acceptable before TDD |
| E-08 | Section 7 (Untraced Requirements) is empty | PASS | No untraced requirements identified |

---

## Section F: Internal Consistency

| # | Item | Status | Notes |
|---|------|--------|-------|
| F-01 | Error codes in api.contract.spec match errors.contract.spec | NA | No language-specific contract files |
| F-02 | Status types consistent across spec files | NA | No language-specific contract files |
| F-03 | OrchestratorItem fields consistent with FeatureItem | NA | No language-specific contract files |
| F-04 | ItemSummary fields consistent with OrchestratorItem | NA | No language-specific contract files |
| F-05 | Default values in orchestrator.contract.spec match behavior.spec.md | NA | No language-specific contract files |
| F-06 | Rate limit values consistent across spec files | NA | No language-specific contract files |
| F-07 | All spec files reference same spec_id and feature_name | PASS | Single spec file for this module; consistent |
| F-08 | All spec files have consistent version numbers | PASS | Single spec file; no cross-file inconsistency |

---

## Section G: Constitution Compliance Verification

| # | Item | Status | Notes |
|---|------|--------|-------|
| G-01 | Article I (Library-First): no custom implementations where libraries exist | PASS | Spec references sklearn, torch, numpy — existing libraries throughout |
| G-02 | Article II (Test-First): spec makes no assumptions about implementation order | PASS | Spec defines observable contracts only; no implementation ordering assumed |
| G-03 | Article III (Simplicity Gate): all modules trace to a requirement | NA | No language-specific modules in this spec |
| G-04 | Article IV (Anti-Abstraction Gate): no speculative abstractions | NA | No language-specific contract files |
| G-05 | Article V (Integration-First Testing): all P1 ACs have traceability row | PASS | All P1 ACs mapped in shared traceability (pending is acceptable at spec stage) |
| G-06 | Article VI (Security-by-Default): auth requirements for all endpoints | NA | Python ML library; no HTTP endpoints in this module |
| G-07 | Article VII (Spec Integrity): spec_id and content_hash present and correct | PASS | Both present in metadata header |
| G-08 | Article VIII (Observability): structured error payloads with correlationId | NA | Python library; no errors.contract.spec required |

---

## Section H: Final Gate

| # | Item | Status | Notes |
|---|------|--------|-------|
| H-01 | **Implementation Readiness Gate:** A new developer can implement from spec alone without clarifying questions | PASS | Function signature, return tuple (12 fields), preprocessing pipeline steps, split ratios, stratification logic, dtype requirements, and all edge cases are fully specified. A developer can implement `prepare_tensors` from this spec without any additional context. |

---

## Summary

| Section | Items | Passing | Failing | NA |
|---------|-------|---------|---------|-----|
| A: Package Completeness | 10 | 2 | 0 | 8 |
| B: feature.spec.md Quality | 31 | 21 | 7 | 3 |
| C: Contract Quality | 19 | 0 | 0 | 19 |
| D: Behavior Rules Quality | 10 | 0 | 0 | 10 |
| E: Traceability Quality | 8 | 7 | 0 | 1 |
| F: Internal Consistency | 8 | 2 | 0 | 6 |
| G: Constitution Compliance | 8 | 4 | 0 | 4 |
| H: Final Gate | 1 | 1 | 0 | 0 |
| **TOTAL** | **95** | **37** | **7** | **51** |

**Overall DoD Result:** FAIL — 7 items require resolution before Architect dispatch.

> H-01 PASSES: spec content is implementation-ready. All FAIL items are metadata/structural gaps that do not affect developer ability to implement from the spec.

---

## Blocking Issues

| Item ID | Issue | Required Change | Owner | Target Date |
|---------|-------|----------------|-------|-------------|
| B-02 | Version `0.5` is not valid semver | Change to `0.5.0` in metadata header; recompute hash | — | — |
| B-03 | No `status` field in metadata | Add `status: APPROVED` after human review; recompute hash | — | — |
| B-05 | FEAT number not registered in pipeline | Create `AI-Dev-Shop-speckit/reports/pipeline/` entry for this spec | — | — |
| B-07 | No `owner` field | Add `owner: <name>` to metadata header; recompute hash | — | — |
| B-09 | No formal Problem Statement | Add "Current state / Desired state / Success signal" section | — | — |
| B-29 | No Constitution Compliance table | Add constitution compliance table covering all 8 articles | — | — |
| B-31 | No Implementation Readiness Gate section | Add self-check section to feature.spec.md | — | — |

---

## Sign-Off Block

| Role | Name / Agent ID | Date (ISO-8601 UTC) | Signature |
|------|-----------------|---------------------|-----------|
| Spec Agent | Spec Agent (Direct) / 2026-02-25 | 2026-02-25T00:00:00Z | — |
| Coordinator | — | — | — |
