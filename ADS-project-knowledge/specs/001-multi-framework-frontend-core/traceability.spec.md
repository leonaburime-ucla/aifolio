# Traceability Matrix: Multi-Framework Frontend Core Extraction

---

## Header Metadata

| Field | Value |
|-------|-------|
| spec_id | SPEC-001 |
| feature_name | FEAT-001-multi-framework-frontend-core |
| version | 1.0.0 |
| content_hash | sha256:placeholder |
| last_edited | 2026-06-09T17:00:00Z |
| traceability_status | PENDING IMPLEMENTATION |

---

## 1. Requirement-to-Implementation-to-Test Matrix

| REQ/AC ID | Description | Priority | Impl File | Impl Function | Test File | Test ID | Status |
|-----------|-------------|----------|-----------|---------------|-----------|---------|--------|
| REQ-01 | Shared domain contracts defined in contracts package with zero framework imports | -- | pending | pending | pending | pending | PENDING |
| AC-01 (REQ-01) | ChartSpec type matches existing shape with all fields | P1 | pending | pending | pending | pending | PENDING |
| AC-02 (REQ-01) | ChatMessage type has correct fields | P1 | pending | pending | pending | pending | PENDING |
| AC-03 (REQ-01) | MlDatasetOption type has correct fields | P1 | pending | pending | pending | pending | PENDING |
| REQ-02 | Pure logic functions extractable with zero framework imports | -- | pending | pending | pending | pending | PENDING |
| AC-04 (REQ-02) | resolveFallbackModelSelection returns correct fallback | P1 | pending | pending | pending | pending | PENDING |
| AC-05 (REQ-02) | resolveFetchedModelSelection returns correct fetched result | P1 | pending | pending | pending | pending | PENDING |
| REQ-03 | Contracts package exports Zod schemas for every domain entity | -- | pending | pending | pending | pending | PENDING |
| AC-06 (REQ-03) | ChartSpec Zod schema rejects missing required field | P1 | pending | pending | pending | pending | PENDING |
| REQ-04 | Logic package exports pure functions with no framework deps | -- | pending | pending | pending | pending | PENDING |
| AC-07 (REQ-04) | All functions execute in Node.js without framework packages | P1 | pending | pending | pending | pending | PENDING |
| REQ-05 | Next.js app functions without regression after extraction | -- | pending | pending | pending | pending | PENDING |
| AC-08 (REQ-05) | Existing frontend test suite passes after import updates | P1 | pending | pending | pending | pending | PENDING |
| REQ-06 | Framework apps satisfy identical behavioral contracts | -- | pending | pending | pending | pending | PENDING |
| AC-09 (REQ-06) | Shared contract tests pass for both Next.js and second framework | P1 | pending | pending | pending | pending | PENDING |
| AC-14 (REQ-06) | validateTrainingInput rejects empty dataset ID | P2 | pending | pending | pending | pending | PENDING |
| REQ-07 | State portability contracts define shape and transitions | -- | pending | pending | pending | pending | PENDING |
| AC-10 (REQ-07) | ChatState initial values match contract | P2 | pending | pending | pending | pending | PENDING |
| AC-15 (REQ-07) | MlDatasetState loading transition produces cache or error | P2 | pending | pending | pending | pending | PENDING |
| REQ-08 | Non-breaking extraction via temporary re-exports | -- | pending | pending | pending | pending | PENDING |
| AC-11 (REQ-08) | Re-export at original path forwards to shared package | P1 | pending | pending | pending | pending | PENDING |
| REQ-09 | Cross-feature coupling resolved through shared contracts | -- | pending | pending | pending | pending | PENDING |
| AC-12 (REQ-09) | Zero direct cross-feature imports for extracted types after boundary rules | P2 | pending | pending | pending | pending | PENDING |
| REQ-10 | Every extracted module has framework-agnostic tests | -- | pending | pending | pending | pending | PENDING |
| AC-13 (REQ-10) | Package tests pass in isolation without framework deps | P1 | pending | pending | pending | pending | PENDING |

---

## 2. Invariant Traceability

| INV ID | Invariant (copied from feature.spec.md) | Test File | Test ID | Status |
|--------|-----------------------------------------|-----------|---------|--------|
| INV-01 | Shared contract packages must never import from framework-specific packages | pending | pending | PENDING |
| INV-02 | Shared logic packages must never import from framework-specific or state management packages | pending | pending | PENDING |
| INV-03 | Every exported type must have a corresponding runtime validation schema | pending | pending | PENDING |
| INV-04 | Shared contract tests executable without installing frontend frameworks | pending | pending | PENDING |
| INV-05 | Re-export modules must re-export exact same public API surface | pending | pending | PENDING |
| INV-06 | State portability contracts must define initial values for every field | pending | pending | PENDING |
| INV-07 | Pure logic functions must be deterministic | pending | pending | PENDING |

---

## 3. Edge Case Traceability

| EC ID | Edge Case (copied from feature.spec.md) | Test File | Test ID | Status |
|-------|-----------------------------------------|-----------|---------|--------|
| EC-01 | Shared contracts imported by consumer with older TypeScript | pending | pending | PENDING |
| EC-02 | Re-export imported but shared package not built yet | pending | pending | PENDING |
| EC-03 | Framework app omits optional field from state shape | pending | pending | PENDING |
| EC-04 | Framework apps produce different intermediate chart representations | pending | pending | PENDING |
| EC-05 | Cross-feature import not resolvable through shared contracts (React hook dep) | pending | pending | PENDING |
| EC-06 | New chart type added after extraction | pending | pending | PENDING |

---

## 4. Error Code Traceability

This feature does not define new error codes. Existing error envelope format from the backend is preserved unchanged. See `errors.spec.md` OMITTED justification in `spec-manifest.md`.

| Error Code | Produced By (file/function) | Test File | Test ID | Status |
|------------|-----------------------------|-----------|---------|--------|
| -- | -- | -- | -- | -- |

---

## 5. Behavior Rule Traceability

| Rule | Section in behavior.spec.md | Test File | Test ID | Status |
|------|-----------------------------|-----------|---------|--------|
| Contract source precedence: shared package wins | 1.1 | pending | pending | PENDING |
| Import path precedence: direct package import preferred | 1.2 | pending | pending | PENDING |
| Extraction ordering: dependency order, alphabetical within tier | 2.1 | pending | pending | PENDING |
| Migration phase ordering: strict sequence 1-5 | 2.2 | pending | pending | PENDING |
| Cross-feature decoupling order: by coupling severity | 2.3 | pending | pending | PENDING |
| Verification gate G-01: contracts package builds and validates | 7 | pending | pending | PENDING |
| Verification gate G-02: logic package builds and tests pass | 7 | pending | pending | PENDING |
| Verification gate G-03: Next.js builds and tests pass after migration | 7 | pending | pending | PENDING |
| Verification gate G-04: app builds after re-export removal | 7 | pending | pending | PENDING |
| Verification gate G-05: boundary rules pass with zero violations | 7 | pending | pending | PENDING |
| Type deduplication: replace with re-export of canonical | 5.1 | pending | pending | PENDING |
| Circular dependency extraction: introduce interface at boundary | 8 (edge case) | pending | pending | PENDING |

---

## 6. Coverage Gaps

### 6.1 Unimplemented Requirements

| REQ/AC ID | Reason Unimplemented | Target Completion | Owner |
|-----------|---------------------|-------------------|-------|
| -- | -- | -- | -- |

### 6.2 Untested Requirements

| REQ/AC ID | Reason Untested | Target Completion | Owner |
|-----------|----------------|-------------------|-------|
| -- | -- | -- | -- |

### 6.3 Untested Error Codes

| Error Code | Reason Untested | Target Completion | Owner |
|------------|----------------|-------------------|-------|
| -- | -- | -- | -- |

### 6.4 Deferred Items

| REQ/AC ID | Deferred To | Reason | Approved By |
|-----------|-------------|--------|-------------|
| -- | -- | -- | -- |

---

## 7. Untraced Requirements

| REQ/AC ID | Reason Not In Matrix |
|-----------|---------------------|
| -- | -- |

---

## 8. Traceability Completeness Checklist

- [x] All REQ-* from feature.spec.md appear in the Section 1 matrix
- [x] All AC-* from feature.spec.md appear in the Section 1 matrix
- [x] All INV-* from feature.spec.md appear in the Section 2 matrix
- [x] All EC-* from feature.spec.md appear in the Section 3 matrix
- [ ] All error codes from errors.spec.md appear in the Section 4 matrix
- [x] All behavior rules from behavior.spec.md appear in the Section 5 matrix
- [x] Section 6.1 (unimplemented) is empty or all entries are DEFERRED with approval
- [x] Section 6.2 (untested) is empty or all entries are DEFERRED with approval
- [x] Section 6.3 (untested error codes) is empty or all entries are DEFERRED with approval
- [x] Section 7 (untraced) is empty
- [ ] All VERIFIED rows have been reviewed and signed off by the Code Review Agent

**[ ] TRACEABILITY COMPLETE** -- all requirements are implemented and tested. Feature is ready to ship.

---

## Sign-Off

| Role | Name / Agent | Date (ISO-8601) | Notes |
|------|--------------|-----------------|-------|
| Spec Agent | Spec Agent | 2026-06-09T17:00:00Z | Initial seeding -- all rows PENDING IMPLEMENTATION |
| TDD Agent | | | |
| Programmer Agent | | | |
| Code Review Agent | | | |
| Coordinator | | | |
