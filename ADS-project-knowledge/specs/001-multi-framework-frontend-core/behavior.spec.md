# Behavior Rules Spec: Multi-Framework Frontend Core Extraction

---

## Header Metadata

| Field | Value |
|-------|-------|
| spec_id | SPEC-001 |
| feature_name | FEAT-001-multi-framework-frontend-core |
| version | 1.0.0 |
| content_hash | sha256:placeholder |
| last_edited | 2026-06-09T17:00:00Z |

**Purpose:** This file captures the deterministic extraction sequencing rules, migration ordering, rollback gates, and verification precedence that govern how the multi-framework extraction proceeds. These are behavioral rules that affect correctness of the migration -- not general feature behavior.

---

## 1. Precedence Rules

### 1.1 Contract Source Precedence

**Situation:** When the same type definition exists in both the shared contracts package and in `web/nextjs/src/features/*/`, which is authoritative?

**Sources in precedence order (highest to lowest):**
1. Shared contracts package (`contracts/entities/*`) -- canonical source of truth after extraction
2. Re-export in `web/nextjs` -- compatibility shim, must forward to #1 exactly
3. Inline type definition in a feature slice -- stale duplicate, must be removed

**Example:**
- Scenario: `ChartSpec` is defined in shared contracts and also still imported from `features/charts/contracts/chart.types.ts` in some files.
- Input: Both locations exist. Shared package has the type. Feature slice re-exports it.
- Result: The shared package definition is authoritative. The feature slice file is a re-export.

**Test requirement:** A build-time lint rule must flag any type definition in a feature slice that duplicates a type exported from the shared contracts package.

### 1.2 Import Path Precedence

**Situation:** When importing a shared contract, which import path takes precedence?

**Sources in precedence order (highest to lowest):**
1. Direct package import (`@aifolio/contracts/entities/chart` or equivalent package path) -- preferred final state
2. Re-export path (`@/features/charts/contracts/chart.types`) -- acceptable during migration
3. Relative path to shared package source files -- forbidden in consuming apps

**Test requirement:** After migration completes, zero re-export path imports must remain. During migration, both #1 and #2 are acceptable.

---

## 2. Ordering Rules

### 2.1 Extraction Ordering

**Rule:** Extraction must proceed in dependency order. A module that is imported by other extraction candidates must be extracted first.

**Order:**
1. Contracts/types with zero cross-package dependencies (ChartSpec, ChatMessage, MlDatasetOption)
2. Contracts/types that depend on previously extracted types (ChatAssistantPayload depends on ChartSpec)
3. Pure logic functions that depend on extracted contracts (resolveFallbackModelSelection depends on ChatModelOption)
4. Pure logic functions that depend on other extracted logic (buildChatHistoryWindow depends on ChatHistoryMessage)

**Stability:** Within the same dependency tier, extraction order is alphabetical by module name for determinism.

**Invariant:** WHEN a module is extracted, THEN all types it imports must already exist in the shared contracts package or be extracted in the same batch.

### 2.2 Migration Phase Ordering

**Phases must execute in strict sequence:**
1. Phase 1: Extract contracts package (types + schemas only)
2. Phase 2: Extract frontend-core package (pure logic)
3. Phase 3: Update Next.js imports to use shared packages
4. Phase 4: Remove re-export shims (only after all consumers migrated)
5. Phase 5: Add boundary enforcement rules

**Invariant:** No phase N+1 work may begin until phase N verification gate passes (see Section 7).

### 2.3 Cross-Feature Decoupling Order

**Rule:** Cross-feature imports must be resolved in order of coupling severity (most imports first).

**Current cross-feature coupling by import count:**
1. `ag-ui-chat -> ml`: 14 imports -- resolve first
2. `ag-ui-chat -> agentic-research`: 6 imports -- resolve second
3. `ag-ui-chat -> recharts`: 4 imports -- resolve third
4. `agentic-research -> ag-ui-chat`: 1 import -- resolve fourth

**Invariant:** Resolving coupling at position N must not increase coupling at position N+1 or later.

---

## 3. Default Values

| Field | Scope | Default Value | Why |
|-------|-------|---------------|-----|
| Extraction batch size | Migration execution | 1 module per PR | Keeps PRs reviewable and rollback-safe. Batching only when modules are tightly co-dependent. |
| Re-export retention period | Migration timeline | Until all consumers in `web/nextjs` are migrated to direct imports | Prevents breaking changes during incremental migration. |
| Shared test runner | Contract test suite | Vitest | Aligned with existing frontend tooling. Open Question OQ-01 may override. |
| Zod schema generation | Contracts package | Co-located with type in same file | Keeps type and runtime validator in sync. Separating them invites drift. |
| Package compilation target | Shared packages | ESNext with declaration files | Consuming apps handle downleveling via their own build pipeline. |

---

## 4. Limits and Bounds

| Constraint | Value | Enforcement | Notes |
|------------|-------|-------------|-------|
| Maximum cross-feature imports after extraction (contracts resolved) | 0 for extracted types | ESLint no-restricted-imports rule | Only applies to types/contracts that have been extracted. Framework-specific cross-feature imports are separate. |
| Maximum re-export files per feature slice | 1 per extracted module | Code review | Each feature may have at most one re-export barrel per extracted contract area. |
| Maximum extraction batch PR size | 500 lines changed | Code review policy | PRs exceeding this must be split unless the batch is a single tightly-coupled module group. |
| Minimum shared test coverage per extracted module | 1 test per exported symbol | CI gate | Every public export (type schema validation + function) must have at least one test. |
| Maximum time a re-export may live after all consumers migrated | 1 sprint (2 weeks) | Manual tracking | Stale re-exports must be removed within one sprint of consumer migration completion. |

---

## 5. Deduplication Rules

### 5.1 Type Definition Deduplication

A type definition is considered a duplicate if:
1. It defines the same shape (same field names, same field types, same optionality) as an existing export from the shared contracts package.
2. It exists in `web/nextjs/src/features/` (not in the shared package itself).
3. It is not a re-export (i.e., it defines the type locally rather than importing and re-exporting).

**How duplicates are handled:**
- WHEN a duplicate type definition is detected after extraction, THEN the duplicate must be replaced with a re-export of the canonical shared package type within the same PR that extracts the canonical version.
- IF the duplicate has minor differences (e.g., an extra optional field), THEN the canonical version must be the superset. The difference must be documented in the extraction PR.

### 5.2 Logic Function Deduplication

A logic function is considered a duplicate if:
1. It performs the same computation (same inputs produce same outputs) as an exported function from the shared logic package.
2. It exists in `web/nextjs/src/features/*/logic/`.
3. It is not a thin wrapper adding framework-specific behavior.

**How duplicates are handled:**
- Replace with import from shared logic package.
- If the duplicate adds framework-specific behavior (e.g., calls a React hook), keep the framework-specific wrapper but extract the pure computation.

---

## 6. Tie-Break Logic

### 6.1 Conflicting Type Definitions Across Feature Slices

**When does this apply:** Two feature slices define types with the same semantic meaning but slightly different shapes (e.g., both define a "model option" type with different optional fields).

**Tie-break rule:** The type with more consumers (more import sites) is the canonical version. If consumer counts are equal, the type in the feature slice that was created first (earlier git history) wins.

**Rationale:** Canonical version should be the one requiring fewer consumer changes during migration.

**Invariant:** The tie-break produces a single canonical type. The non-canonical version must be migrated to use the canonical one.

---

## 7. Verification Gates (Rollback Points)

Each migration phase must pass its verification gate before the next phase begins. If a gate fails, the phase must be rolled back.

| Gate | Phase | Verification Criteria | Rollback Action |
|------|-------|----------------------|-----------------|
| G-01 | After contracts extraction | Shared contracts package builds. All Zod schemas validate sample data. No type errors in shared package. | Revert shared package. Restore original type locations. |
| G-02 | After logic extraction | Shared logic package builds. All framework-agnostic tests pass. No type errors. | Revert shared logic package. Restore original logic locations. |
| G-03 | After Next.js import migration | Next.js app builds without errors. All existing Next.js tests pass. No runtime behavior change. | Revert import changes. Re-exports remain in place. |
| G-04 | After re-export removal | Next.js app builds. No remaining references to removed re-exports. | Restore re-exports. |
| G-05 | After boundary rule enforcement | ESLint/Steiger passes with zero violations. No new cross-feature type imports. | Disable rules temporarily and fix violations before re-enabling. |

**EARS rule:** IF any verification gate returns FAIL, THEN the system shall not proceed to the next phase and the failing phase must be corrected or rolled back.

---

## 8. Edge Case Handling

| Edge Case | Expected Behavior | Test Required? |
|-----------|-------------------|----------------|
| Extracted type has circular dependency with a non-extracted type | WHEN a circular dependency is detected during extraction, THEN the extraction must split the cycle by introducing an interface/port at the boundary. Both sides import the interface from contracts. | Yes |
| Re-export path is imported by a test file but not production code | WHEN re-export consumers are counted for removal timing, THEN test files count as consumers. Re-export is not removed until tests are also migrated. | Yes |
| A pure logic function imports a framework-specific utility (e.g., for ID generation using `crypto.randomUUID`) | IF a logic function depends on a platform API (not a framework API), THEN it is still extractable. Platform APIs (crypto, Date, Math) are acceptable in shared logic. Framework APIs (useEffect, ref, inject) are not. | Yes |
| Two framework apps need the same contract but use different validation libraries | WHEN the contracts package exports Zod schemas, THEN consumers that do not use Zod may use the TypeScript type definitions only and implement their own validation. The Zod schema is canonical but not mandatory for consumption. | No (informational) |
| Extraction changes the runtime behavior of a function due to module-level side effects | IF a module to be extracted has module-level side effects (e.g., global registration), THEN extraction is blocked for that module until side effects are removed or moved to the consuming app's initialization. | Yes |
