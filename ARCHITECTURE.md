# Architecture Constitution

> Source of truth for architectural decisions and cross-language parity.
> Version: 1.1.0
> Last updated: 2026-01-20

---

## Table of Contents

1. [Philosophy & Principles](#philosophy--principles)
2. [Core Architecture Pattern](#core-architecture-pattern)
3. [Feature Structure (Bounded Contexts)](#feature-structure-bounded-contexts)
4. [Versioned Specification System](#versioned-specification-system)
5. [Ports & Adapters Explained](#ports--adapters-explained)
6. [Multi-Language Strategy](#multi-language-strategy)
7. [State Management Strategy](#state-management-strategy)
8. [AI Agent Architecture](#ai-agent-architecture)
9. [Database & RPC Specifications](#database--rpc-specifications)
10. [Code Conventions](#code-conventions)
11. [Directory Structure](#directory-structure)
12. [Reading List & References](#reading-list--references)
13. [Hook Integration Pattern](#hook-integration-pattern)
14. [Spec IDs & Versioning](#spec-ids--versioning)

---

## Philosophy & Principles

### Guiding Principles

1. **Feature Isolation (Bounded Contexts)**
   - Each feature is fully self-contained
   - A feature can be removed by deleting one folder
   - Features share only: types, ports, and shared utilities

2. **Spec-Driven Development (SDD)**
   - Specs are written first, code second
   - Each spec has a version number for cross-platform tracking
   - Tests map 1:1 to spec items

3. **Zero Inter-Layer Dependencies (Orc-BASH)**
   - API, Logic, State, Hooks have no dependencies on each other
   - Only the Orchestrator wires them together
   - Dependencies are injected, never imported directly

4. **Ports & Adapters (Hexagonal)**
   - Core business logic has no external dependencies
   - External systems are accessed through ports (interfaces)
   - Adapters implement ports for specific technologies

5. **Multi-Platform from Day One**
   - Business logic is language/platform agnostic
   - Views are platform-specific implementations
   - Specs ensure feature parity across platforms

---

## Core Architecture Pattern

### Orc-BASH + DDD + Hexagonal

```
┌─────────────────────────────────────────────────────────────────┐
│                        FEATURE (Bounded Context)                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    __specs__/                            │   │
│  │    Versioned specifications (source of truth)            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      domain/                             │   │
│  │    Entities, Value Objects, Domain Services, Rules       │   │
│  │    (ZERO external dependencies - pure business logic)    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   ports/    │     │   logic/    │     │   state/    │       │
│  │ Interfaces  │     │  Services   │     │   Stores    │       │
│  │ (contracts) │     │(orchestrate)│     │  (Zustand)  │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   orchestrators/                         │
│  │         Wires everything together per use-case           │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                       views/                             │
│  │    react  |  react-native  |  flutter                    │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     __tests__/                           │
│  │    Mirrors feature structure, maps to spec versions      │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Structure (Bounded Contexts)

### Complete Self-Contained Feature

```
<feature>/
├── __specs__/                            # SPECIFICATIONS (versioned)
│   ├── <feature>.spec.md                 # Human-readable requirements
│   ├── <feature>.domain.spec.ts          # Domain rules (v1.0.0)
│   ├── <feature>.api.spec.ts             # API contracts (v1.0.0)
│   ├── <feature>.state.spec.ts           # State shape (v1.0.0)
│   ├── <feature>.ui.spec.ts              # UI behaviors (v1.0.0)
│   └── CHANGELOG.md                      # Spec version history
│
├── __tests__/                             # TESTS (mirror structure)
│   ├── domain/
│   ├── logic/
│   ├── state/
│   ├── orchestrators/
│   └── integration/
│
├── domain/                                # DOMAIN (pure, no deps)
│   ├── entities/
│   ├── value-objects/
│   ├── events/
│   └── rules/
│
├── ports/                                 # PORTS (interfaces/contracts)
│   ├── *.port.ts
│   └── index.ts
│
├── adapters/                              # ADAPTERS (port implementations)
│   ├── mock/
│   ├── <tech>/
│   └── index.ts
│
├── logic/                                 # BUSINESS LOGIC SERVICES
│   ├── *Service.ts
│   └── *Formatter.ts
│
├── state/                                 # STATE MANAGEMENT
│   └── zustand/
│       ├── *Store.ts
│       ├── *Store.selectors.ts
│       └── *Store.actions.ts
│
├── hooks/                                 # REACT HOOKS (deps injected)
│   ├── use*.ts
│
├── orchestrators/                         # ORCHESTRATORS (wire everything)
│   ├── *Orchestrator.ts
│   └── shared/
│
├── views/                                 # PLATFORM-SPECIFIC UI
│   ├── react/
│   ├── react-native/
│   └── flutter/
│
├── types/                                 # FEATURE-SPECIFIC TYPES
│   ├── *.types.ts
│   └── index.ts
│
└── index.ts                               # Public API barrel export
```

---

## Versioned Specification System

- Specs are the canonical source of truth for feature behavior.
- Each spec file includes a semantic version and a CHANGELOG entry.
- Tests reference spec versions in their headers.
- Breaking changes require a new major version and a migration note.

---

## Ports & Adapters Explained

- **Ports** are interfaces that define how external systems are called.
- **Adapters** are concrete implementations for a specific tech stack.
- Business logic depends on ports, never on adapters.

---

## Multi-Language Strategy

- Domain and port specs are language-agnostic.
- Each platform implements the same spec versions.
- A single feature can be re-implemented in different languages without behavior drift.

---

## State Management Strategy

- State is isolated per feature.
- Stores expose selectors and actions only.
- Hooks are the only layer that binds stores to the UI.

---

## AI Agent Architecture

- Orchestrators own routing, tool fan-out, and response shaping.
- Business logic remains deterministic and pure.
- API/tool adapters return typed results and structured errors.

---

## Database & RPC Specifications

- Database schemas align with domain aggregates and repository ports.
- RPC and API contracts are versioned alongside feature specs.

---

## Code Conventions

- ASCII only unless the file already uses Unicode.
- Keep modules small and composable.
- Prefer dependency injection over direct imports across layers.

---

## Directory Structure

```
crypto/
├── __specs__/
├── __tests__/
├── nextjs/
└── (other platforms and adapters)

real-estate/
├── __specs__/
├── __tests__/
└── (placeholders / future implementations)
```

---

## Hook Integration Pattern

Use the four-hook pattern to keep UI, state, and logic modular and testable:

1. **State selector hook**: extracts reactive state variables (via dependency injection).
2. **UI hook**: owns local UI state (input value, tooltip state).
3. **Logic hook**: business logic; consumes UI state + injected dependencies.
4. **Integration hook**: composes (1–3) and returns a single object to the view.

### Dependency Injection Rules

- Hooks never import state managers or API adapters directly.
- Orchestrators extract state/actions/API functions and pass them into hooks.
- Pass dependencies as a named object (not positional args) to avoid ordering bugs.

---

## Requirement IDs & Traceability

Every requirement in a spec must include a stable numeric identifier so tests can
map 1:1 to each requirement across languages.

**Format:** `<section>.<subsection>.<item>` (example: `1.2.3`)

Rules:
- Each requirement line starts with its numeric ID.
- Tests must reference the requirement ID they validate.
- Maintain IDs across language implementations.

---

## Reading List & References

- Orc-BASH: https://medium.com/@leonaburime/the-orc-bash-pattern-orchestrated-architecture-for-maximum-reusability-5d6b4734c9f6
- SHARP Testing: https://medium.com/@leonaburime/the-sharp-pattern-a-test-first-paradigm-for-react-react-native-apps-with-ai-f1853df47390
- SUIF Testing: https://medium.com/@leonaburime/the-suif-pattern-a-simpler-pragmatic-sharp-variant-b222e731ad0b
