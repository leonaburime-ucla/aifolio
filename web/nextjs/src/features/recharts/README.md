# Recharts Feature

## Structure
- `logic`: framework-agnostic chart logic.
- `react/orchestrators`: React orchestrators.
- `react/hooks`: React hooks.
- `react/state`: React state adapters and stores.
- `react/views`: React chart view components.
- `__tests__`: requirement-aligned tests mirrored by architecture boundary.

Versioned specs live in `ADS-project-knowledge/specs/001-multi-framework-frontend-core/nextjs-feature-specs/recharts`.
Reusable chart contracts live in `@aifolio/contracts/entities/chart` and `@aifolio/contracts/entities/recharts`; reusable chart formatting lives in `@aifolio/frontend-core/recharts`; React-only types are colocated with their React modules.
