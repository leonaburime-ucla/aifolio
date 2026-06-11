# ML Feature

## Structure
- `api`: feature API transport and adapters.
- `config`: feature configuration and defaults.
- `utils`: framework-agnostic utilities.
- `validators`: validation helpers.
- `react/orchestrators`: React orchestrators.
- `react/hooks`: React hooks.
- `react/state`: React state adapters and stores.
- `react/views`: React view components.

## Notes
- Keep route files under `app/ml/*` thin and composed from this feature slice.
- Long-running jobs should be started server-side and polled or streamed in the UI.
- Versioned specs live in `ADS-project-knowledge/specs/001-multi-framework-frontend-core/nextjs-feature-specs/ml`.
- Reusable contracts live in `@aifolio/contracts/entities/ml-training`; reusable pure logic lives in `@aifolio/frontend-core/ml-training`; React-only types are colocated with their React modules.
