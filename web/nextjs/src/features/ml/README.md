# ML Feature

## Structure
- `typescript/api`: feature API transport and adapters.
- `typescript/config`: feature configuration and defaults.
- `typescript/utils`: framework-agnostic utilities.
- `typescript/validators`: validation helpers.
- `typescript/react/orchestrators`: React orchestrators.
- `typescript/react/hooks`: React hooks.
- `typescript/react/state`: React state adapters and stores.
- `typescript/react/views`: React view components.
- `__types__/typescript`: feature type contracts.

## Notes
- Keep route files under `app/ml/*` thin and composed from this feature slice.
- Long-running jobs should be started server-side and polled or streamed in the UI.
