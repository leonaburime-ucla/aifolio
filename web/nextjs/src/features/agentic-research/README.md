# Agentic Research Feature

## Structure
- `api`: feature API transport + adapters.
- `logic`: framework-agnostic feature logic.
- `config`: feature configuration.
- `utils`: framework-agnostic utilities.
- `ai/tools`: framework-agnostic AI tool logic.
- `react/hooks`: React hooks.
- `react/orchestrators`: React orchestrators.
- `react/state`: React state adapters and stores.
- `react/views`: React view components.
- `react/ai`: React AI-surface adapters/views.
- `__tests__`: requirement-aligned tests mirrored by architecture boundary.

Versioned specs live in `ADS-project-knowledge/specs/001-multi-framework-frontend-core/nextjs-feature-specs/agentic-research`.
Reusable contracts live in `@aifolio/contracts/entities/agentic-research`; reusable pure logic lives in `@aifolio/frontend-core/agentic-research`; React-only types are colocated with their React modules.

## Hook Documentation Standard
- Every exported hook and utility function must include JSDoc.
- JSDoc must describe parameters (`@param`) and return values (`@returns`).
- Any stateful side effects must be called out in the description.

## Store vs Actions
- **Store state**: reactive values consumed by UI (`useAgenticResearchState`).
- **Actions**: imperative setters/mutators that update the store (`useAgenticResearchActions`).
- The orchestrator should only consume the adapter hooks, not the store implementation.

Example:
```ts
const { state, actions } = useAgenticResearchStateAdapter();

actions.setSelectedDatasetId("wine-quality-red");
```

Chart state port example:
```ts
const { chartSpecs } = useAgenticResearchChartActionsAdapter();
```

Non-reactive snapshot helpers:
```ts
const snapshot = getAgenticResearchSnapshot();
const payload = getActiveDatasetPayload(500);
```
