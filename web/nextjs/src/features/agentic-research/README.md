# Agentic Research Feature

## Structure
- `typescript/api`: feature API transport + adapters.
- `typescript/logic`: framework-agnostic feature logic.
- `typescript/config`: feature configuration.
- `typescript/utils`: framework-agnostic utilities.
- `typescript/ai/tools`: framework-agnostic AI tool logic.
- `typescript/react/hooks`: React hooks.
- `typescript/react/orchestrators`: React orchestrators.
- `typescript/react/state`: React state adapters and stores.
- `typescript/react/views`: React view components.
- `typescript/react/ai`: React AI-surface adapters/views.
- `__types__/typescript`: feature type contracts.
- `__types__/typescript/logic`: framework-agnostic logic contracts.
- `__tests__/typescript`: requirement-aligned tests mirrored by architecture boundary.

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
