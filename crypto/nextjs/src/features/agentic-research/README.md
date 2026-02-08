# Agentic Research Feature

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
const state = useAgenticResearchState();
const actions = useAgenticResearchActions();

actions.setSelectedDatasetId("wine-quality-red");
```

Non-reactive snapshot helpers:
```ts
const snapshot = getAgenticResearchSnapshot();
const payload = getActiveDatasetPayload(500);
```
