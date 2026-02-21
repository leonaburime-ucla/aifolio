# Project Learnings

## 2026-02-09: Infinite fetch loop from broad `useEffect` dependencies

Primary cause: `useEffect` repeatedly re-ran because its dependency (`load`) kept changing identity.
Why `load` changed: `useCallback` depended on a broad object (for example `actions`) that changed across renders.

Rule:
- In `useEffect`, avoid broad object dependencies (directly or indirectly through callbacks).
- Destructure and depend on narrow, stable primitives/functions only.

```ts
// Avoid
const load = useCallback(async () => {
  actions.setLoading(true);
}, [actions]);

useEffect(() => {
  load();
}, [load]); // reruns if load is recreated
```

```ts
// Prefer
const { setLoading } = actions;
const load = useCallback(async () => {
  setLoading(true);
}, [setLoading, selectedDatasetId]);
```
