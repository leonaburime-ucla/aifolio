# Web

Monorepo for the AIfolio web frontend.

## Package Dependency Graph

```
contracts (types + Zod schemas, zero runtime deps)
    ↑
frontend-core (pure logic, depends on contracts)
    ↑
nextjs (React app, depends on both)
```

## Why `contracts` is separate from `frontend-core`

`@aifolio/contracts` holds entity types and validation schemas with **zero runtime dependencies**. `@aifolio/frontend-core` holds pure business logic that depends on those types.

Keeping them split enforces a one-way dependency: types never accidentally import runtime logic. This matters if:

- A second frontend (mobile, CLI) needs the same type contracts without pulling logic.
- A TypeScript backend shares the same entity types.
- You want to guarantee that changing logic never breaks type-only consumers.

### Could they merge?

Yes — contracts could become `frontend-core/entities/`. The Python backend doesn't consume them, and today only `frontend-core` + `nextjs` import from contracts. Merging would mean fewer packages, fewer `tsconfig` paths, and simpler DX.

**Tradeoff:** you lose the hard boundary that prevents entity definitions from importing runtime utilities. If contracts stays mostly `type` exports and Zod schemas with no logic, the risk of accidental coupling is low.

**Decision (current):** keep them separate. Revisit if the overhead of two packages outweighs the safety benefit or if no second consumer materializes.
