/**
 * Spec: recharts.api.spec.ts
 * Version: 1.1.0
 */
export const RECHARTS_API_SPEC_VERSION = "1.1.0";

export const rechartsApiSpec = {
  id: "recharts.api",
  version: RECHARTS_API_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  contracts: {
    note: "Feature is state-driven and does not issue remote HTTP requests.",
    dependency: "ChartSpec inputs are provided by upstream orchestrators/chat actions.",
  },
  deterministicRules: [
    "No raw fetch/HTTP calls in chart orchestrator, store, or renderers.",
  ],
} as const;
