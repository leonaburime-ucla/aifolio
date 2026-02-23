/**
 * Spec: recharts.state.spec.ts
 * Version: 1.1.0
 */
export const RECHARTS_STATE_SPEC_VERSION = "1.1.0";

export const rechartsStateSpec = {
  id: "recharts.state",
  version: RECHARTS_STATE_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  stores: ["useChartStore"],
  adapters: [
    "useRechartsChartStateAdapter",
    "useRechartsChartActionsAdapter",
    "useChartManagementAdapter",
  ],
  invariants: [
    "addChartSpec prepends latest and deduplicates by id.",
    "removeChartSpec only removes matching id.",
    "clearChartSpecs empties chartSpecs array.",
  ],
} as const;
