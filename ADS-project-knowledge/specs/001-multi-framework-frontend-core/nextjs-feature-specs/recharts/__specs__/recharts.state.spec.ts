/**
 * Spec: recharts.state.spec.ts
 * Version: 1.2.0
 */
export const RECHARTS_STATE_SPEC_VERSION = "1.2.0";

export const rechartsStateSpec = {
  id: "recharts.state",
  version: RECHARTS_STATE_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-03-03",
  stores: ["useChartStore"],
  adapters: [
    "useChartManagementAdapter",
    "useCopilotChartActionsAdapter",
  ],
  invariants: [
    "addChartSpec prepends latest and deduplicates by id.",
    "removeChartSpec only removes matching id.",
    "clearChartSpecs empties chartSpecs array.",
  ],
} as const;
