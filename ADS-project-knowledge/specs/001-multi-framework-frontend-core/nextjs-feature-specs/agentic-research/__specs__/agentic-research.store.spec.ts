/**
 * Spec: agentic-research.store.spec.ts
 * Version: 1.1.0
 */
export const AGENTIC_RESEARCH_STORE_SPEC_VERSION = "1.1.0";

export const agenticResearchStoreSpec = {
  id: "agentic-research.store",
  version: AGENTIC_RESEARCH_STORE_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  stores: ["agenticResearchStore", "agenticResearchChartStore"],
  requirements: [
    "Research store setters are single-purpose and do not perform hidden cross-field mutations.",
    "Chart store addChartSpec prepends latest chart and deduplicates by chart id.",
    "Chart store removeChartSpec removes by exact chart id only.",
    "Chart store reorderChartSpecs preserves unspecified chart ids by appending them in current order.",
    "getAgenticResearchSnapshot returns a read-only projection of current store state.",
    "getActiveDatasetPayload enforces maxRows truncation for rows and numericMatrix using the same upper bound.",
  ],
} as const;
