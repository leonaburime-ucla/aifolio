/**
 * Spec: agentic-research.state.spec.ts
 * Version: 1.1.0
 */
export const AGENTIC_RESEARCH_STATE_SPEC_VERSION = "1.1.0";

export const agenticResearchStateSpec = {
  id: "agentic-research.state",
  version: AGENTIC_RESEARCH_STATE_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  stores: ["agenticResearchStore", "agenticResearchChartStore"],
  invariants: [
    "Research state updates occur only through typed actions/adapters.",
    "addChartSpec deduplicates by id and prepends latest chart.",
    "removeChartSpec removes only matching id.",
    "reorderChartSpecs preserves unspecified ids in current order.",
  ],
  transitions: [
    {
      action: "loadDataset:start",
      before: "existing table/chart data may exist",
      after: "tableRows=[], tableColumns=[], pcaChartSpec=null, numericMatrix=[], featureNames=[]",
    },
    {
      action: "ar-set_active_dataset",
      before: "chartSpecs may be non-empty",
      after: "chartSpecs=[] then selectedDatasetId=<requested id>",
    },
  ],
} as const;
