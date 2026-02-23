/**
 * Spec: agentic-research.ui.spec.ts
 * Version: 1.1.0
 */
export const AGENTIC_RESEARCH_UI_SPEC_VERSION = "1.1.0";

export const agenticResearchUiSpec = {
  id: "agentic-research.ui",
  version: AGENTIC_RESEARCH_UI_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  surfaces: ["DatasetCombobox", "AgenticResearchFrontendTools"],
  controlledContracts: [
    {
      component: "DatasetCombobox",
      requiredProps: ["options", "selectedId", "onChange"],
      behavior: "Pure controlled input; no internal fetching or state side effects.",
    },
  ],
  renderBranches: [
    "loading: consumer may show skeletons",
    "empty: no chart/table data available",
    "error: show user-safe error string from orchestrator/state",
    "success: render chart/table from orchestrator model",
  ],
} as const;
