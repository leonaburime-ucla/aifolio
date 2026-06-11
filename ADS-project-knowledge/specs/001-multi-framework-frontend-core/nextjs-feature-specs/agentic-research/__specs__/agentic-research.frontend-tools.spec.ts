/**
 * Spec: agentic-research.frontend-tools.spec.ts
 * Version: 1.1.0
 */
export const AGENTIC_RESEARCH_FRONTEND_TOOLS_SPEC_VERSION = "1.1.0";

export const agenticResearchFrontendToolsSpec = {
  id: "agentic-research.frontend-tools",
  version: AGENTIC_RESEARCH_FRONTEND_TOOLS_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  tools: {
    add: ["ar-add_chart_spec", "add_chart_spec"],
    clear: ["ar-clear_charts", "clear_charts"],
    remove: ["ar-remove_chart_spec"],
    reorder: ["ar-reorder_chart_specs"],
    setActiveDataset: ["ar-set_active_dataset"],
  },
  requirements: [
    "Add-chart tools accept either chartSpec or chartSpecs and route through shared chart validation/orchestration.",
    "Remove-chart returns CHART_NOT_FOUND when chart_id does not exist.",
    "Reorder-chart returns INDEX_OUT_OF_RANGE for invalid index moves.",
    "Reorder-chart returns INVALID_REORDER_PAYLOAD when neither ordered_ids nor valid from_index/to_index are provided.",
    "Set-active-dataset returns INVALID_DATASET_ID when requested id is not in manifest.",
    "Set-active-dataset clears local chart specs before switching dataset.",
  ],
} as const;
