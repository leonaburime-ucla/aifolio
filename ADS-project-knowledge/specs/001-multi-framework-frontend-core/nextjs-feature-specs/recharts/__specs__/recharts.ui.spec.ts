/**
 * Spec: recharts.ui.spec.ts
 * Version: 1.1.0
 */
export const RECHARTS_UI_SPEC_VERSION = "1.1.0";

export const rechartsUiSpec = {
  id: "recharts.ui",
  version: RECHARTS_UI_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  components: ["ChartRenderer", "EChartsRenderer"],
  renderBranches: [
    "ECharts path for heatmap/box/dendrogram",
    "fallback unsupported panel for violin/surface",
    "Recharts path for area/scatter/bar/line variants",
  ],
  formattingContracts: [
    "formatValue applies currency formatter when spec.currency exists.",
    "formatValue appends unit when spec.unit exists and currency absent.",
    "formatXAxisValue handles year/date heuristics and scatter precision formatting.",
  ],
} as const;
