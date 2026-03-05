import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";

export type AgenticResearchChartStoreState = {
  chartSpecs: ChartSpec[];
  addChartSpec: (spec: ChartSpec) => void;
  removeChartSpec: (id: string) => void;
  clearChartSpecs: () => void;
  reorderChartSpecs: (orderedIds: string[]) => void;
};
