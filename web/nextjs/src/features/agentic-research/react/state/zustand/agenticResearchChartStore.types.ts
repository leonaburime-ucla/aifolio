import type { ChartSpec } from "@aifolio/contracts/entities/chart";

export type AgenticResearchChartStoreState = {
  chartSpecs: ChartSpec[];
  addChartSpec: (spec: ChartSpec) => void;
  removeChartSpec: (id: string) => void;
  clearChartSpecs: () => void;
  reorderChartSpecs: (orderedIds: string[]) => void;
};
