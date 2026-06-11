import type { ChartSpec } from "@aifolio/contracts/entities/chart";

export type ChartStoreState = {
  chartSpecs: ChartSpec[];
  addChartSpec: (spec: ChartSpec) => void;
  removeChartSpec: (id: string) => void;
  clearChartSpecs: () => void;
};
