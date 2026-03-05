import type { ChartSpec } from "@/features/recharts/__types__/typescript/chart.types";

export type ChartStoreState = {
  chartSpecs: ChartSpec[];
  addChartSpec: (spec: ChartSpec) => void;
  removeChartSpec: (id: string) => void;
  clearChartSpecs: () => void;
};
