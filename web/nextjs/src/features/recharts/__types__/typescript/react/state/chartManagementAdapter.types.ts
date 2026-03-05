import type { ChartSpec } from "@/features/recharts/__types__/typescript/chart.types";

export type ChartManagementPort = {
  chartSpecs: ChartSpec[];
  removeChartSpec: (id: string) => void;
  clearChartSpecs: () => void;
};
