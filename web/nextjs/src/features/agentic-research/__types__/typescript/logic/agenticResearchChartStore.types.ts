import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";

export type AddChartSpecInput = {
  chartSpecs: ChartSpec[];
  spec: ChartSpec;
};

export type AddChartSpecResult = ChartSpec[];

export type ReorderChartSpecsInput = {
  chartSpecs: ChartSpec[];
  orderedIds: string[];
};

export type ReorderChartSpecsResult = ChartSpec[];
