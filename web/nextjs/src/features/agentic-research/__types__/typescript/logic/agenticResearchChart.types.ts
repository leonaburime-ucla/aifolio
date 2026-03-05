import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";

export type ResolveActiveChartSpecInput = {
  pcaChartSpec: ChartSpec | null;
  chartSpecs: ChartSpec[];
};

export type ResolveChartToolNameInput = {
  name: string;
};

export type ResolveChartToolNameOptions = {
  acronyms?: Record<string, string>;
};
