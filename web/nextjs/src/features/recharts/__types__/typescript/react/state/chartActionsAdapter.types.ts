import type { ChartActionsPort } from "@/features/recharts/__types__/typescript/chart.types";

export type CopilotChartActionsPort = ChartActionsPort & {
  clearChartSpecs: () => void;
};
