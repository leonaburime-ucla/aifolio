import type { ChartSpec } from "@/features/recharts/__types__/typescript/chart.types";

export type UseChartRendererModelParams = {
  spec: ChartSpec;
};

export type UseChartRendererModelResult = {
  isExpanded: boolean;
  setIsExpanded: (value: boolean) => void;
  yKeys: string[];
  scatterLabelKey: string | null;
  data: Array<Record<string, unknown>>;
  chartProps: {
    data: Array<Record<string, unknown>>;
    margin: { top: number; right: number; left: number; bottom: number };
  };
};
