import type { ChartSpec } from "@/features/recharts/__types__/typescript/chart.types";

export type ChartRendererProps = {
  spec: ChartSpec;
  onRemove?: (id: string) => void;
};

export type LoadingLabelProps = {
  x?: number;
  y?: number;
  value?: string | number;
};
