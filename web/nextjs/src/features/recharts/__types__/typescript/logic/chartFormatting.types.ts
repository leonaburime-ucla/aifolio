import type { ChartSpec } from "@/features/recharts/__types__/typescript/chart.types";

export type CoerceNumberParams = {
  value: unknown;
};

export type FormatValueParams = {
  value: unknown;
  spec: ChartSpec;
};

export type FormatXAxisValueParams = {
  value: unknown;
  spec: ChartSpec;
};
