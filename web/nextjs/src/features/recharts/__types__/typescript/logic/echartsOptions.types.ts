import type { ChartSpec } from "@/features/recharts/__types__/typescript/chart.types";

export type TreeNode = {
  name: string;
  children?: TreeNode[];
};

export type ScatterFormatterParams = {
  data?: Record<string, unknown>;
};

export type EChartsOptionBuilder = {
  spec: ChartSpec;
};
