import type { ChartSpec } from "../../chart/model/types";

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
