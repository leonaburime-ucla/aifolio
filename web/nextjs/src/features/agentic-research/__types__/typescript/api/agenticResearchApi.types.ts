import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";

export type PcaToolResult = {
  transformed?: number[][];
  explained_variance_ratio?: number[];
  feature_importance?: Array<{ feature: string; importance: number }>;
  feature_names?: string[];
};

export type PcaToolResponse = {
  status?: string;
  mode?: string;
  result?: PcaToolResult | null;
};

export type DatasetRowsResponse = {
  rows?: Array<Record<string, string | number | null>>;
  columns?: string[];
};

export type FetchPcaChartSpecPayload = {
  data: number[][];
  feature_names?: string[];
  n_components?: number;
  dataset_id?: string;
  dataset_meta?: Record<string, unknown>;
};

export type FetchPcaChartSpecResult = ChartSpec | null;
