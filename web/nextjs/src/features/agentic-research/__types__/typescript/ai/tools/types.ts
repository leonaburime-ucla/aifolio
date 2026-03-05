import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";

export type ChartSpecSnapshot = {
  chartSpecs: ChartSpec[];
};

export type ClearChartsResponse = {
  status: "ok";
  cleared: true;
};

export type RemoveChartSpecSuccessResponse = {
  status: "ok";
  removed_chart_id: string;
  remaining_count: number;
};

export type RemoveChartSpecErrorResponse = {
  status: "error";
  code: "CHART_NOT_FOUND";
  chart_id: string;
  available_chart_ids: string[];
};

export type ReorderChartSpecsSuccessResponse = {
  status: "ok";
  mode: "ordered_ids" | "index_move";
  chart_ids: string[];
};

export type ReorderChartSpecsIndexErrorResponse = {
  status: "error";
  code: "INDEX_OUT_OF_RANGE";
  from_index: number;
  to_index: number;
  chart_count: number;
};

export type ReorderChartSpecsPayloadErrorResponse = {
  status: "error";
  code: "INVALID_REORDER_PAYLOAD";
  hint: string;
};

export type SetActiveDatasetSuccessResponse = {
  status: "ok";
  active_dataset_id: string;
};

export type SetActiveDatasetErrorResponse = {
  status: "error";
  code: "INVALID_DATASET_ID";
  dataset_id: string;
  allowed_dataset_ids: string[];
};
