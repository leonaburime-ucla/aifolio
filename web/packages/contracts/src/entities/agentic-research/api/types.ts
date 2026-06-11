import type { ChartSpec } from "../../chart/model/types";
import type { DatasetManifestEntry } from "../model/types";

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

export type AgenticResearchApiRuntimeDeps = {
  fetchImpl?: typeof fetch;
  resolveBaseUrl?: () => string;
};

export type FetchAgenticDatasetManifestInput = Record<string, never>;

export type FetchAgenticSklearnToolsInput = Record<string, never>;

export type FetchAgenticDatasetRowsInput = {
  datasetId: string;
};

export type AgenticResearchApiOptions = {
  runtimeDeps?: AgenticResearchApiRuntimeDeps;
};

export type CreateAgenticResearchApiAdapterInput = Record<string, never>;

export type CreateAgenticResearchApiAdapterOptions = {
  fetchDatasetManifest?: () => Promise<DatasetManifestEntry[]>;
  fetchSklearnTools?: () => Promise<string[]>;
  fetchDatasetRows?: (datasetId: string) => Promise<DatasetRowsResponse>;
  fetchPcaChartSpec?: (
    payload: FetchPcaChartSpecPayload
  ) => Promise<FetchPcaChartSpecResult>;
};
