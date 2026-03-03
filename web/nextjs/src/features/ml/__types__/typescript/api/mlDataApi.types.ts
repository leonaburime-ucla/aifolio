import type { MlDatasetOption } from "@/features/ml/__types__/typescript/mlData.types";

export type MlDataManifestEntry = {
  id: string;
  label?: string;
  format?: string;
};

export type MlDataManifestResponse = {
  status?: string;
  datasets?: MlDataManifestEntry[];
};

export type MlDatasetRowsResponse = {
  columns?: string[];
  rows?: Array<Record<string, string | number | null>>;
  rowCount?: number;
  totalRowCount?: number;
};

export type MlDatasetOptionsLoader = () => Promise<MlDatasetOption[]>;
export type MlDatasetRowsLoader = (
  params: { datasetId: string }
) => Promise<MlDatasetRowsResponse>;

export type MlDataApiRuntime = {
  fetchImpl: typeof fetch;
  resolveBaseUrl: () => string;
};
