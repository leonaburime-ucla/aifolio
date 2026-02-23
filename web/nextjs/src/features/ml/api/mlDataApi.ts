import { getAiApiBaseUrl } from "@/core/config/aiApi";
import type { MlDatasetOption } from "@/features/ml/types/mlData.types";

type MlDataManifestEntry = {
  id: string;
  label?: string;
  format?: string;
};

type MlDataManifestResponse = {
  status?: string;
  datasets?: MlDataManifestEntry[];
};

export type MlDatasetRowsResponse = {
  columns?: string[];
  rows?: Array<Record<string, string | number | null>>;
  rowCount?: number;
  totalRowCount?: number;
};

export const ML_WINE_DATASET_OPTIONS: MlDatasetOption[] = [
  {
    id: "customer_churn_telco.csv",
    label: "customer_churn_telco.csv",
    description: "CSV dataset from ai/ml/data",
  },
  {
    id: "house_prices_ames.csv",
    label: "house_prices_ames.csv",
    description: "CSV dataset from ai/ml/data",
  },
];

export const DEFAULT_ML_DATASET_ID = ML_WINE_DATASET_OPTIONS[0]?.id ?? null;

/**
 * Fetches available ML datasets and maps response rows to normalized option objects.
 */
export async function fetchMlDatasetOptions(): Promise<MlDatasetOption[]> {
  const response = await fetch(`${getAiApiBaseUrl()}/ml-data`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("Failed to load ML datasets.");
  }

  const payload = (await response.json()) as MlDataManifestResponse;
  const datasets = payload.datasets ?? [];
  return datasets.map((entry) => ({
    id: entry.id,
    label: entry.label ?? entry.id,
    description: entry.format
      ? `${entry.format.toUpperCase()} dataset from ai/ml/data`
      : "Dataset from ai/ml/data",
  }));
}

/**
 * Fetches row data for a single ML dataset by dataset ID.
 */
export async function fetchMlDatasetRows(
  datasetId: string
): Promise<MlDatasetRowsResponse> {
  const response = await fetch(
    `${getAiApiBaseUrl()}/ml-data/${encodeURIComponent(datasetId)}`
  );
  if (!response.ok) {
    throw new Error("Failed to load dataset rows.");
  }
  return (await response.json()) as MlDatasetRowsResponse;
}
