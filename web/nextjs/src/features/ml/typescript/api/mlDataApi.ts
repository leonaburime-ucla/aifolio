import { getAiApiBaseUrl } from "@/core/config/aiApi";
import type {
  MlDataApiRuntime,
  MlDataManifestResponse,
  MlDatasetRowsResponse,
} from "@/features/ml/__types__/typescript/api/mlDataApi.types";
import type { MlDatasetOption } from "@/features/ml/__types__/typescript/mlData.types";
export type { MlDatasetRowsResponse } from "@/features/ml/__types__/typescript/api/mlDataApi.types";

export const ML_WINE_DATASET_OPTIONS: MlDatasetOption[] = [
  {
    id: "customer_churn_telco.csv",
    label: "customer_churn_telco.csv",
    description: "CSV dataset from backend/data/ml_data",
  },
  {
    id: "house_prices_ames.csv",
    label: "house_prices_ames.csv",
    description: "CSV dataset from backend/data/ml_data",
  },
];

export const DEFAULT_ML_DATASET_ID = ML_WINE_DATASET_OPTIONS[0].id;

/**
 * Fetches available ML datasets and maps response rows to normalized option objects.
 */
export async function fetchMlDatasetOptions(
  {
    fetchImpl = fetch,
    resolveBaseUrl = getAiApiBaseUrl,
  }: Partial<MlDataApiRuntime> = {}
): Promise<MlDatasetOption[]> {
  const response = await fetchImpl(`${resolveBaseUrl()}/ml-data`, {
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
      ? `${entry.format.toUpperCase()} dataset from backend/data/ml_data`
      : "Dataset from backend/data/ml_data",
  }));
}

/**
 * Fetches row data for a single ML dataset by dataset ID.
 */
export async function fetchMlDatasetRows(
  {
    datasetId,
  }: {
    datasetId: string;
  },
  {
    fetchImpl = fetch,
    resolveBaseUrl = getAiApiBaseUrl,
  }: Partial<MlDataApiRuntime> = {}
): Promise<MlDatasetRowsResponse> {
  const response = await fetchImpl(
    `${resolveBaseUrl()}/ml-data/${encodeURIComponent(datasetId)}`
  );
  if (!response.ok) {
    throw new Error("Failed to load dataset rows.");
  }
  return (await response.json()) as MlDatasetRowsResponse;
}
