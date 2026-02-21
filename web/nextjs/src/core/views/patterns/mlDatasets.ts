import type { CsvDatasetOption } from "@/core/views/patterns/CsvDatasetCombobox";

const AI_API_BASE_URL =
  process.env.NEXT_PUBLIC_AI_API_URL || "http://127.0.0.1:8000";

type MlDataManifestEntry = {
  id: string;
  label?: string;
  format?: string;
};

type MlDataManifestResponse = {
  status?: string;
  datasets?: MlDataManifestEntry[];
};

export async function fetchMlDatasetOptions(): Promise<CsvDatasetOption[]> {
  const response = await fetch(`${AI_API_BASE_URL}/ml-data`, {
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

// Backward-compatible fallback options for pages not migrated yet.
export const ML_WINE_DATASET_OPTIONS: CsvDatasetOption[] = [
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
