export const DEFAULT_ML_DATASET_ID = "customer_churn_telco.csv";

export type DatasetIdOption = {
  id: string;
};

export type ResolvePreferredDatasetIdInput<TDataset extends DatasetIdOption> = {
  selectedDatasetId: string | null;
  datasets: TDataset[];
  preferredDatasetId?: string;
};

export function resolvePreferredDatasetId<TDataset extends DatasetIdOption>(
  input: ResolvePreferredDatasetIdInput<TDataset>
): string | null {
  if (input.selectedDatasetId) return input.selectedDatasetId;

  const preferredDatasetId = input.preferredDatasetId ?? DEFAULT_ML_DATASET_ID;
  return (
    input.datasets.find((dataset) => dataset.id === preferredDatasetId)?.id ??
    input.datasets[0]?.id ??
    null
  );
}
