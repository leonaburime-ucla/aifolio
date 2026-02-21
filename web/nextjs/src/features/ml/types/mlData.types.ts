export type MlDatasetCacheEntry = {
  columns: string[];
  rows: Array<Record<string, string | number | null>>;
  rowCount: number;
  totalRowCount: number;
};

export type MlDatasetState = {
  datasetOptions: import("@/core/views/patterns/CsvDatasetCombobox").CsvDatasetOption[];
  selectedDatasetId: string | null;
  datasetCache: Record<string, MlDatasetCacheEntry>;
  manifestLoaded: boolean;
  isLoadingManifest: boolean;
  isLoadingDataset: boolean;
  error: string | null;
};

export type MlDatasetActions = {
  setDatasetOptions: (
    value: import("@/core/views/patterns/CsvDatasetCombobox").CsvDatasetOption[]
  ) => void;
  setSelectedDatasetId: (value: string | null) => void;
  setDatasetCacheEntry: (datasetId: string, value: MlDatasetCacheEntry) => void;
  setManifestLoaded: (value: boolean) => void;
  setLoadingManifest: (value: boolean) => void;
  setLoadingDataset: (value: boolean) => void;
  setError: (value: string | null) => void;
};
