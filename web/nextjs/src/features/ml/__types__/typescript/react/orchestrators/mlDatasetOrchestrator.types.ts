import type {
  MlDatasetOptionsLoader,
  MlDatasetRowsLoader,
} from "@/features/ml/__types__/typescript/api/mlDataApi.types";
import type {
  MlDatasetActions,
  MlDatasetCacheEntry,
  MlDatasetState,
} from "@/features/ml/__types__/typescript/mlData.types";

export type MlDatasetStatePort = {
  state: MlDatasetState;
  actions: MlDatasetActions;
};

export type MlDatasetOrchestratorDeps = {
  useDatasetState?: () => MlDatasetStatePort;
  loadDatasetOptions?: MlDatasetOptionsLoader;
  loadDatasetRows?: MlDatasetRowsLoader;
  autoLoad?: boolean;
};

export type MlActiveDataset = MlDatasetCacheEntry | null;

export type MlDatasetViewModel = {
  datasetOptions: { id: string; label: string }[];
  selectedDatasetId: string | null;
  setSelectedDatasetId: (datasetId: string | null) => void;
  isLoading: boolean;
  error: string | null;
  tableRows: Record<string, unknown>[];
  tableColumns: string[];
  rowCount: number;
  totalRowCount: number;
  reloadManifest: () => Promise<void>;
  reloadDataset: () => Promise<void>;
};
