import type {
  MlDatasetOptionsLoader,
  MlDatasetRowsLoader,
} from "@/features/ml/__types__/typescript/api/mlDataApi.types";
import type { MlDatasetCacheEntry } from "@/features/ml/__types__/typescript/mlData.types";
import type { useMlDatasetStateAdapter } from "@/features/ml/typescript/react/state/adapters/mlDataset.adapter";

export type MlDatasetOrchestratorDeps = {
  useDatasetState?: typeof useMlDatasetStateAdapter;
  loadDatasetOptions?: MlDatasetOptionsLoader;
  loadDatasetRows?: MlDatasetRowsLoader;
  autoLoad?: boolean;
};

export type MlActiveDataset = MlDatasetCacheEntry | null;
