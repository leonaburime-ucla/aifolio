import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";
import type { MlDatasetActions, MlDatasetState } from "@/features/ml/types/mlData.types";

type MlDatasetStore = MlDatasetState & MlDatasetActions;

const mlDatasetStore = create<MlDatasetStore>((set) => ({
  datasetOptions: [],
  selectedDatasetId: null,
  datasetCache: {},
  manifestLoaded: false,
  isLoadingManifest: false,
  isLoadingDataset: false,
  error: null,
  setDatasetOptions: (value) => set({ datasetOptions: value }),
  setSelectedDatasetId: (value) => set({ selectedDatasetId: value }),
  setDatasetCacheEntry: (datasetId, value) =>
    set((store) => ({ datasetCache: { ...store.datasetCache, [datasetId]: value } })),
  setManifestLoaded: (value) => set({ manifestLoaded: value }),
  setLoadingManifest: (value) => set({ isLoadingManifest: value }),
  setLoadingDataset: (value) => set({ isLoadingDataset: value }),
  setError: (value) => set({ error: value }),
}));

export function useMlDatasetState(): MlDatasetState {
  return mlDatasetStore(
    useShallow((store): MlDatasetState => ({
      datasetOptions: store.datasetOptions,
      selectedDatasetId: store.selectedDatasetId,
      datasetCache: store.datasetCache,
      manifestLoaded: store.manifestLoaded,
      isLoadingManifest: store.isLoadingManifest,
      isLoadingDataset: store.isLoadingDataset,
      error: store.error,
    }))
  );
}

export function useMlDatasetActions(): MlDatasetActions {
  return mlDatasetStore(
    useShallow((store): MlDatasetActions => ({
      setDatasetOptions: store.setDatasetOptions,
      setSelectedDatasetId: store.setSelectedDatasetId,
      setDatasetCacheEntry: store.setDatasetCacheEntry,
      setManifestLoaded: store.setManifestLoaded,
      setLoadingManifest: store.setLoadingManifest,
      setLoadingDataset: store.setLoadingDataset,
      setError: store.setError,
    }))
  );
}

export { mlDatasetStore };
