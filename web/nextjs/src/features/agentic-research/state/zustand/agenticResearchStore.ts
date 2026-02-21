import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";
import type {
  AgenticResearchActions,
  AgenticResearchState,
} from "@/features/agentic-research/types/agenticResearch.types";

type AgenticResearchStore = AgenticResearchState & AgenticResearchActions;

/**
 * Internal Zustand store for agentic research data.
 */
const agenticResearchStore = create<AgenticResearchStore>((set) => ({
  datasetManifest: [],
  selectedDatasetId: null,
  sklearnTools: [],
  tableRows: [],
  tableColumns: [],
  numericMatrix: [],
  featureNames: [],
  pcaChartSpec: null,
  isLoading: false,
  error: null,
  setDatasetManifest: (value) => set({ datasetManifest: value }),
  setSelectedDatasetId: (value) => set({ selectedDatasetId: value }),
  setSklearnTools: (value) => set({ sklearnTools: value }),
  setTableRows: (value) => set({ tableRows: value }),
  setTableColumns: (value) => set({ tableColumns: value }),
  setNumericMatrix: (value) => set({ numericMatrix: value }),
  setFeatureNames: (value) => set({ featureNames: value }),
  setPcaChartSpec: (value) => set({ pcaChartSpec: value }),
  setLoading: (value) => set({ isLoading: value }),
  setError: (value) => set({ error: value }),
}));

/**
 * React hook that returns the reactive agentic research state slice.
 * @returns Selected state values for agentic research.
 */
export function useAgenticResearchState(): AgenticResearchState {
  return agenticResearchStore(
    useShallow((store): AgenticResearchState => ({
      datasetManifest: store.datasetManifest,
      selectedDatasetId: store.selectedDatasetId,
      sklearnTools: store.sklearnTools,
      tableRows: store.tableRows,
      tableColumns: store.tableColumns,
      numericMatrix: store.numericMatrix,
      featureNames: store.featureNames,
      pcaChartSpec: store.pcaChartSpec,
      isLoading: store.isLoading,
      error: store.error,
    }))
  );
}

/**
 * Getter for imperative state mutation actions.
 * @returns Stable action functions for updating agentic research state.
 */
export function useAgenticResearchActions(): AgenticResearchActions {
  return agenticResearchStore(
    useShallow((store): AgenticResearchActions => ({
      setDatasetManifest: store.setDatasetManifest,
      setSelectedDatasetId: store.setSelectedDatasetId,
      setSklearnTools: store.setSklearnTools,
      setTableRows: store.setTableRows,
      setTableColumns: store.setTableColumns,
      setNumericMatrix: store.setNumericMatrix,
      setFeatureNames: store.setFeatureNames,
      setPcaChartSpec: store.setPcaChartSpec,
      setLoading: store.setLoading,
      setError: store.setError,
    }))
  );
}

/**
 * Snapshot helper for non-reactive consumers (e.g., API payload builders).
 * @returns Current agentic research state values.
 */
export function getAgenticResearchSnapshot(): AgenticResearchState {
  const {
    datasetManifest,
    selectedDatasetId,
    sklearnTools,
    tableRows,
    tableColumns,
    numericMatrix,
    featureNames,
    pcaChartSpec,
    isLoading,
    error,
  } = agenticResearchStore.getState();
  return {
    datasetManifest,
    selectedDatasetId,
    sklearnTools,
    tableRows,
    tableColumns,
    numericMatrix,
    featureNames,
    pcaChartSpec,
    isLoading,
    error,
  };
}

/**
 * Build a lightweight payload for sending the active dataset to the backend.
 * @param maxRows - Maximum number of rows to include.
 * @returns Minimal dataset payload for analysis requests.
 */
export function getActiveDatasetPayload(maxRows = 500) {
  const {
    selectedDatasetId,
    tableRows,
    tableColumns,
    featureNames,
    numericMatrix,
  } = agenticResearchStore.getState();

  return {
    datasetId: selectedDatasetId,
    columns: tableColumns,
    rows: tableRows.slice(0, maxRows),
    featureNames,
    numericMatrix: numericMatrix.slice(0, maxRows),
  };
}

/**
 * Export the raw store for advanced integrations or tests.
 */
export { agenticResearchStore };
