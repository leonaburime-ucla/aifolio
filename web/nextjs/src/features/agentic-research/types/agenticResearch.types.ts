import type { ChartSpec } from "@/features/ai/types/chart.types";

export type DatasetManifestEntry = {
  id: string;
  label: string;
  description?: string;
  files?: {
    data: string;
    names?: string;
  };
  task?: string;
  targetColumn?: string;
  source?: string;
  metadata?: {
    context?: string;
    files?: {
      data?: string;
      names?: string;
    };
    [key: string]: unknown;
  };
};

export type DatasetOption = {
  id: string;
  label: string;
  description?: string;
};

export type AgenticResearchState = {
  datasetManifest: DatasetManifestEntry[];
  selectedDatasetId: string | null;
  sklearnTools: string[];
  tableRows: Array<Record<string, string | number | null>>;
  tableColumns: string[];
  numericMatrix: number[][];
  featureNames: string[];
  pcaChartSpec: ChartSpec | null;
  isLoading: boolean;
  error: string | null;
};

export type AgenticResearchActions = {
  setDatasetManifest: (value: DatasetManifestEntry[]) => void;
  setSelectedDatasetId: (value: string | null) => void;
  setSklearnTools: (value: string[]) => void;
  setTableRows: (value: Array<Record<string, string | number | null>>) => void;
  setTableColumns: (value: string[]) => void;
  setNumericMatrix: (value: number[][]) => void;
  setFeatureNames: (value: string[]) => void;
  setPcaChartSpec: (value: ChartSpec | null) => void;
  setLoading: (value: boolean) => void;
  setError: (value: string | null) => void;
};

export type AgenticResearchApiDeps = {
  fetchDatasetManifest: () => Promise<DatasetManifestEntry[]>;
  fetchSklearnTools: () => Promise<string[]>;
  fetchDatasetRows: (datasetId: string) => Promise<{
    rows?: Array<Record<string, string | number | null>>;
    columns?: string[];
  }>;
  fetchPcaChartSpec: (payload: {
    data: number[][];
    feature_names?: string[];
    n_components?: number;
    dataset_id?: string;
    dataset_meta?: Record<string, unknown>;
  }) => Promise<ChartSpec | null>;
};

export type AgenticResearchDeps = {
  state: AgenticResearchState;
  actions: AgenticResearchActions;
  api: AgenticResearchApiDeps;
};

export type AgenticResearchStatePort = {
  state: AgenticResearchState;
  actions: AgenticResearchActions;
};

export type UseAgenticResearchStatePort = () => AgenticResearchStatePort;

export type AgenticResearchIntegration = AgenticResearchState & {
  groupedTools: Record<string, string[]>;
  datasetOptions: DatasetOption[];
  reloadManifest: () => void;
  setSelectedDatasetId: (value: string | null) => void;
};

export type AgenticResearchOrchestratorModel = AgenticResearchIntegration & {
  activeChartSpec: ChartSpec | null;
  chartSpecs: ChartSpec[];
  formatToolName: (name: string) => string;
};

export type AgenticResearchChartStatePort = {
  chartSpecs: ChartSpec[];
};

export type UseAgenticResearchChartStatePort = () => AgenticResearchChartStatePort;

export type AgenticResearchChartStateSnapshot = {
  chartSpecs: ChartSpec[];
};

export type AgenticResearchChartActionsPort = {
  addChartSpec: (spec: ChartSpec) => void;
  clearChartSpecs: () => void;
  removeChartSpec: (id: string) => void;
  reorderChartSpecs: (orderedIds: string[]) => void;
  getChartStateSnapshot: () => AgenticResearchChartStateSnapshot;
};

export type UseAgenticResearchChartActionsPort = () => AgenticResearchChartActionsPort;
