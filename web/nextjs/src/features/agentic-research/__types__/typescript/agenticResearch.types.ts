import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";
import type {
  DatasetRowsResponse,
  FetchPcaChartSpecPayload,
  FetchPcaChartSpecResult,
} from "@/features/agentic-research/__types__/typescript/api/agenticResearchApi.types";

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
  fetchDatasetRows: (datasetId: string) => Promise<DatasetRowsResponse>;
  fetchPcaChartSpec: (
    payload: FetchPcaChartSpecPayload
  ) => Promise<FetchPcaChartSpecResult>;
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
  removeChartSpec: (id: string) => void;
  formatToolName: (name: string) => string;
};

export type AgenticResearchChartStateSnapshot = {
  chartSpecs: ChartSpec[];
};

export type AgenticResearchChartActionsPort = {
  chartSpecs: ChartSpec[];
  addChartSpec: (spec: ChartSpec) => void;
  clearChartSpecs: () => void;
  removeChartSpec: (id: string) => void;
  reorderChartSpecs: (orderedIds: string[]) => void;
  getChartStateSnapshot: () => AgenticResearchChartStateSnapshot;
};

export type UseAgenticResearchChartActionsPort = () => AgenticResearchChartActionsPort;
