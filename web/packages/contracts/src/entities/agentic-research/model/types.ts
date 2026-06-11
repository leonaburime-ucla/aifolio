import type { ChartSpec } from "../../chart/model/types";
import type {
  DatasetRowsResponse,
  FetchPcaChartSpecPayload,
  FetchPcaChartSpecResult,
} from "../api/types";

// --- Domain types ---

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

// --- Logic types ---

export type ApplyDatasetLoadResetInput = {
  actions: Pick<
    AgenticResearchActions,
    | "setTableRows"
    | "setTableColumns"
    | "setNumericMatrix"
    | "setFeatureNames"
    | "setPcaChartSpec"
  >;
};

export type ResolveActiveChartSpecInput = {
  pcaChartSpec: ChartSpec | null;
  chartSpecs: ChartSpec[];
};

export type ResolveChartToolNameInput = {
  name: string;
};

export type ResolveChartToolNameOptions = {
  acronyms?: Record<string, string>;
};

export type ResolveDefaultDatasetIdInput = {
  selectedDatasetId: string | null;
  datasets: DatasetManifestEntry[];
};

export type ToDatasetOptionsInput = {
  datasetManifest: DatasetManifestEntry[];
};

export type ToDatasetOptionsResult = DatasetOption[];

export type GroupSklearnToolsInput = {
  tools: string[];
};

export type GroupSklearnToolsResult = Record<string, string[]>;

export type AddChartSpecInput = {
  chartSpecs: ChartSpec[];
  spec: ChartSpec;
};

export type AddChartSpecResult = ChartSpec[];

export type ReorderChartSpecsInput = {
  chartSpecs: ChartSpec[];
  orderedIds: string[];
};

export type ReorderChartSpecsResult = ChartSpec[];

// --- Utils types ---

export type ParsedRow = Record<string, string | number | null>;

// --- AI tool response types ---

export type ChartSpecSnapshot = {
  chartSpecs: ChartSpec[];
};

export type ClearChartsResponse = {
  status: "ok";
  cleared: true;
};

export type RemoveChartSpecSuccessResponse = {
  status: "ok";
  removed_chart_id: string;
  remaining_count: number;
};

export type RemoveChartSpecErrorResponse = {
  status: "error";
  code: "CHART_NOT_FOUND";
  chart_id: string;
  available_chart_ids: string[];
};

export type ReorderChartSpecsSuccessResponse = {
  status: "ok";
  mode: "ordered_ids" | "index_move";
  chart_ids: string[];
};

export type ReorderChartSpecsIndexErrorResponse = {
  status: "error";
  code: "INDEX_OUT_OF_RANGE";
  from_index: number;
  to_index: number;
  chart_count: number;
};

export type ReorderChartSpecsPayloadErrorResponse = {
  status: "error";
  code: "INVALID_REORDER_PAYLOAD";
  hint: string;
};

export type SetActiveDatasetSuccessResponse = {
  status: "ok";
  active_dataset_id: string;
};

export type SetActiveDatasetErrorResponse = {
  status: "error";
  code: "INVALID_DATASET_ID";
  dataset_id: string;
  allowed_dataset_ids: string[];
};
