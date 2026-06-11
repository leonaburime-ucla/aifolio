import type { DatasetManifestEntry } from "@aifolio/contracts/entities/agentic-research";
import type {
  ClearChartsResponse,
  RemoveChartSpecErrorResponse,
  RemoveChartSpecSuccessResponse,
  ReorderChartSpecsIndexErrorResponse,
  ReorderChartSpecsPayloadErrorResponse,
  ReorderChartSpecsSuccessResponse,
  SetActiveDatasetErrorResponse,
  SetActiveDatasetSuccessResponse,
} from "@aifolio/contracts/entities/agentic-research";

export type AgenticResearchAiSurface = {
  handlers: {
    addChartSpec: (payload: { chartSpec?: unknown; chartSpecs?: unknown[] }) => {
      status: "ok" | "error";
      code?: "INVALID_CHART_SPEC";
      addedCount: number;
      ids?: string[];
    };
    clearCharts: () => ClearChartsResponse;
    removeChartSpec: (chartId: string) => RemoveChartSpecSuccessResponse | RemoveChartSpecErrorResponse;
    reorderChartSpecs: (args: {
      ordered_ids?: string[];
      from_index?: number;
      to_index?: number;
    }) => ReorderChartSpecsSuccessResponse | ReorderChartSpecsIndexErrorResponse | ReorderChartSpecsPayloadErrorResponse;
    setActiveDataset: (datasetId: string) => SetActiveDatasetSuccessResponse | SetActiveDatasetErrorResponse;
  };
  datasetManifest: DatasetManifestEntry[];
};
