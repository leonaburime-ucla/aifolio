import type { ChartSpec } from "@/features/ai/types/chart.types";
import type { DatasetManifestEntry } from "@/features/agentic-research/types/agenticResearch.types";
import { useAgenticResearchChartActionsAdapter } from "@/features/agentic-research/state/adapters/chartActions.adapter";
import { useAgenticResearchStateAdapter } from "@/features/agentic-research/state/adapters/agenticResearchState.adapter";
import { handleAddChartSpec } from "@/features/copilot-chat/orchestrators/frontendTools.orchestrator";
import { useCallback, useMemo } from "react";

type ChartSpecSnapshot = {
  chartSpecs: ChartSpec[];
};

type ClearChartsResponse = {
  status: "ok";
  cleared: true;
};

type RemoveChartSpecSuccessResponse = {
  status: "ok";
  removed_chart_id: string;
  remaining_count: number;
};

type RemoveChartSpecErrorResponse = {
  status: "error";
  code: "CHART_NOT_FOUND";
  chart_id: string;
  available_chart_ids: string[];
};

type ReorderChartSpecsSuccessResponse = {
  status: "ok";
  mode: "ordered_ids" | "index_move";
  chart_ids: string[];
};

type ReorderChartSpecsIndexErrorResponse = {
  status: "error";
  code: "INDEX_OUT_OF_RANGE";
  from_index: number;
  to_index: number;
  chart_count: number;
};

type ReorderChartSpecsPayloadErrorResponse = {
  status: "error";
  code: "INVALID_REORDER_PAYLOAD";
  hint: string;
};

type SetActiveDatasetSuccessResponse = {
  status: "ok";
  active_dataset_id: string;
};

type SetActiveDatasetErrorResponse = {
  status: "error";
  code: "INVALID_DATASET_ID";
  dataset_id: string;
  allowed_dataset_ids: string[];
};

/**
 * Return type for the frontend tools orchestrator hook.
 */
export type AgenticResearchFrontendToolsOrchestrator = {
  handlers: {
    addChartSpec: (payload: { chartSpec?: unknown; chartSpecs?: unknown[] }) => ReturnType<typeof handleAddChartSpec>;
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

/**
 * Handler for clearing all charts.
 */
export function handleClearCharts(clearFn: () => void): ClearChartsResponse {
  clearFn();
  return { status: "ok", cleared: true };
}

/**
 * Handler for removing a chart spec by ID.
 */
export function handleRemoveChartSpec(
  chartId: string,
  getSnapshot: () => ChartSpecSnapshot,
  removeFn: (id: string) => void
): RemoveChartSpecSuccessResponse | RemoveChartSpecErrorResponse {
  const current = getSnapshot().chartSpecs;
  const exists = current.some((spec) => spec.id === chartId);

  if (!exists) {
    return {
      status: "error",
      code: "CHART_NOT_FOUND",
      chart_id: chartId,
      available_chart_ids: current.map((spec) => spec.id),
    };
  }

  removeFn(chartId);

  return {
    status: "ok",
    removed_chart_id: chartId,
    remaining_count: getSnapshot().chartSpecs.length,
  };
}

/**
 * Handler for reordering chart specs.
 * Supports two modes:
 * 1. ordered_ids: Provide an array of chart IDs in the desired order
 * 2. index_move: Provide from_index and to_index to move a single chart
 */
export function handleReorderChartSpecs(
  args: {
    ordered_ids?: string[];
    from_index?: number;
    to_index?: number;
  },
  getSnapshot: () => ChartSpecSnapshot,
  reorderFn: (orderedIds: string[]) => void
):
  | ReorderChartSpecsSuccessResponse
  | ReorderChartSpecsIndexErrorResponse
  | ReorderChartSpecsPayloadErrorResponse {
  const { ordered_ids, from_index, to_index } = args;
  const current = getSnapshot().chartSpecs;

  // Mode 1: ordered_ids array
  if (Array.isArray(ordered_ids) && ordered_ids.length > 0) {
    reorderFn(ordered_ids);
    return {
      status: "ok",
      mode: "ordered_ids",
      chart_ids: getSnapshot().chartSpecs.map((spec) => spec.id),
    };
  }

  // Mode 2: index-based move
  if (
    typeof from_index === "number" &&
    typeof to_index === "number" &&
    Number.isInteger(from_index) &&
    Number.isInteger(to_index)
  ) {
    if (
      from_index < 0 ||
      from_index >= current.length ||
      to_index < 0 ||
      to_index >= current.length
    ) {
      return {
        status: "error",
        code: "INDEX_OUT_OF_RANGE",
        from_index,
        to_index,
        chart_count: current.length,
      };
    }

    const ids = current.map((spec) => spec.id);
    const [moved] = ids.splice(from_index, 1);
    ids.splice(to_index, 0, moved);
    reorderFn(ids);

    return {
      status: "ok",
      mode: "index_move",
      chart_ids: getSnapshot().chartSpecs.map((spec) => spec.id),
    };
  }

  // Invalid payload
  return {
    status: "error",
    code: "INVALID_REORDER_PAYLOAD",
    hint: "Provide ordered_ids or both from_index and to_index.",
  };
}

/**
 * Handler for setting the active dataset.
 * Validates the dataset ID, clears existing charts, and sets the new dataset.
 */
export function handleSetActiveDataset(
  datasetId: string,
  allowedIds: string[],
  clearChartsFn: () => void,
  setDatasetFn: (id: string) => void
): SetActiveDatasetSuccessResponse | SetActiveDatasetErrorResponse {
  if (!allowedIds.includes(datasetId)) {
    return {
      status: "error",
      code: "INVALID_DATASET_ID",
      dataset_id: datasetId,
      allowed_dataset_ids: allowedIds,
    };
  }

  clearChartsFn();
  setDatasetFn(datasetId);

  return {
    status: "ok",
    active_dataset_id: datasetId,
  };
}

/**
 * Orchestrator hook for Agentic Research frontend tools.
 * Wires adapters to handler functions and exposes ready-to-use handlers to the view.
 * The view component should ONLY import this hook, not the adapters directly.
 */
export function useAgenticResearchFrontendToolsOrchestrator(): AgenticResearchFrontendToolsOrchestrator {
  // Import adapters internally - view should not import these directly
  const chartActions = useAgenticResearchChartActionsAdapter();
  const { state, actions } = useAgenticResearchStateAdapter();

  // Create stable handler references
  const handleAddChart = useCallback(
    (payload: { chartSpec?: unknown; chartSpecs?: unknown[] }) => {
      return handleAddChartSpec(payload, chartActions.addChartSpec);
    },
    [chartActions.addChartSpec]
  );

  const handleClear = useCallback(() => {
    return handleClearCharts(chartActions.clearChartSpecs);
  }, [chartActions.clearChartSpecs]);

  const handleRemove = useCallback(
    (chartId: string) => {
      return handleRemoveChartSpec(
        chartId,
        chartActions.getChartStateSnapshot,
        chartActions.removeChartSpec
      );
    },
    [chartActions.getChartStateSnapshot, chartActions.removeChartSpec]
  );

  const handleReorder = useCallback(
    (args: { ordered_ids?: string[]; from_index?: number; to_index?: number }) => {
      return handleReorderChartSpecs(
        args,
        chartActions.getChartStateSnapshot,
        chartActions.reorderChartSpecs
      );
    },
    [chartActions.getChartStateSnapshot, chartActions.reorderChartSpecs]
  );

  const handleSetDataset = useCallback(
    (datasetId: string) => {
      const allowedIds = state.datasetManifest.map((entry) => entry.id);
      return handleSetActiveDataset(
        datasetId,
        allowedIds,
        chartActions.clearChartSpecs,
        actions.setSelectedDatasetId
      );
    },
    [state.datasetManifest, chartActions.clearChartSpecs, actions.setSelectedDatasetId]
  );

  // Memoize handlers object to prevent unnecessary re-renders
  const handlers = useMemo(
    () => ({
      addChartSpec: handleAddChart,
      clearCharts: handleClear,
      removeChartSpec: handleRemove,
      reorderChartSpecs: handleReorder,
      setActiveDataset: handleSetDataset,
    }),
    [handleAddChart, handleClear, handleRemove, handleReorder, handleSetDataset]
  );

  return {
    handlers,
    datasetManifest: state.datasetManifest,
  };
}
