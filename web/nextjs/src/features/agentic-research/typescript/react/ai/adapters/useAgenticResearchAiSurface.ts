import { useCallback, useMemo } from "react";
import { useAgenticResearchChartActionsAdapter } from "@/features/agentic-research/typescript/react/state/adapters/chartActions.adapter";
import { useAgenticResearchStateAdapter } from "@/features/agentic-research/typescript/react/state/adapters/agenticResearchState.adapter";
import {
  handleAgenticAddChartSpec,
  handleAgenticClearCharts,
  handleAgenticRemoveChartSpec,
  handleAgenticReorderChartSpecs,
} from "@/features/agentic-research/typescript/ai/tools/chartTools";
import { handleAgenticSetActiveDataset } from "@/features/agentic-research/typescript/ai/tools/datasetTools";
import type { AgenticResearchAiSurface } from "@/features/agentic-research/__types__/typescript/react/ai/agenticResearchAiSurface.types";

export function useAgenticResearchAiSurface(): AgenticResearchAiSurface {
  const chartActions = useAgenticResearchChartActionsAdapter();
  const { state, actions } = useAgenticResearchStateAdapter();

  const handleAddChart = useCallback(
    (payload: { chartSpec?: unknown; chartSpecs?: unknown[] }) => {
      return handleAgenticAddChartSpec(payload, chartActions.addChartSpec);
    },
    [chartActions.addChartSpec]
  );

  const handleClear = useCallback(() => {
    return handleAgenticClearCharts(chartActions.clearChartSpecs);
  }, [chartActions.clearChartSpecs]);

  const handleRemove = useCallback(
    (chartId: string) => {
      return handleAgenticRemoveChartSpec(
        chartId,
        chartActions.getChartStateSnapshot,
        chartActions.removeChartSpec
      );
    },
    [chartActions.getChartStateSnapshot, chartActions.removeChartSpec]
  );

  const handleReorder = useCallback(
    (args: { ordered_ids?: string[]; from_index?: number; to_index?: number }) => {
      return handleAgenticReorderChartSpecs(
        args,
        chartActions.getChartStateSnapshot,
        chartActions.reorderChartSpecs
      );
    },
    [chartActions.getChartStateSnapshot, chartActions.reorderChartSpecs]
  );

  const handleSetDataset = useCallback(
    (datasetId: string) => {
      return handleAgenticSetActiveDataset(
        datasetId,
        state.datasetManifest,
        chartActions.clearChartSpecs,
        actions.setSelectedDatasetId
      );
    },
    [state.datasetManifest, chartActions.clearChartSpecs, actions.setSelectedDatasetId]
  );

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
