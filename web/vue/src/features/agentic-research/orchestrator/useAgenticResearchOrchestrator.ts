import { onMounted, watch } from "vue";
import { storeToRefs } from "pinia";
import { useChartStore } from "~/composables/useChartStore";
import { useAgenticResearch } from "../model";
import type { UseAgenticResearchOptions } from "../model";

export type UseAgenticResearchOrchestratorOptions = UseAgenticResearchOptions;

export const samplePrompts = ["Run PCA analysis", "Run NMF Decomposition and PLSR"];


export function useAgenticResearchOrchestrator(
  options: UseAgenticResearchOrchestratorOptions
) {
  const model = useAgenticResearch(options);
  const chartStore = useChartStore();
  const { chartSpecs } = storeToRefs(chartStore);

  onMounted(() => model.init());
  watch(model.selectedDatasetId, (id) => model.onDatasetWatch(id));

  return {
    datasetOptions: model.datasetOptions,
    selectedDatasetId: model.selectedDatasetId,
    tableRows: model.tableRows,
    tableColumns: model.tableColumns,
    sklearnTools: model.sklearnTools,
    chartSpecs,
    samplePrompts,
    isLoading: model.isLoading,
    error: model.error,
    toolGroups: model.toolGroups,
    onDatasetChange: model.onDatasetChange,
    removeChartSpec: chartStore.removeChartSpec,
  };
}
