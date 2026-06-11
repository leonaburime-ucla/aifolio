import { ref, computed } from "vue";
import {
  groupSklearnTools,
  formatToolName,
  toDatasetOptions,
  resolveDefaultDatasetId,
} from "@aifolio/frontend-core/agentic-research";
import { createAgenticResearchApi } from "../api";
import type { ChartSpec } from "~/composables/useChartStore";

export type AgenticResearchApi = ReturnType<typeof createAgenticResearchApi>;

export type UseAgenticResearchOptions = {
  baseUrl: string;
  onDatasetChange?: (id: string) => void;
  api?: AgenticResearchApi;
};

export function useAgenticResearch(options: UseAgenticResearchOptions) {
  const api = options.api ?? createAgenticResearchApi({ baseUrl: options.baseUrl });

  const datasetOptions = ref<{ id: string; label: string }[]>([]);
  const selectedDatasetId = ref<string | null>(null);
  const tableRows = ref<Record<string, unknown>[]>([]);
  const tableColumns = ref<string[]>([]);
  const sklearnTools = ref<string[]>([]);
  const chartSpecs = ref<ChartSpec[]>([]);
  const isLoading = ref(false);
  const error = ref<string | null>(null);

  let initialLoadDone = false;

  const toolGroups = computed(() => {
    const grouped = groupSklearnTools({ tools: sklearnTools.value });
    return ["Decomposition & Embeddings", "Classification", "Clustering", "Regression"]
      .filter((name) => grouped[name]?.length)
      .map((name) => ({
        name,
        formatted: grouped[name].map((t: string) => formatToolName({ name: t })).join(", "),
      }));
  });

  async function loadManifest() {
    try {
      const entries = await api.loadManifest();
      datasetOptions.value = toDatasetOptions({ datasetManifest: entries });
      const resolved = resolveDefaultDatasetId({
        selectedDatasetId: selectedDatasetId.value,
        datasets: entries,
      });
      if (resolved && !selectedDatasetId.value) {
        selectedDatasetId.value = resolved;
      }
    } catch (err) {
      error.value = err instanceof Error ? err.message : "Failed to load manifest.";
    }
  }

  async function loadTools() {
    try {
      sklearnTools.value = await api.loadTools();
    } catch {
      sklearnTools.value = [];
    }
  }

  async function loadDataset(id: string) {
    isLoading.value = true;
    error.value = null;
    tableRows.value = [];
    tableColumns.value = [];
    try {
      const payload = await api.loadDataset(id);
      const rows = payload.rows ?? [];
      tableRows.value = rows;
      tableColumns.value = payload.columns ?? (rows.length > 0 ? Object.keys(rows[0]) : []);
    } catch (err) {
      error.value = err instanceof Error ? err.message : "Failed to load dataset.";
    } finally {
      isLoading.value = false;
    }
  }

  async function init() {
    await Promise.all([loadManifest(), loadTools()]);
    if (selectedDatasetId.value) {
      options.onDatasetChange?.(selectedDatasetId.value);
      initialLoadDone = true;
      await loadDataset(selectedDatasetId.value);
    }
  }

  function onDatasetChange(id: string) {
    selectedDatasetId.value = id;
    options.onDatasetChange?.(id);
  }

  async function onDatasetWatch(id: string | null) {
    if (!id) return;
    if (initialLoadDone) {
      initialLoadDone = false;
      return;
    }
    await loadDataset(id);
  }

  function removeChartSpec(id: string) {
    chartSpecs.value = chartSpecs.value.filter((s) => s.id !== id);
  }

  return {
    datasetOptions,
    selectedDatasetId,
    tableRows,
    tableColumns,
    sklearnTools,
    chartSpecs,
    isLoading,
    error,
    toolGroups,
    init,
    onDatasetChange,
    onDatasetWatch,
    removeChartSpec,
  };
}
