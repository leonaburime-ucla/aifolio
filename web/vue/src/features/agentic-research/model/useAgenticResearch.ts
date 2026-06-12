import { ref, shallowRef, computed } from "vue";
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

type LoadedDataset = {
  rows: Record<string, unknown>[];
  columns: string[];
};

const maxCachedDatasets = 3;

export function useAgenticResearch(options: UseAgenticResearchOptions) {
  const api = options.api ?? createAgenticResearchApi({ baseUrl: options.baseUrl });

  const datasetOptions = ref<{ id: string; label: string }[]>([]);
  const selectedDatasetId = ref<string | null>(null);
  const tableRows = shallowRef<Record<string, unknown>[]>([]);
  const tableColumns = shallowRef<string[]>([]);
  const sklearnTools = ref<string[]>([]);
  const chartSpecs = ref<ChartSpec[]>([]);
  const isLoading = ref(false);
  const error = ref<string | null>(null);

  let initialLoadDone = false;
  let datasetLoadRequestId = 0;
  const datasetCache = new Map<string, LoadedDataset>();

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
    const requestId = ++datasetLoadRequestId;
    const selectedIdAtStart = selectedDatasetId.value;
    isLoading.value = true;
    error.value = null;

    const cached = datasetCache.get(id);
    if (cached) {
      datasetCache.delete(id);
      datasetCache.set(id, cached);
      applyDataset(cached);
      isLoading.value = false;
      return;
    }

    tableRows.value = [];
    tableColumns.value = [];
    try {
      const payload = await api.loadDataset(id);
      if (requestId !== datasetLoadRequestId || (selectedIdAtStart !== null && id !== selectedDatasetId.value)) return;
      const rows = payload.rows ?? [];
      const loaded = {
        rows,
        columns: payload.columns ?? (rows.length > 0 ? Object.keys(rows[0]) : []),
      };
      cacheDataset(id, loaded);
      applyDataset(loaded);
    } catch (err) {
      if (requestId !== datasetLoadRequestId || (selectedIdAtStart !== null && id !== selectedDatasetId.value)) return;
      error.value = err instanceof Error ? err.message : "Failed to load dataset.";
    } finally {
      if (requestId === datasetLoadRequestId) isLoading.value = false;
    }
  }

  function cacheDataset(id: string, dataset: LoadedDataset) {
    datasetCache.delete(id);
    datasetCache.set(id, dataset);
    while (datasetCache.size > maxCachedDatasets) {
      const oldest = datasetCache.keys().next().value;
      if (!oldest) break;
      datasetCache.delete(oldest);
    }
  }

  function applyDataset(dataset: LoadedDataset) {
    tableRows.value = dataset.rows;
    tableColumns.value = dataset.columns;
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
