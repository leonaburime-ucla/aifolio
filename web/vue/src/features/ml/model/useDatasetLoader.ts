import { ref, shallowRef, watch } from "vue";
import { resolveDefaultTrainingDatasetId } from "@aifolio/frontend-core/ml-training";

export type DatasetLoaderApi = {
  fetchManifest: () => Promise<{ datasets: { id: string; label?: string }[] }>;
  fetchDataset: (id: string) => Promise<{ rows: Record<string, unknown>[]; columns?: string[] }>;
};

export function createDatasetLoaderApi({ baseUrl }: { baseUrl: string }): DatasetLoaderApi {
  return {
    async fetchManifest() {
      const res = await fetch(`${baseUrl}/ml-data`);
      if (!res.ok) throw new Error("Failed to load datasets.");
      const payload = (await res.json()) as { datasets?: { id: string; label?: string }[] };
      return { datasets: payload.datasets ?? [] };
    },
    async fetchDataset(id: string) {
      const res = await fetch(`${baseUrl}/ml-data/${encodeURIComponent(id)}`);
      if (!res.ok) throw new Error("Failed to load dataset.");
      return (await res.json()) as { rows: Record<string, unknown>[]; columns?: string[] };
    },
  };
}

export type UseDatasetLoaderOptions = {
  baseUrl: string;
  api?: DatasetLoaderApi;
};

type LoadedDataset = {
  rows: Record<string, unknown>[];
  columns: string[];
};

const maxCachedDatasets = 3;

export function useDatasetLoader(options: UseDatasetLoaderOptions) {
  const api = options.api ?? createDatasetLoaderApi({ baseUrl: options.baseUrl });

  const datasetOptions = ref<{ id: string; label: string }[]>([]);
  const selectedDatasetId = ref<string | null>(null);
  const tableRows = shallowRef<Record<string, unknown>[]>([]);
  const tableColumns = shallowRef<string[]>([]);
  const targetColumn = ref("");
  const datasetError = ref<string | null>(null);
  let datasetLoadRequestId = 0;
  const datasetCache = new Map<string, LoadedDataset>();

  async function loadManifest() {
    try {
      const { datasets } = await api.fetchManifest();
      datasetOptions.value = datasets.map((d) => ({ id: d.id, label: d.label ?? d.id }));
      const resolved = resolveDefaultTrainingDatasetId({
        selectedDatasetId: selectedDatasetId.value,
        datasets: datasetOptions.value,
      });
      if (resolved && !selectedDatasetId.value) {
        selectedDatasetId.value = resolved;
      }
    } catch (err) {
      datasetError.value = err instanceof Error ? err.message : "Error";
    }
  }

  async function loadDataset(id: string) {
    const requestId = ++datasetLoadRequestId;
    const selectedIdAtStart = selectedDatasetId.value;
    datasetError.value = null;

    const cached = datasetCache.get(id);
    if (cached) {
      datasetCache.delete(id);
      datasetCache.set(id, cached);
      applyDataset(cached);
      return;
    }

    tableRows.value = [];
    tableColumns.value = [];
    try {
      const payload = await api.fetchDataset(id);
      if (requestId !== datasetLoadRequestId || (selectedIdAtStart !== null && id !== selectedDatasetId.value)) return;
      const rows = payload.rows ?? [];
      const columns = payload.columns ?? (rows.length > 0 ? Object.keys(rows[0]) : []);
      const loaded = { rows, columns };
      cacheDataset(id, loaded);
      applyDataset(loaded);
    } catch (err) {
      if (requestId !== datasetLoadRequestId || (selectedIdAtStart !== null && id !== selectedDatasetId.value)) return;
      datasetError.value = err instanceof Error ? err.message : "Error";
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
    if (dataset.columns.length > 0 && !targetColumn.value) {
      targetColumn.value = dataset.columns[dataset.columns.length - 1];
    }
  }

  function onDatasetChange(id: string) {
    selectedDatasetId.value = id;
  }

  watch(selectedDatasetId, (id) => { if (id) loadDataset(id); });

  return {
    datasetOptions,
    selectedDatasetId,
    tableRows,
    tableColumns,
    targetColumn,
    datasetError,
    loadManifest,
    loadDataset,
    onDatasetChange,
  };
}
