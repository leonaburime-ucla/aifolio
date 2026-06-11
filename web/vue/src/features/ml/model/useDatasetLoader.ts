import { ref, watch } from "vue";

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

export function useDatasetLoader(options: UseDatasetLoaderOptions) {
  const api = options.api ?? createDatasetLoaderApi({ baseUrl: options.baseUrl });

  const datasetOptions = ref<{ id: string; label: string }[]>([]);
  const selectedDatasetId = ref<string | null>(null);
  const tableRows = ref<Record<string, unknown>[]>([]);
  const tableColumns = ref<string[]>([]);
  const targetColumn = ref("");
  const datasetError = ref<string | null>(null);

  async function loadManifest() {
    try {
      const { datasets } = await api.fetchManifest();
      datasetOptions.value = datasets.map((d) => ({ id: d.id, label: d.label ?? d.id }));
      if (datasetOptions.value.length > 0 && !selectedDatasetId.value) {
        selectedDatasetId.value = datasetOptions.value[0].id;
      }
    } catch (err) {
      datasetError.value = err instanceof Error ? err.message : "Error";
    }
  }

  async function loadDataset(id: string) {
    datasetError.value = null;
    try {
      const payload = await api.fetchDataset(id);
      tableRows.value = payload.rows ?? [];
      tableColumns.value = payload.columns ?? (tableRows.value.length > 0 ? Object.keys(tableRows.value[0]) : []);
      if (tableColumns.value.length > 0 && !targetColumn.value) {
        targetColumn.value = tableColumns.value[tableColumns.value.length - 1];
      }
    } catch (err) {
      datasetError.value = err instanceof Error ? err.message : "Error";
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
