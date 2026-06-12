import { defineStore } from "pinia";
import { ref, shallowRef, computed } from "vue";

export interface DatasetOption {
  id: string;
  label: string;
}

export const useMlDataStore = defineStore("mlData", () => {
  const datasetOptions = ref<DatasetOption[]>([]);
  const selectedDatasetId = ref<string | null>(null);
  const rows = shallowRef<Record<string, unknown>[]>([]);
  const columns = shallowRef<string[]>([]);
  const isLoading = ref(false);
  const error = ref<string | null>(null);

  const rowCount = computed(() => rows.value.length);
  const totalRowCount = ref(0);

  function setDataset(id: string, data: { rows: Record<string, unknown>[]; columns: string[]; totalRows: number }) {
    selectedDatasetId.value = id;
    rows.value = data.rows;
    columns.value = data.columns;
    totalRowCount.value = data.totalRows;
  }

  function setDatasetOptions(options: DatasetOption[]) {
    datasetOptions.value = options;
  }

  function setError(msg: string | null) {
    error.value = msg;
  }

  function setLoading(val: boolean) {
    isLoading.value = val;
  }

  return {
    datasetOptions,
    selectedDatasetId,
    rows,
    columns,
    isLoading,
    error,
    rowCount,
    totalRowCount,
    setDataset,
    setDatasetOptions,
    setError,
    setLoading,
  };
});
