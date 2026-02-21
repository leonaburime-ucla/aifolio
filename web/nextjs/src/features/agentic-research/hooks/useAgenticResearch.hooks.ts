import { useCallback, useEffect, useMemo } from "react";
import type {
  AgenticResearchActions,
  AgenticResearchDeps,
  AgenticResearchIntegration,
  DatasetOption,
  DatasetManifestEntry,
} from "@/features/agentic-research/types/agenticResearch.types";
import { getColumnsFromRows, normalizeRowKeys } from "@/features/agentic-research/utils/datatable.util";

const AI_API_BASE_URL =
  process.env.NEXT_PUBLIC_AI_API_URL || "http://127.0.0.1:8000";
const MANIFEST_PATH = `${AI_API_BASE_URL}/sample-data`;
const SKLEARN_TOOLS_PATH = `${AI_API_BASE_URL}/sklearn-tools`;

function groupSklearnTools(tools: string[]): Record<string, string[]> {
  return tools.reduce<Record<string, string[]>>((acc, tool) => {
    let group = "Other";
    if (tool.includes("regression")) group = "Regression";
    else if (tool.includes("classification")) group = "Classification";
    else if (tool.includes("clustering")) group = "Clustering";
    else if (
      tool.includes("pca") ||
      tool.includes("svd") ||
      tool.includes("ica") ||
      tool.includes("nmf") ||
      tool.includes("tsne")
    )
      group = "Decomposition & Embeddings";
    else if (
      tool.includes("scaler") ||
      tool.includes("encoder") ||
      tool.includes("transformer") ||
      tool.includes("imputer")
    )
      group = "Preprocessing";
    else if (
      tool.includes("select_") ||
      tool.includes("rfe") ||
      tool.includes("rfecv")
    )
      group = "Feature Selection";
    else if (tool.includes("train_test_split")) group = "Model Selection";
    else if (
      tool.includes("accuracy") ||
      tool.includes("precision") ||
      tool.includes("recall") ||
      tool.includes("f1") ||
      tool.includes("roc") ||
      tool.includes("auc")
    )
      group = "Metrics";

    if (!acc[group]) acc[group] = [];
    acc[group].push(tool);
    return acc;
  }, {});
}

/**
 * Hook 1: Local UI state for agentic research screens.
 * @returns Local UI state (placeholder for future UI-only state).
 */
export function useAgenticResearchUiState() {
  return {};
}

/**
 * Hook 2: Business logic for loading datasets, parsing files, and PCA requests.
 * @param deps - Injected state, actions, and API dependencies.
 * @returns Commands to reload the manifest and change the selected dataset.
 */
export function useAgenticResearchLogic(
  deps: AgenticResearchDeps
): {
  reloadManifest: () => void;
  setSelectedDatasetId: AgenticResearchActions["setSelectedDatasetId"];
} {
  const { state, actions } = deps;
  const {
    setLoading,
    setError,
    setDatasetManifest,
    setSelectedDatasetId,
    setSklearnTools,
    setTableRows,
    setTableColumns,
    setPcaChartSpec,
    setNumericMatrix,
    setFeatureNames,
  } = actions;

  /**
   * Fetch the dataset manifest and set a default selection.
   * @returns Promise that resolves when the manifest finishes loading.
   */
  const loadManifest = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(MANIFEST_PATH);
      if (!response.ok) {
        throw new Error("Failed to load dataset manifest.");
      }
      const payload = (await response.json()) as {
        datasets?: DatasetManifestEntry[];
      };
      const datasets = payload.datasets ?? [];
      setDatasetManifest(datasets);
      setSelectedDatasetId(
        state.selectedDatasetId ?? datasets[0]?.id ?? null
      );
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Failed to load manifest."
      );
    } finally {
      setLoading(false);
    }
  }, [setDatasetManifest, setError, setLoading, setSelectedDatasetId, state.selectedDatasetId]);

  /**
   * Fetch the list of available sklearn tools for display.
   * @returns Promise that resolves when tools are loaded.
   */
  const loadSklearnTools = useCallback(async () => {
    try {
      const response = await fetch(SKLEARN_TOOLS_PATH);
      if (!response.ok) {
        throw new Error("Failed to load sklearn tools.");
      }
      const payload = (await response.json()) as { tools?: string[] };
      setSklearnTools(payload.tools ?? []);
    } catch (error) {
      setSklearnTools([]);
      setError(
        error instanceof Error ? error.message : "Failed to load sklearn tools."
      );
    }
  }, [setError, setSklearnTools]);

  /**
   * Load the selected dataset, parse rows, and update table state.
   * @returns Promise that resolves when the dataset finishes loading.
   */
  const loadDataset = useCallback(async () => {
    const selected = state.datasetManifest.find(
      (entry) => entry.id === state.selectedDatasetId
    );
    if (!selected) return;
    try {
      setLoading(true);
      setError(null);
      setTableRows([]);
      setTableColumns([]);
      setPcaChartSpec(null);
      const response = await fetch(
        `${AI_API_BASE_URL}/sample-data/${selected.id}`
      );
      if (!response.ok) {
        throw new Error("Failed to load dataset file.");
      }
      const payload = (await response.json()) as {
        rows?: Array<Record<string, string | number | null>>;
        columns?: string[];
      };
      let rows: Array<Record<string, string | number | null>> = payload.rows ?? [];
      const columns = payload.columns ?? getColumnsFromRows(rows);
      rows = normalizeRowKeys(rows);
      setTableColumns(columns);
      console.log("[agentic-research] columns", {
        datasetId: selected.id,
        columns,
      });
      setTableRows(rows);

      setNumericMatrix([]);
      setFeatureNames([]);
      setPcaChartSpec(null);
    } catch (error) {
      setPcaChartSpec(null);
      setTableRows([]);
      setTableColumns([]);
      setNumericMatrix([]);
      setFeatureNames([]);
      setError(
        error instanceof Error ? error.message : "Failed to load dataset."
      );
    } finally {
      setLoading(false);
    }
  }, [
    setError,
    setFeatureNames,
    setLoading,
    setNumericMatrix,
    setPcaChartSpec,
    setTableColumns,
    setTableRows,
    state.datasetManifest,
    state.selectedDatasetId,
  ]);

  /**
   * Load manifest on first mount.
   */
  useEffect(() => {
    loadManifest();
    loadSklearnTools();
  }, [loadManifest, loadSklearnTools]);

  /**
   * Reload dataset whenever selection changes or manifest arrives.
   */
  useEffect(() => {
    if (!state.selectedDatasetId) return;
    if (state.datasetManifest.length === 0) return;
    loadDataset();
  }, [loadDataset, state.datasetManifest.length, state.selectedDatasetId]);

  return {
    reloadManifest: loadManifest,
    setSelectedDatasetId,
  };
}

/**
 * Hook 3: Integration layer that composes state, UI, and logic.
 * @param deps - Injected state, actions, and API dependencies.
 * @returns Combined state, actions, and derived dataset options for UI.
 */
export function useAgenticResearchIntegration(
  deps: AgenticResearchDeps
): AgenticResearchIntegration {
  const uiState = useAgenticResearchUiState();
  const actions = useAgenticResearchLogic(deps);

  const datasetOptions = useMemo<DatasetOption[]>(
    () =>
      deps.state.datasetManifest.map((entry) => ({
        id: entry.id,
        label: entry.label,
        description: entry.description,
      })),
    [deps.state.datasetManifest]
  );
  const groupedTools = useMemo(
    () => groupSklearnTools(deps.state.sklearnTools),
    [deps.state.sklearnTools]
  );

  return useMemo(
    () => ({
      ...deps.state,
      ...uiState,
      ...actions,
      groupedTools,
      datasetOptions,
    }),
    [actions, datasetOptions, deps.state, groupedTools, uiState]
  );
}
