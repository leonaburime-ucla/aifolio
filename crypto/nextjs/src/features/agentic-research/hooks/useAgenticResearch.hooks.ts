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
  const { state, actions, api } = deps;

  /**
   * Fetch the dataset manifest and set a default selection.
   * @returns Promise that resolves when the manifest finishes loading.
   */
  const loadManifest = useCallback(async () => {
    try {
      actions.setLoading(true);
      actions.setError(null);
      const response = await fetch(MANIFEST_PATH);
      if (!response.ok) {
        throw new Error("Failed to load dataset manifest.");
      }
      const payload = (await response.json()) as {
        datasets?: DatasetManifestEntry[];
      };
      const datasets = payload.datasets ?? [];
      actions.setDatasetManifest(datasets);
      actions.setSelectedDatasetId(
        state.selectedDatasetId ?? datasets[0]?.id ?? null
      );
    } catch (error) {
      actions.setError(
        error instanceof Error ? error.message : "Failed to load manifest."
      );
    } finally {
      actions.setLoading(false);
    }
  }, [actions]);

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
      actions.setSklearnTools(payload.tools ?? []);
    } catch (error) {
      actions.setSklearnTools([]);
      actions.setError(
        error instanceof Error ? error.message : "Failed to load sklearn tools."
      );
    }
  }, [actions]);

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
      actions.setLoading(true);
      actions.setError(null);
      actions.setTableRows([]);
      actions.setTableColumns([]);
      actions.setPcaChartSpec(null);
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
      actions.setTableColumns(columns);
      console.log("[agentic-research] columns", {
        datasetId: selected.id,
        columns,
      });
      actions.setTableRows(rows);

      actions.setNumericMatrix([]);
      actions.setFeatureNames([]);
      actions.setPcaChartSpec(null);
    } catch (error) {
      actions.setPcaChartSpec(null);
      actions.setTableRows([]);
      actions.setTableColumns([]);
      actions.setNumericMatrix([]);
      actions.setFeatureNames([]);
      actions.setError(
        error instanceof Error ? error.message : "Failed to load dataset."
      );
    } finally {
      actions.setLoading(false);
    }
  }, [actions, api, state.datasetManifest, state.selectedDatasetId]);

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
    setSelectedDatasetId: actions.setSelectedDatasetId,
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

  return useMemo(
    () => ({
      ...deps.state,
      ...uiState,
      ...actions,
      datasetOptions,
    }),
    [actions, datasetOptions, deps.state, uiState]
  );
}
