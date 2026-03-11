import { useCallback, useEffect, useMemo } from "react";
import type {
  AgenticResearchActions,
  AgenticResearchDeps,
  AgenticResearchIntegration,
  DatasetOption,
} from "@/features/agentic-research/__types__/typescript/agenticResearch.types";
import { getColumnsFromRows, normalizeRowKeys } from "@/features/agentic-research/typescript/utils/datatable.util";
import {
  resolveDefaultDatasetId,
  toDatasetOptions,
} from "@/features/agentic-research/typescript/logic/agenticResearchManifest.logic";
import { groupSklearnTools } from "@/features/agentic-research/typescript/logic/agenticResearchTools.logic";
import { applyDatasetLoadReset } from "@/features/agentic-research/typescript/logic/agenticResearchDataset.logic";

const DEBUG_EFFECTS = process.env.NEXT_PUBLIC_DEBUG_EFFECTS === "1";

function getDebugPath(): string {
  return globalThis.location?.pathname ?? "";
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
      const datasets = await deps.api.fetchDatasetManifest();
      setDatasetManifest(datasets);
      setSelectedDatasetId(
        resolveDefaultDatasetId({
          selectedDatasetId: state.selectedDatasetId,
          datasets,
        })
      );
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Failed to load manifest."
      );
    } finally {
      setLoading(false);
    }
  }, [
    deps.api,
    setDatasetManifest,
    setError,
    setLoading,
    setSelectedDatasetId,
    state.selectedDatasetId,
  ]);

  /**
   * Fetch the list of available sklearn tools for display.
   * @returns Promise that resolves when tools are loaded.
   */
  const loadSklearnTools = useCallback(async () => {
    try {
      const tools = await deps.api.fetchSklearnTools();
      setSklearnTools(tools);
    } catch (error) {
      setSklearnTools([]);
      setError(
        error instanceof Error ? error.message : "Failed to load sklearn tools."
      );
    }
  }, [deps.api, setError, setSklearnTools]);

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
      applyDatasetLoadReset({
        actions: {
          setTableRows,
          setTableColumns,
          setNumericMatrix,
          setFeatureNames,
          setPcaChartSpec,
        },
      });
      const payload = await deps.api.fetchDatasetRows(selected.id);
      let rows: Array<Record<string, string | number | null>> = payload.rows ?? [];
      const columns = payload.columns ?? getColumnsFromRows(rows);
      rows = normalizeRowKeys(rows);
      setTableColumns(columns);
      setTableRows(rows);
    } catch (error) {
      applyDatasetLoadReset({
        actions: {
          setTableRows,
          setTableColumns,
          setNumericMatrix,
          setFeatureNames,
          setPcaChartSpec,
        },
      });
      setError(
        error instanceof Error ? error.message : "Failed to load dataset."
      );
    } finally {
      setLoading(false);
    }
  }, [
    deps.api,
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
    if (DEBUG_EFFECTS) {
      console.log("[agentic-debug] load_manifest_effect", {
        path: getDebugPath(),
        selectedDatasetId: state.selectedDatasetId,
      });
    }
    loadManifest();
    loadSklearnTools();
  }, [loadManifest, loadSklearnTools]);

  /**
   * Reload dataset whenever selection changes or manifest arrives.
   */
  useEffect(() => {
    if (DEBUG_EFFECTS) {
      console.log("[agentic-debug] load_dataset_effect", {
        path: getDebugPath(),
        selectedDatasetId: state.selectedDatasetId,
        manifestCount: state.datasetManifest.length,
      });
    }
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
      toDatasetOptions({
        datasetManifest: deps.state.datasetManifest,
      }),
    [deps.state.datasetManifest]
  );
  const groupedTools = useMemo(
    () => groupSklearnTools({ tools: deps.state.sklearnTools }),
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
