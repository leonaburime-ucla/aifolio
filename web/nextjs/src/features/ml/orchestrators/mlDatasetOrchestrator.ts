"use client";

import { useCallback, useEffect, useMemo } from "react";
import { fetchMlDatasetOptions, fetchMlDatasetRows } from "@/features/ml/api/mlDataApi";
import { useMlDatasetStateAdapter } from "@/features/ml/state/adapters/mlDataset.adapter";
import type { MlDatasetCacheEntry } from "@/features/ml/types/mlData.types";

type MlDatasetOrchestratorDeps = {
  useDatasetState?: typeof useMlDatasetStateAdapter;
  loadDatasetOptions?: typeof fetchMlDatasetOptions;
  loadDatasetRows?: typeof fetchMlDatasetRows;
};

/**
 * Orchestrates ML dataset manifest + row loading through injected state/API dependencies.
 */
export function useMlDatasetOrchestrator({
  useDatasetState = useMlDatasetStateAdapter,
  loadDatasetOptions = fetchMlDatasetOptions,
  loadDatasetRows = fetchMlDatasetRows,
}: MlDatasetOrchestratorDeps = {}) {
  const { state, actions } = useDatasetState();

  const loadManifest = useCallback(async () => {
    if (state.manifestLoaded || state.isLoadingManifest) return;
    try {
      actions.setLoadingManifest(true);
      actions.setError(null);
      const options = await loadDatasetOptions();
      actions.setDatasetOptions(options);
      actions.setSelectedDatasetId(state.selectedDatasetId ?? options[0]?.id ?? null);
      actions.setManifestLoaded(true);
    } catch (error) {
      actions.setError(
        error instanceof Error ? error.message : "Failed to load ML datasets."
      );
    } finally {
      actions.setLoadingManifest(false);
    }
  }, [
    actions,
    loadDatasetOptions,
    state.isLoadingManifest,
    state.manifestLoaded,
    state.selectedDatasetId,
  ]);

  const loadDataset = useCallback(async () => {
    const datasetId = state.selectedDatasetId;
    if (!datasetId) return;
    if (state.datasetCache[datasetId]) return;

    try {
      actions.setLoadingDataset(true);
      actions.setError(null);
      const payload = await loadDatasetRows(datasetId);
      const rows = payload.rows ?? [];
      const columns = payload.columns ?? Object.keys(rows[0] ?? {});
      const cacheEntry: MlDatasetCacheEntry = {
        columns,
        rows,
        rowCount: payload.rowCount ?? rows.length,
        totalRowCount: payload.totalRowCount ?? rows.length,
      };
      actions.setDatasetCacheEntry(datasetId, cacheEntry);
    } catch (error) {
      actions.setError(
        error instanceof Error ? error.message : "Failed to load dataset rows."
      );
    } finally {
      actions.setLoadingDataset(false);
    }
  }, [actions, loadDatasetRows, state.datasetCache, state.selectedDatasetId]);

  useEffect(() => {
    loadManifest();
  }, [loadManifest]);

  useEffect(() => {
    loadDataset();
  }, [loadDataset]);

  const activeDataset = useMemo(() => {
    if (!state.selectedDatasetId) return null;
    return state.datasetCache[state.selectedDatasetId] ?? null;
  }, [state.datasetCache, state.selectedDatasetId]);

  const isLoading = state.isLoadingManifest || state.isLoadingDataset;

  return {
    datasetOptions: state.datasetOptions,
    selectedDatasetId: state.selectedDatasetId,
    setSelectedDatasetId: actions.setSelectedDatasetId,
    isLoading,
    error: state.error,
    tableRows: activeDataset?.rows ?? [],
    tableColumns: activeDataset?.columns ?? [],
    rowCount: activeDataset?.rowCount ?? 0,
    totalRowCount: activeDataset?.totalRowCount ?? 0,
  };
}
