"use client";

import { useCallback, useEffect, useMemo } from "react";
import { fetchMlDatasetOptions } from "@/core/views/patterns/mlDatasets";
import {
  useMlDatasetActions,
  useMlDatasetState,
} from "@/features/ml/state/zustand/mlDataStore";
import type { MlDatasetCacheEntry } from "@/features/ml/types/mlData.types";

const AI_API_BASE_URL =
  process.env.NEXT_PUBLIC_AI_API_URL || "http://127.0.0.1:8000";

export function useMlDatasetOrchestrator() {
  const state = useMlDatasetState();
  const actions = useMlDatasetActions();

  const loadManifest = useCallback(async () => {
    if (state.manifestLoaded || state.isLoadingManifest) return;
    try {
      actions.setLoadingManifest(true);
      actions.setError(null);
      const options = await fetchMlDatasetOptions();
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
      const response = await fetch(
        `${AI_API_BASE_URL}/ml-data/${encodeURIComponent(datasetId)}`
      );
      if (!response.ok) {
        throw new Error("Failed to load dataset rows.");
      }
      const payload = (await response.json()) as {
        columns?: string[];
        rows?: Array<Record<string, string | number | null>>;
        rowCount?: number;
        totalRowCount?: number;
      };
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
  }, [actions, state.datasetCache, state.selectedDatasetId]);

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
