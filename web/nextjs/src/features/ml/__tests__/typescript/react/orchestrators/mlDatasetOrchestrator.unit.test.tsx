import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useMlDatasetOrchestrator } from "@/features/ml/typescript/react/orchestrators/mlDatasetOrchestrator";
import type {
  MlDatasetActions,
  MlDatasetState,
} from "@/features/ml/__types__/typescript/mlData.types";

function createDatasetHarness(initialState?: Partial<MlDatasetState>) {
  const state: MlDatasetState = {
    datasetOptions: [],
    selectedDatasetId: null,
    datasetCache: {},
    manifestLoaded: false,
    isLoadingManifest: false,
    isLoadingDataset: false,
    error: null,
    ...initialState,
  };

  const actions: MlDatasetActions = {
    setDatasetOptions: vi.fn((value) => {
      state.datasetOptions = value;
    }),
    setSelectedDatasetId: vi.fn((value) => {
      state.selectedDatasetId = value;
    }),
    setDatasetCacheEntry: vi.fn((datasetId, value) => {
      state.datasetCache[datasetId] = value;
    }),
    setManifestLoaded: vi.fn((value) => {
      state.manifestLoaded = value;
    }),
    setLoadingManifest: vi.fn((value) => {
      state.isLoadingManifest = value;
    }),
    setLoadingDataset: vi.fn((value) => {
      state.isLoadingDataset = value;
    }),
    setError: vi.fn((value) => {
      state.error = value;
    }),
  };

  const useDatasetState = () => ({ state, actions });

  return { state, actions, useDatasetState };
}

describe("useMlDatasetOrchestrator", () => {
  it("does not auto-load when autoLoad is false", () => {
    const harness = createDatasetHarness();
    const loadDatasetOptions = vi.fn(async () => []);
    const loadDatasetRows = vi.fn(async () => ({ rows: [], columns: [] }));

    renderHook(() =>
      useMlDatasetOrchestrator({
        useDatasetState: harness.useDatasetState,
        loadDatasetOptions,
        loadDatasetRows,
        autoLoad: false,
      })
    );

    expect(loadDatasetOptions).not.toHaveBeenCalled();
    expect(loadDatasetRows).not.toHaveBeenCalled();
  });

  it("reloadManifest loads options and selects default dataset id", async () => {
    const harness = createDatasetHarness();
    const options = [
      { id: "d1.csv", label: "d1.csv", description: "dataset 1" },
      { id: "d2.csv", label: "d2.csv", description: "dataset 2" },
    ];
    const loadDatasetOptions = vi.fn(async () => options);

    const { result } = renderHook(() =>
      useMlDatasetOrchestrator({
        useDatasetState: harness.useDatasetState,
        loadDatasetOptions,
        autoLoad: false,
      })
    );

    await act(async () => {
      await result.current.reloadManifest();
    });

    expect(loadDatasetOptions).toHaveBeenCalledTimes(1);
    expect(harness.actions.setDatasetOptions).toHaveBeenCalledWith(options);
    expect(harness.actions.setSelectedDatasetId).toHaveBeenCalledWith("d1.csv");
    expect(harness.actions.setManifestLoaded).toHaveBeenCalledWith(true);
  });

  it("reloadManifest skips when already loaded or currently loading", async () => {
    const loaded = createDatasetHarness({ manifestLoaded: true });
    const loading = createDatasetHarness({ isLoadingManifest: true });
    const loadDatasetOptions = vi.fn(async () => [{ id: "d1", label: "d1", description: "" }]);

    const { result: loadedResult } = renderHook(() =>
      useMlDatasetOrchestrator({
        useDatasetState: loaded.useDatasetState,
        loadDatasetOptions,
        autoLoad: false,
      })
    );
    const { result: loadingResult } = renderHook(() =>
      useMlDatasetOrchestrator({
        useDatasetState: loading.useDatasetState,
        loadDatasetOptions,
        autoLoad: false,
      })
    );

    await act(async () => {
      await loadedResult.current.reloadManifest();
      await loadingResult.current.reloadManifest();
    });

    expect(loadDatasetOptions).not.toHaveBeenCalled();
  });

  it("reloadDataset skips fetch when selected dataset is already cached", async () => {
    const harness = createDatasetHarness({
      selectedDatasetId: "cached.csv",
      datasetCache: {
        "cached.csv": {
          columns: ["a"],
          rows: [{ a: 1 }],
          rowCount: 1,
          totalRowCount: 1,
        },
      },
    });
    const loadDatasetRows = vi.fn(async () => ({ rows: [], columns: [] }));

    const { result } = renderHook(() =>
      useMlDatasetOrchestrator({
        useDatasetState: harness.useDatasetState,
        loadDatasetRows,
        autoLoad: false,
      })
    );

    await act(async () => {
      await result.current.reloadDataset();
    });

    expect(loadDatasetRows).not.toHaveBeenCalled();
    expect(harness.actions.setDatasetCacheEntry).not.toHaveBeenCalled();
  });

  it("reloadDataset caches fetched rows and derives columns when absent", async () => {
    const harness = createDatasetHarness({
      selectedDatasetId: "fresh.csv",
    });
    const loadDatasetRows = vi.fn(async () => ({
      rows: [{ feature_a: 1, feature_b: "x" }],
      rowCount: 1,
      totalRowCount: 10,
    }));

    const { result } = renderHook(() =>
      useMlDatasetOrchestrator({
        useDatasetState: harness.useDatasetState,
        loadDatasetRows,
        autoLoad: false,
      })
    );

    await act(async () => {
      await result.current.reloadDataset();
    });

    expect(loadDatasetRows).toHaveBeenCalledWith({ datasetId: "fresh.csv" });
    expect(harness.actions.setDatasetCacheEntry).toHaveBeenCalledWith(
      "fresh.csv",
      expect.objectContaining({
        columns: ["feature_a", "feature_b"],
        rowCount: 1,
        totalRowCount: 10,
      })
    );
  });

  it("handles dataset/manifest load errors and missing selected dataset", async () => {
    const harness = createDatasetHarness({ selectedDatasetId: null });
    const loadDatasetOptions = vi.fn(async () => {
      throw new Error("manifest boom");
    });
    const loadDatasetRows = vi.fn(async () => {
      throw new Error("rows boom");
    });

    const { result } = renderHook(() =>
      useMlDatasetOrchestrator({
        useDatasetState: harness.useDatasetState,
        loadDatasetOptions,
        loadDatasetRows,
        autoLoad: false,
      })
    );

    await act(async () => {
      await result.current.reloadManifest();
      await result.current.reloadDataset();
    });

    expect(harness.actions.setError).toHaveBeenCalledWith("manifest boom");
    expect(loadDatasetRows).not.toHaveBeenCalled();
  });

  it("maps non-Error throws to default error copy", async () => {
    const harness = createDatasetHarness({ selectedDatasetId: "fresh.csv" });
    const loadDatasetOptions = vi.fn(async () => {
      throw "bad-manifest";
    });
    const loadDatasetRows = vi.fn(async () => {
      throw "bad-rows";
    });

    const { result } = renderHook(() =>
      useMlDatasetOrchestrator({
        useDatasetState: harness.useDatasetState,
        loadDatasetOptions,
        loadDatasetRows,
        autoLoad: false,
      })
    );

    await act(async () => {
      await result.current.reloadManifest();
      await result.current.reloadDataset();
    });

    expect(harness.actions.setError).toHaveBeenCalledWith("Failed to load ML datasets.");
    expect(harness.actions.setError).toHaveBeenCalledWith("Failed to load dataset rows.");
  });

  it("auto-loads manifest + dataset effects when enabled", async () => {
    const harness = createDatasetHarness({ selectedDatasetId: "auto.csv" });
    const loadDatasetOptions = vi.fn(async () => [
      { id: "auto.csv", label: "auto.csv", description: "auto dataset" },
    ]);
    const loadDatasetRows = vi.fn(async () => ({
      rows: [{ feature_a: 1 }],
      columns: ["feature_a"],
    }));

    renderHook(() =>
      useMlDatasetOrchestrator({
        useDatasetState: harness.useDatasetState,
        loadDatasetOptions,
        loadDatasetRows,
      })
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(loadDatasetOptions).toHaveBeenCalledTimes(1);
    expect(loadDatasetRows).toHaveBeenCalledWith({ datasetId: "auto.csv" });
  });
});
