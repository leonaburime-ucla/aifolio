import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import {
  mlDatasetStore,
  useMlDatasetActions,
  useMlDatasetState,
} from "@/features/ml/typescript/react/state/zustand/mlDataStore";

describe("mlDataStore", () => {
  beforeEach(() => {
    mlDatasetStore.setState({
      datasetOptions: [],
      selectedDatasetId: null,
      datasetCache: {},
      manifestLoaded: false,
      isLoadingManifest: false,
      isLoadingDataset: false,
      error: null,
    });
  });

  it("updates dataset state via actions", () => {
    const { result: state } = renderHook(() => useMlDatasetState());
    const { result: actions } = renderHook(() => useMlDatasetActions());

    act(() => {
      actions.current.setDatasetOptions([{ id: "d1", label: "Dataset" }]);
      actions.current.setSelectedDatasetId("d1");
      actions.current.setDatasetCacheEntry("d1", {
        columns: ["a"],
        rows: [{ a: 1 }],
        rowCount: 1,
        totalRowCount: 1,
      });
      actions.current.setManifestLoaded(true);
      actions.current.setLoadingManifest(true);
      actions.current.setLoadingDataset(true);
      actions.current.setError("bad");
    });

    expect(state.current.datasetOptions).toEqual([{ id: "d1", label: "Dataset" }]);
    expect(state.current.selectedDatasetId).toBe("d1");
    expect(state.current.datasetCache.d1?.rowCount).toBe(1);
    expect(state.current.manifestLoaded).toBe(true);
    expect(state.current.isLoadingManifest).toBe(true);
    expect(state.current.isLoadingDataset).toBe(true);
    expect(state.current.error).toBe("bad");
  });
});
