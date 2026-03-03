import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";

const mockState = {
  datasetOptions: [],
  selectedDatasetId: null,
  datasetCache: {},
  manifestLoaded: false,
  isLoadingManifest: false,
  isLoadingDataset: false,
  error: null,
};
const mockActions = {
  setDatasetOptions: vi.fn(),
  setSelectedDatasetId: vi.fn(),
  setDatasetCacheEntry: vi.fn(),
  setManifestLoaded: vi.fn(),
  setLoadingManifest: vi.fn(),
  setLoadingDataset: vi.fn(),
  setError: vi.fn(),
};

vi.mock("@/features/ml/typescript/react/state/zustand/mlDataStore", () => ({
  useMlDatasetState: () => mockState,
  useMlDatasetActions: () => mockActions,
}));

import { useMlDatasetStateAdapter } from "@/features/ml/typescript/react/state/adapters/mlDataset.adapter";

describe("mlDataset.adapter", () => {
  it("exposes state and actions from zustand hooks", () => {
    const { result } = renderHook(() => useMlDatasetStateAdapter());
    expect(result.current.state).toBe(mockState);
    expect(result.current.actions).toBe(mockActions);
  });
});
