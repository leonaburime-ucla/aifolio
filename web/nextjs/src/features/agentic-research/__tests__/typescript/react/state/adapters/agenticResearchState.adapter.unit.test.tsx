import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";

const state = {
  datasetManifest: [],
  selectedDatasetId: null,
  sklearnTools: [],
  tableRows: [],
  tableColumns: [],
  numericMatrix: [],
  featureNames: [],
  pcaChartSpec: null,
  isLoading: false,
  error: null,
};
const actions = {
  setDatasetManifest: vi.fn(),
  setSelectedDatasetId: vi.fn(),
  setSklearnTools: vi.fn(),
  setTableRows: vi.fn(),
  setTableColumns: vi.fn(),
  setNumericMatrix: vi.fn(),
  setFeatureNames: vi.fn(),
  setPcaChartSpec: vi.fn(),
  setLoading: vi.fn(),
  setError: vi.fn(),
};

vi.mock("@/features/agentic-research/typescript/react/state/zustand/agenticResearchStore", () => ({
  useAgenticResearchState: () => state,
  useAgenticResearchActions: () => actions,
}));

import { useAgenticResearchStateAdapter } from "@/features/agentic-research/typescript/react/state/adapters/agenticResearchState.adapter";

describe("agenticResearchState.adapter", () => {
  it("returns state/actions from zustand adapters", () => {
    const { result } = renderHook(() => useAgenticResearchStateAdapter());
    expect(result.current.state).toBe(state);
    expect(result.current.actions).toBe(actions);
  });
});
