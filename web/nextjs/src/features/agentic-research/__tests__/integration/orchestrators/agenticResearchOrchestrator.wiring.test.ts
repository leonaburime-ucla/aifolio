import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useAgenticResearchOrchestrator } from "@/features/agentic-research/orchestrators/agenticResearchOrchestrator";

const mocks = vi.hoisted(() => ({
  useAgenticResearchIntegrationMock: vi.fn(() => ({
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
    groupedTools: {},
    datasetOptions: [],
    reloadManifest: vi.fn(),
    setSelectedDatasetId: vi.fn(),
  })),
}));

vi.mock("@/features/agentic-research/hooks/useAgenticResearch.hooks", () => ({
  useAgenticResearchIntegration: mocks.useAgenticResearchIntegrationMock,
}));

describe("useAgenticResearchOrchestrator", () => {
  it("injects dataset/tools/rows/pca API deps into integration", () => {
    const useStatePort = () => ({
      state: {
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
      },
      actions: {
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
      },
    });

    const useChartStatePort = () => ({ chartSpecs: [] });

    const { result } = renderHook(() =>
      useAgenticResearchOrchestrator({ useStatePort, useChartStatePort })
    );

    const depsArg = mocks.useAgenticResearchIntegrationMock.mock.calls[0][0];
    expect(typeof depsArg.api.fetchDatasetManifest).toBe("function");
    expect(typeof depsArg.api.fetchSklearnTools).toBe("function");
    expect(typeof depsArg.api.fetchDatasetRows).toBe("function");
    expect(typeof depsArg.api.fetchPcaChartSpec).toBe("function");
    expect(result.current.chartSpecs).toEqual([]);
  });
});
