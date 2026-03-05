import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import type {
  AgenticResearchChartActionsPort,
  AgenticResearchIntegration,
  AgenticResearchStatePort,
} from "@/features/agentic-research/__types__/typescript/agenticResearch.types";

const { useIntegrationMock } = vi.hoisted(() => ({
  useIntegrationMock: vi.fn(
    (deps) =>
      ({
        ...deps.state,
        groupedTools: {},
        datasetOptions: [],
        reloadManifest: vi.fn(),
        setSelectedDatasetId: deps.actions.setSelectedDatasetId,
      }) as AgenticResearchIntegration
  ),
}));

const statePortMock: AgenticResearchStatePort = {
  state: {
    datasetManifest: [{ id: "iris", label: "Iris" }],
    selectedDatasetId: "iris",
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
};

const chartActionsPortMock: AgenticResearchChartActionsPort = {
  chartSpecs: [
    {
      id: "chart-a",
      title: "A",
      type: "scatter",
      xKey: "x",
      yKeys: ["y"],
      data: [],
    },
  ],
  addChartSpec: vi.fn(),
  clearChartSpecs: vi.fn(),
  removeChartSpec: vi.fn(),
  reorderChartSpecs: vi.fn(),
  getChartStateSnapshot: vi.fn(() => ({ chartSpecs: [] })),
};

vi.mock("@/features/agentic-research/typescript/react/hooks/useAgenticResearch.hooks", () => ({
  useAgenticResearchIntegration: useIntegrationMock,
}));

vi.mock("@/features/agentic-research/typescript/react/state/adapters/agenticResearchState.adapter", () => ({
  useAgenticResearchStateAdapter: () => statePortMock,
}));

vi.mock("@/features/agentic-research/typescript/react/state/adapters/chartActions.adapter", () => ({
  useAgenticResearchChartActionsAdapter: () => chartActionsPortMock,
}));

vi.mock("@/features/agentic-research/typescript/api/agenticResearchApi.adapter", () => ({
  createAgenticResearchApiAdapter: () => ({
    fetchDatasetManifest: vi.fn(async () => []),
    fetchSklearnTools: vi.fn(async () => []),
    fetchDatasetRows: vi.fn(async () => ({ rows: [], columns: [] })),
    fetchPcaChartSpec: vi.fn(async () => null),
  }),
}));

import { useAgenticResearchOrchestrator } from "@/features/agentic-research/typescript/react/orchestrators/agenticResearchOrchestrator";

describe("agenticResearchOrchestrator", () => {
  it("wires state, chart state, integration, and formatting", () => {
    const { result } = renderHook(() => useAgenticResearchOrchestrator());

    expect(useIntegrationMock).toHaveBeenCalledTimes(1);
    expect(result.current.chartSpecs.map((item) => item.id)).toEqual(["chart-a"]);
    expect(result.current.activeChartSpec?.id).toBe("chart-a");
    expect(result.current.formatToolName("pca_transform")).toBe("PCA Transform");

    result.current.removeChartSpec("chart-a");
    expect(chartActionsPortMock.removeChartSpec).toHaveBeenCalledWith("chart-a");
  });

  it("uses injected state/chart ports when provided", () => {
    const customStatePort = () => statePortMock;
    const customChartPort = () => chartActionsPortMock;

    renderHook(() =>
      useAgenticResearchOrchestrator({
        useStatePort: customStatePort,
        useChartPort: customChartPort,
      })
    );

    expect(useIntegrationMock).toHaveBeenCalled();
  });
});
