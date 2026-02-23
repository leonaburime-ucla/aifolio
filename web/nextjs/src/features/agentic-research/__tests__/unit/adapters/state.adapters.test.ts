import { beforeEach, describe, expect, it } from "vitest";
import { renderHook } from "@testing-library/react";
import { useAgenticResearchStateAdapter } from "@/features/agentic-research/state/adapters/agenticResearchState.adapter";
import { useAgenticResearchChartStateAdapter } from "@/features/agentic-research/state/adapters/chartState.adapter";
import { useAgenticResearchChartActionsAdapter } from "@/features/agentic-research/state/adapters/chartActions.adapter";
import { agenticResearchStore } from "@/features/agentic-research/state/zustand/agenticResearchStore";
import { useAgenticResearchChartStore } from "@/features/agentic-research/state/zustand/agenticResearchChartStore";

describe("agentic-research state adapters", () => {
  beforeEach(() => {
    agenticResearchStore.setState({
      datasetManifest: [{ id: "d1", label: "D1" }],
      selectedDatasetId: "d1",
      sklearnTools: [],
      tableRows: [],
      tableColumns: [],
      numericMatrix: [],
      featureNames: [],
      pcaChartSpec: null,
      isLoading: false,
      error: null,
    });
    useAgenticResearchChartStore.setState({ chartSpecs: [] });
  });

  it("useAgenticResearchStateAdapter exposes state and actions", () => {
    const { result } = renderHook(() => useAgenticResearchStateAdapter());
    expect(result.current.state.selectedDatasetId).toBe("d1");
    expect(typeof result.current.actions.setSelectedDatasetId).toBe("function");
  });

  it("chart adapters expose chart state and write actions", () => {
    const { result: statePort } = renderHook(() => useAgenticResearchChartStateAdapter());
    const { result: actionsPort } = renderHook(() => useAgenticResearchChartActionsAdapter());

    expect(statePort.current.chartSpecs).toEqual([]);
    actionsPort.current.addChartSpec({
      id: "c1",
      title: "c1",
      type: "line",
      data: [],
      xKey: "x",
      yKeys: ["y"],
    } as any);

    expect(useAgenticResearchChartStore.getState().chartSpecs.map((c) => c.id)).toEqual(["c1"]);
  });
});
