import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  agenticResearchStore,
  useAgenticResearchActions,
  useAgenticResearchState,
} from "@/features/agentic-research/typescript/react/state/zustand/agenticResearchStore";

describe("agenticResearchStore hooks", () => {
  it("exposes reactive state and action selectors", () => {
    agenticResearchStore.setState({
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
    });

    const { result: stateResult } = renderHook(() => useAgenticResearchState());
    const { result: actionResult } = renderHook(() => useAgenticResearchActions());

    act(() => {
      actionResult.current.setDatasetManifest([{ id: "iris", label: "Iris" }]);
      actionResult.current.setSklearnTools(["pca_transform"]);
      actionResult.current.setTableRows([{ a: 1 }]);
      actionResult.current.setTableColumns(["a"]);
      actionResult.current.setNumericMatrix([[1]]);
      actionResult.current.setFeatureNames(["a"]);
      actionResult.current.setPcaChartSpec({
        id: "chart",
        title: "Chart",
        type: "scatter",
        xKey: "x",
        yKeys: ["y"],
        data: [],
      });
      actionResult.current.setSelectedDatasetId("iris");
      actionResult.current.setLoading(true);
      actionResult.current.setError("boom");
    });

    expect(stateResult.current.datasetManifest).toEqual([{ id: "iris", label: "Iris" }]);
    expect(stateResult.current.sklearnTools).toEqual(["pca_transform"]);
    expect(stateResult.current.tableRows).toEqual([{ a: 1 }]);
    expect(stateResult.current.tableColumns).toEqual(["a"]);
    expect(stateResult.current.numericMatrix).toEqual([[1]]);
    expect(stateResult.current.featureNames).toEqual(["a"]);
    expect(stateResult.current.pcaChartSpec?.id).toBe("chart");
    expect(stateResult.current.selectedDatasetId).toBe("iris");
    expect(stateResult.current.isLoading).toBe(true);
    expect(stateResult.current.error).toBe("boom");
  });
});
