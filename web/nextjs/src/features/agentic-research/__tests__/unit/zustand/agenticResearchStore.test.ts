import { beforeEach, describe, expect, it } from "vitest";
import {
  agenticResearchStore,
  getActiveDatasetPayload,
  getAgenticResearchSnapshot,
} from "@/features/agentic-research/state/zustand/agenticResearchStore";

describe("agenticResearchStore", () => {
  beforeEach(() => {
    agenticResearchStore.setState({
      datasetManifest: [],
      selectedDatasetId: "dataset-a",
      sklearnTools: [],
      tableRows: [{ a: 1 }, { a: 2 }, { a: 3 }],
      tableColumns: ["a"],
      numericMatrix: [[1], [2], [3]],
      featureNames: ["a"],
      pcaChartSpec: null,
      isLoading: false,
      error: null,
    });
  });

  it("getAgenticResearchSnapshot returns current state projection", () => {
    const snapshot = getAgenticResearchSnapshot();
    expect(snapshot.selectedDatasetId).toBe("dataset-a");
    expect(snapshot.tableRows).toHaveLength(3);
  });

  it("getActiveDatasetPayload truncates rows and matrix by maxRows", () => {
    const payload = getActiveDatasetPayload(2);
    expect(payload.datasetId).toBe("dataset-a");
    expect(payload.rows).toHaveLength(2);
    expect(payload.numericMatrix).toHaveLength(2);
    expect(payload.columns).toEqual(["a"]);
  });

  it("exposes and applies all imperative store setters", () => {
    const store = agenticResearchStore.getState();
    store.setDatasetManifest([{ id: "d1", label: "D1" }]);
    store.setSelectedDatasetId("d1");
    store.setSklearnTools(["pca_transform"]);
    store.setTableRows([{ x: 1 }]);
    store.setTableColumns(["x"]);
    store.setNumericMatrix([[1]]);
    store.setFeatureNames(["x"]);
    store.setPcaChartSpec({
      id: "chart-1",
      title: "Chart 1",
      type: "line",
      data: [{ x: 1, y: 2 }],
      xKey: "x",
      yKeys: ["y"],
    } as any);
    store.setLoading(true);
    store.setError("err");

    const snapshot = getAgenticResearchSnapshot();
    expect(snapshot.datasetManifest).toEqual([{ id: "d1", label: "D1" }]);
    expect(snapshot.selectedDatasetId).toBe("d1");
    expect(snapshot.sklearnTools).toEqual(["pca_transform"]);
    expect(snapshot.tableRows).toEqual([{ x: 1 }]);
    expect(snapshot.tableColumns).toEqual(["x"]);
    expect(snapshot.numericMatrix).toEqual([[1]]);
    expect(snapshot.featureNames).toEqual(["x"]);
    expect(snapshot.pcaChartSpec?.id).toBe("chart-1");
    expect(snapshot.isLoading).toBe(true);
    expect(snapshot.error).toBe("err");
  });
});
