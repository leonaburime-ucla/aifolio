import { describe, expect, it } from "vitest";
import {
  agenticResearchStore,
  getActiveDatasetPayload,
  getAgenticResearchSnapshot,
} from "@/features/agentic-research/typescript/react/state/zustand/agenticResearchStore";

describe("agenticResearchStore", () => {
  it("updates state and returns snapshot/payload", () => {
    agenticResearchStore.setState({
      datasetManifest: [{ id: "iris", label: "Iris" }],
      selectedDatasetId: "iris",
      sklearnTools: ["pca_transform"],
      tableRows: [{ a: 1 }, { a: 2 }, { a: 3 }],
      tableColumns: ["a"],
      numericMatrix: [[1], [2], [3]],
      featureNames: ["a"],
      pcaChartSpec: null,
      isLoading: false,
      error: null,
    });

    const snapshot = getAgenticResearchSnapshot();
    expect(snapshot.selectedDatasetId).toBe("iris");
    expect(snapshot.tableRows).toHaveLength(3);

    const payload = getActiveDatasetPayload(2);
    expect(payload).toEqual({
      datasetId: "iris",
      columns: ["a"],
      rows: [{ a: 1 }, { a: 2 }],
      featureNames: ["a"],
      numericMatrix: [[1], [2]],
    });
  });

  it("supports action setters", () => {
    const state = agenticResearchStore.getState();
    state.setLoading(true);
    state.setError("boom");
    state.setSelectedDatasetId("wine");

    const next = agenticResearchStore.getState();
    expect(next.isLoading).toBe(true);
    expect(next.error).toBe("boom");
    expect(next.selectedDatasetId).toBe("wine");
  });
});
