import { describe, expect, it, vi } from "vitest";
import { createAgenticResearchApiAdapter } from "@/features/agentic-research/typescript/api/agenticResearchApi.adapter";

vi.mock("@/features/agentic-research/typescript/api/agenticResearchApi", () => ({
  fetchDatasetManifest: vi.fn(async () => []),
  fetchSklearnTools: vi.fn(async () => []),
  fetchDatasetRows: vi.fn(async () => ({ rows: [], columns: [] })),
  fetchPcaChartSpec: vi.fn(async () => null),
}));

describe("agenticResearchApi.adapter", () => {
  it("returns default API functions", () => {
    const api = createAgenticResearchApiAdapter({});
    expect(typeof api.fetchDatasetManifest).toBe("function");
    expect(typeof api.fetchSklearnTools).toBe("function");
    expect(typeof api.fetchDatasetRows).toBe("function");
    expect(typeof api.fetchPcaChartSpec).toBe("function");
  });

  it("respects injected function overrides", async () => {
    const fetchDatasetManifest = vi.fn(async () => [{ id: "a", label: "A" }]);
    const fetchSklearnTools = vi.fn(async () => ["pca_transform"]);
    const fetchDatasetRows = vi.fn(async () => ({ rows: [{ x: 1 }], columns: ["x"] }));
    const fetchPcaChartSpec = vi.fn(async () => ({
      id: "chart",
      title: "Chart",
      type: "scatter" as const,
      xKey: "x",
      yKeys: ["y"],
      data: [{ x: 1, y: 2 }],
    }));

    const api = createAgenticResearchApiAdapter(
      {},
      {
        fetchDatasetManifest,
        fetchSklearnTools,
        fetchDatasetRows,
        fetchPcaChartSpec,
      }
    );

    await api.fetchDatasetManifest();
    await api.fetchSklearnTools();
    await api.fetchDatasetRows("a");
    await api.fetchPcaChartSpec({ data: [[1, 2]] });

    expect(fetchDatasetManifest).toHaveBeenCalledTimes(1);
    expect(fetchSklearnTools).toHaveBeenCalledTimes(1);
    expect(fetchDatasetRows).toHaveBeenCalledWith("a");
    expect(fetchPcaChartSpec).toHaveBeenCalledWith({ data: [[1, 2]] });
  });
});
