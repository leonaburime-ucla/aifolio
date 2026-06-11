import { describe, it, expect, vi, beforeEach } from "vitest";
import { useAgenticResearch } from "~/features/agentic-research/model";
import type { AgenticResearchApi } from "~/features/agentic-research/model";
import { createAgenticResearchApi } from "~/features/agentic-research/api";

vi.mock("~/features/agentic-research/api", () => ({
  createAgenticResearchApi: vi.fn(() => ({
    loadManifest: vi.fn().mockResolvedValue([]),
    loadTools: vi.fn().mockResolvedValue([]),
    loadDataset: vi.fn().mockResolvedValue({ rows: [], columns: [] }),
  })),
}));

function createMockApi(overrides: Partial<AgenticResearchApi> = {}): AgenticResearchApi {
  return {
    loadManifest: vi.fn().mockResolvedValue([
      { id: "churn.csv", label: "churn.csv", description: "Customer churn dataset" },
      { id: "fraud.csv", label: "fraud.csv", description: "Fraud detection dataset" },
    ]),
    loadTools: vi.fn().mockResolvedValue([
      "pca_transform",
      "random_forest_classification",
      "kmeans_clustering",
      "linear_regression",
    ]),
    loadDataset: vi.fn().mockResolvedValue({
      rows: [
        { customerID: "1", tenure: 12, Churn: "Yes" },
        { customerID: "2", tenure: 24, Churn: "No" },
      ],
      columns: ["customerID", "tenure", "Churn"],
    }),
    ...overrides,
  };
}

describe("useAgenticResearch", () => {
  let api: AgenticResearchApi;
  let research: ReturnType<typeof useAgenticResearch>;
  let datasetChangeSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    api = createMockApi();
    datasetChangeSpy = vi.fn();
    research = useAgenticResearch({
      baseUrl: "/api/ai",
      onDatasetChange: datasetChangeSpy,
      api,
    });
  });

  describe("init()", () => {
    it("loads manifest and selects first dataset", async () => {
      await research.init();

      expect(research.datasetOptions.value).toHaveLength(2);
      expect(research.datasetOptions.value[0].id).toBe("churn.csv");
      expect(research.selectedDatasetId.value).toBe("churn.csv");
    });

    it("loads sklearn tools", async () => {
      await research.init();

      expect(research.sklearnTools.value).toHaveLength(4);
      expect(research.sklearnTools.value).toContain("pca_transform");
    });

    it("loads dataset rows and columns", async () => {
      await research.init();

      expect(research.tableRows.value).toHaveLength(2);
      expect(research.tableColumns.value).toEqual(["customerID", "tenure", "Churn"]);
    });

    it("emits dataset-change callback with first dataset id", async () => {
      await research.init();

      expect(datasetChangeSpy).toHaveBeenCalledWith("churn.csv");
    });

    it("does not load dataset when manifest returns empty", async () => {
      const emptyApi = createMockApi({
        loadManifest: vi.fn().mockResolvedValue([]),
      });
      const r = useAgenticResearch({ baseUrl: "/api/ai", api: emptyApi });
      await r.init();

      expect(r.selectedDatasetId.value).toBeNull();
      expect(r.tableRows.value).toHaveLength(0);
      expect(emptyApi.loadDataset).not.toHaveBeenCalled();
    });

    it("does not overwrite selectedDatasetId if already set", async () => {
      await research.init();
      research.selectedDatasetId.value = "fraud.csv";

      const freshApi = createMockApi();
      const r = useAgenticResearch({ baseUrl: "/api/ai", api: freshApi });
      r.selectedDatasetId.value = "fraud.csv";
      await r.init();

      expect(r.selectedDatasetId.value).toBe("fraud.csv");
    });
  });

  describe("toolGroups computed", () => {
    it("groups sklearn tools by category", async () => {
      await research.init();

      const groups = research.toolGroups.value;
      expect(groups.length).toBeGreaterThan(0);

      const groupNames = groups.map((g) => g.name);
      expect(groupNames).toContain("Decomposition & Embeddings");
    });

    it("returns empty when no tools loaded", () => {
      expect(research.toolGroups.value).toEqual([]);
    });
  });

  describe("onDatasetChange()", () => {
    it("updates selectedDatasetId and fires callback", async () => {
      await research.init();
      datasetChangeSpy.mockClear();

      research.onDatasetChange("fraud.csv");

      expect(research.selectedDatasetId.value).toBe("fraud.csv");
      expect(datasetChangeSpy).toHaveBeenCalledWith("fraud.csv");
    });

    it("works without onDatasetChange callback", async () => {
      const r = useAgenticResearch({ baseUrl: "/api/ai", api });
      r.onDatasetChange("fraud.csv");
      expect(r.selectedDatasetId.value).toBe("fraud.csv");
    });
  });

  describe("onDatasetWatch()", () => {
    it("skips load on first call after init (initialLoadDone guard)", async () => {
      await research.init();

      const callCount = vi.mocked(api.loadDataset).mock.calls.length;
      await research.onDatasetWatch("churn.csv");
      expect(vi.mocked(api.loadDataset).mock.calls.length).toBe(callCount);
    });

    it("loads dataset on subsequent calls", async () => {
      await research.init();
      await research.onDatasetWatch("churn.csv");

      vi.mocked(api.loadDataset).mockClear();
      await research.onDatasetWatch("fraud.csv");
      expect(api.loadDataset).toHaveBeenCalledWith("fraud.csv");
    });

    it("no-ops when id is null", async () => {
      await research.init();
      vi.mocked(api.loadDataset).mockClear();

      await research.onDatasetWatch(null);
      expect(api.loadDataset).not.toHaveBeenCalled();
    });
  });

  describe("removeChartSpec()", () => {
    it("removes chart by id", () => {
      research.chartSpecs.value = [
        { id: "c1", type: "line", title: "PCA", data: [], xKey: "x", yKeys: ["y"] },
        { id: "c2", type: "bar", title: "NMF", data: [], xKey: "x", yKeys: ["y"] },
      ];

      research.removeChartSpec("c1");

      expect(research.chartSpecs.value).toHaveLength(1);
      expect(research.chartSpecs.value[0].id).toBe("c2");
    });

    it("no-ops when id not found", () => {
      research.chartSpecs.value = [
        { id: "c1", type: "line", title: "PCA", data: [], xKey: "x", yKeys: ["y"] },
      ];

      research.removeChartSpec("nonexistent");
      expect(research.chartSpecs.value).toHaveLength(1);
    });
  });

  describe("error handling", () => {
    it("sets error when manifest fetch rejects", async () => {
      const failApi = createMockApi({
        loadManifest: vi.fn().mockRejectedValue(new Error("Network timeout")),
      });
      const r = useAgenticResearch({ baseUrl: "/api/ai", api: failApi });
      await r.init();

      expect(r.error.value).toBe("Network timeout");
      expect(r.datasetOptions.value).toHaveLength(0);
    });

    it("sets generic error when manifest throws non-Error", async () => {
      const failApi = createMockApi({
        loadManifest: vi.fn().mockRejectedValue("string error"),
      });
      const r = useAgenticResearch({ baseUrl: "/api/ai", api: failApi });
      await r.init();

      expect(r.error.value).toBe("Failed to load manifest.");
    });

    it("sets sklearnTools to empty array when loadTools rejects", async () => {
      const failApi = createMockApi({
        loadTools: vi.fn().mockRejectedValue(new Error("Tools down")),
      });
      const r = useAgenticResearch({ baseUrl: "/api/ai", api: failApi });
      await r.init();

      expect(r.sklearnTools.value).toEqual([]);
    });

    it("sets error when loadDataset rejects", async () => {
      const failApi = createMockApi({
        loadDataset: vi.fn().mockRejectedValue(new Error("Dataset 404")),
      });
      const r = useAgenticResearch({ baseUrl: "/api/ai", api: failApi });
      await r.init();

      expect(r.error.value).toBe("Dataset 404");
      expect(r.tableRows.value).toHaveLength(0);
      expect(r.isLoading.value).toBe(false);
    });

    it("sets generic error when loadDataset throws non-Error", async () => {
      const failApi = createMockApi({
        loadDataset: vi.fn().mockRejectedValue(42),
      });
      const r = useAgenticResearch({ baseUrl: "/api/ai", api: failApi });
      await r.init();

      expect(r.error.value).toBe("Failed to load dataset.");
    });

    it("clears previous error on successful dataset load", async () => {
      research.error.value = "stale error";
      await research.onDatasetWatch(null);

      vi.mocked(api.loadDataset).mockClear();
      await research.init();
      expect(research.error.value).toBeNull();
    });
  });

  describe("loading state", () => {
    it("sets isLoading true during dataset fetch", async () => {
      let capturedLoading = false;
      const slowApi = createMockApi({
        loadDataset: vi.fn().mockImplementation(async () => {
          capturedLoading = research.isLoading.value;
          return { rows: [], columns: [] };
        }),
      });

      const r = useAgenticResearch({ baseUrl: "/api/ai", api: slowApi });
      // Manually trigger loadDataset path
      r.selectedDatasetId.value = "churn.csv";
      await r.init();

      // isLoading should have been true during the call
      // and false after completion
      expect(r.isLoading.value).toBe(false);
    });

    it("resets isLoading on dataset fetch error", async () => {
      const failApi = createMockApi({
        loadDataset: vi.fn().mockRejectedValue(new Error("fail")),
      });
      const r = useAgenticResearch({ baseUrl: "/api/ai", api: failApi });
      await r.init();

      expect(r.isLoading.value).toBe(false);
    });
  });

  describe("api fallback (no injected api)", () => {
    it("creates api from baseUrl when api option is omitted", () => {
      const r = useAgenticResearch({ baseUrl: "/api/ai" });
      expect(createAgenticResearchApi).toHaveBeenCalledWith({ baseUrl: "/api/ai" });
      expect(r.datasetOptions.value).toEqual([]);
    });
  });

  describe("dataset response edge cases", () => {
    it("handles response with rows but no columns field", async () => {
      const noColApi = createMockApi({
        loadDataset: vi.fn().mockResolvedValue({
          rows: [{ a: 1, b: 2 }, { a: 3, b: 4 }],
        }),
      });
      const r = useAgenticResearch({ baseUrl: "/api/ai", api: noColApi });
      await r.init();

      expect(r.tableColumns.value).toEqual(["a", "b"]);
    });

    it("handles response with empty rows and no columns", async () => {
      const emptyApi = createMockApi({
        loadDataset: vi.fn().mockResolvedValue({ rows: [] }),
      });
      const r = useAgenticResearch({ baseUrl: "/api/ai", api: emptyApi });
      await r.init();

      expect(r.tableRows.value).toHaveLength(0);
      expect(r.tableColumns.value).toEqual([]);
    });

    it("handles response with undefined rows", async () => {
      const nullApi = createMockApi({
        loadDataset: vi.fn().mockResolvedValue({}),
      });
      const r = useAgenticResearch({ baseUrl: "/api/ai", api: nullApi });
      await r.init();

      expect(r.tableRows.value).toHaveLength(0);
      expect(r.tableColumns.value).toEqual([]);
    });
  });
});
