import { describe, it, expect, vi, beforeEach } from "vitest";
import { nextTick } from "vue";
import { useDatasetLoader, createDatasetLoaderApi } from "~/features/ml/model";
import type { DatasetLoaderApi } from "~/features/ml/model";


function createMockApi(overrides: Partial<DatasetLoaderApi> = {}): DatasetLoaderApi {
  return {
    fetchManifest: vi.fn().mockResolvedValue({
      datasets: [
        { id: "churn.csv", label: "Churn" },
        { id: "iris.csv" },
      ],
    }),
    fetchDataset: vi.fn().mockResolvedValue({
      rows: [{ col1: "a", col2: 1 }, { col1: "b", col2: 2 }],
      columns: ["col1", "col2"],
    }),
    ...overrides,
  };
}

describe("useDatasetLoader", () => {
  let api: DatasetLoaderApi;
  let loader: ReturnType<typeof useDatasetLoader>;

  beforeEach(() => {
    api = createMockApi();
    loader = useDatasetLoader({ baseUrl: "/api/ai", api });
  });

  describe("loadManifest()", () => {
    it("populates dataset options", async () => {
      await loader.loadManifest();

      expect(loader.datasetOptions.value).toEqual([
        { id: "churn.csv", label: "Churn" },
        { id: "iris.csv", label: "iris.csv" },
      ]);
    });

    it("auto-selects first dataset", async () => {
      await loader.loadManifest();
      expect(loader.selectedDatasetId.value).toBe("churn.csv");
    });

    it("does not overwrite existing selection", async () => {
      loader.selectedDatasetId.value = "existing.csv";
      await loader.loadManifest();
      expect(loader.selectedDatasetId.value).toBe("existing.csv");
    });

    it("handles fetch failure", async () => {
      const failApi = createMockApi({
        fetchManifest: vi.fn().mockRejectedValue(new Error("Network")),
      });
      const l = useDatasetLoader({ baseUrl: "/api/ai", api: failApi });
      await l.loadManifest();

      expect(l.datasetError.value).toBe("Network");
      expect(l.datasetOptions.value).toHaveLength(0);
    });

    it("handles non-Error throw", async () => {
      const failApi = createMockApi({
        fetchManifest: vi.fn().mockRejectedValue(42),
      });
      const l = useDatasetLoader({ baseUrl: "/api/ai", api: failApi });
      await l.loadManifest();

      expect(l.datasetError.value).toBe("Error");
    });
  });

  describe("loadDataset()", () => {
    it("populates rows and columns", async () => {
      await loader.loadDataset("churn.csv");

      expect(loader.tableRows.value).toHaveLength(2);
      expect(loader.tableColumns.value).toEqual(["col1", "col2"]);
    });

    it("auto-selects last column as target", async () => {
      await loader.loadDataset("churn.csv");
      expect(loader.targetColumn.value).toBe("col2");
    });

    it("does not overwrite existing target", async () => {
      loader.targetColumn.value = "existing";
      await loader.loadDataset("churn.csv");
      expect(loader.targetColumn.value).toBe("existing");
    });

    it("infers columns from first row when not provided", async () => {
      const noColApi = createMockApi({
        fetchDataset: vi.fn().mockResolvedValue({
          rows: [{ x: 1, y: 2, z: 3 }],
        }),
      });
      const l = useDatasetLoader({ baseUrl: "/api/ai", api: noColApi });
      await l.loadDataset("test.csv");

      expect(l.tableColumns.value).toEqual(["x", "y", "z"]);
    });

    it("clears error before loading", async () => {
      loader.datasetError.value = "stale";
      await loader.loadDataset("churn.csv");
      expect(loader.datasetError.value).toBeNull();
    });

    it("handles fetch failure", async () => {
      const failApi = createMockApi({
        fetchDataset: vi.fn().mockRejectedValue(new Error("404")),
      });
      const l = useDatasetLoader({ baseUrl: "/api/ai", api: failApi });
      await l.loadDataset("bad.csv");

      expect(l.datasetError.value).toBe("404");
    });

    it("handles non-Error throw", async () => {
      const failApi = createMockApi({
        fetchDataset: vi.fn().mockRejectedValue("string err"),
      });
      const l = useDatasetLoader({ baseUrl: "/api/ai", api: failApi });
      await l.loadDataset("bad.csv");

      expect(l.datasetError.value).toBe("Error");
    });

    it("handles missing rows in response", async () => {
      const emptyApi = createMockApi({
        fetchDataset: vi.fn().mockResolvedValue({}),
      });
      const l = useDatasetLoader({ baseUrl: "/api/ai", api: emptyApi });
      await l.loadDataset("empty.csv");

      expect(l.tableRows.value).toEqual([]);
      expect(l.tableColumns.value).toEqual([]);
    });
  });

  describe("onDatasetChange()", () => {
    it("updates selectedDatasetId", () => {
      loader.onDatasetChange("iris.csv");
      expect(loader.selectedDatasetId.value).toBe("iris.csv");
    });
  });

  describe("watcher", () => {
    it("auto-loads dataset when selectedDatasetId changes", async () => {
      loader.selectedDatasetId.value = "iris.csv";
      await nextTick();
      // Give the watcher time to fire
      await new Promise((r) => setTimeout(r, 10));

      expect(api.fetchDataset).toHaveBeenCalledWith("iris.csv");
    });

    it("does not load when set to null", async () => {
      loader.selectedDatasetId.value = null;
      await nextTick();
      await new Promise((r) => setTimeout(r, 10));

      expect(api.fetchDataset).not.toHaveBeenCalled();
    });
  });

  describe("createDatasetLoaderApi", () => {
    it("fetchManifest calls /ml-data endpoint", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ datasets: [{ id: "a.csv" }] }),
      });
      vi.stubGlobal("fetch", mockFetch);

      const realApi = createDatasetLoaderApi({ baseUrl: "/api/ai" });
      const result = await realApi.fetchManifest();

      expect(mockFetch).toHaveBeenCalledWith("/api/ai/ml-data");
      expect(result.datasets).toEqual([{ id: "a.csv" }]);
      vi.unstubAllGlobals();
    });

    it("fetchManifest handles missing datasets field", async () => {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      }));

      const realApi = createDatasetLoaderApi({ baseUrl: "/api/ai" });
      const result = await realApi.fetchManifest();

      expect(result.datasets).toEqual([]);
      vi.unstubAllGlobals();
    });

    it("fetchManifest throws on non-ok response", async () => {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false }));

      const realApi = createDatasetLoaderApi({ baseUrl: "/api/ai" });
      await expect(realApi.fetchManifest()).rejects.toThrow("Failed to load datasets.");
      vi.unstubAllGlobals();
    });

    it("fetchDataset calls /ml-data/:id endpoint", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ rows: [{ x: 1 }], columns: ["x"] }),
      });
      vi.stubGlobal("fetch", mockFetch);

      const realApi = createDatasetLoaderApi({ baseUrl: "/api/ai" });
      const result = await realApi.fetchDataset("churn.csv");

      expect(mockFetch).toHaveBeenCalledWith("/api/ai/ml-data/churn.csv");
      expect(result.rows).toEqual([{ x: 1 }]);
      vi.unstubAllGlobals();
    });

    it("fetchDataset throws on non-ok response", async () => {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false }));

      const realApi = createDatasetLoaderApi({ baseUrl: "/api/ai" });
      await expect(realApi.fetchDataset("bad")).rejects.toThrow("Failed to load dataset.");
      vi.unstubAllGlobals();
    });

    it("fetchDataset encodes dataset id", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ rows: [] }),
      });
      vi.stubGlobal("fetch", mockFetch);

      const realApi = createDatasetLoaderApi({ baseUrl: "/api/ai" });
      await realApi.fetchDataset("file with spaces.csv");

      expect(mockFetch).toHaveBeenCalledWith("/api/ai/ml-data/file%20with%20spaces.csv");
      vi.unstubAllGlobals();
    });
  });
});
