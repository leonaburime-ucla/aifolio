import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchDatasetManifest,
  fetchDatasetRows,
  fetchPcaChartSpec,
  fetchSklearnTools,
} from "@/features/agentic-research/typescript/api/agenticResearchApi";

vi.mock("@/core/config/aiApi", () => ({
  getAiApiBaseUrl: vi.fn(() => "http://ai-api"),
}));

describe("agenticResearchApi", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("maps dataset manifest payload into agentic-research schema", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          datasets: [
            { id: "iris.csv", label: "Iris", format: "csv" },
            { id: "wine.csv" },
          ],
        }),
      }))
    );

    await expect(fetchDatasetManifest()).resolves.toEqual([
      {
        id: "iris.csv",
        label: "Iris",
        description: "CSV dataset from backend/data/ml_data",
      },
      {
        id: "wine.csv",
        label: "wine.csv",
        description: "Dataset from backend/data/ml_data",
      },
    ]);

    expect(fetch).toHaveBeenCalledWith("http://ai-api/ml-data");
  });

  it("defaults missing manifest/tool arrays to empty arrays", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchDatasetManifest()).resolves.toEqual([]);
    await expect(fetchSklearnTools()).resolves.toEqual([]);
  });

  it("throws for non-ok manifest/tools/rows endpoints", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
      }))
    );

    await expect(fetchDatasetManifest()).rejects.toThrow(
      "Failed to load dataset manifest."
    );
    await expect(fetchSklearnTools()).rejects.toThrow(
      "Failed to load sklearn tools."
    );
    await expect(fetchDatasetRows("iris.csv")).rejects.toThrow(
      "Failed to load dataset file."
    );
  });

  it("loads tools and rows payloads", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ tools: ["pca_transform"] }) })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ rows: [{ x: 1 }], columns: ["x"] }),
      });

    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchSklearnTools()).resolves.toEqual(["pca_transform"]);
    await expect(fetchDatasetRows("iris set.csv")).resolves.toEqual({
      rows: [{ x: 1 }],
      columns: ["x"],
    });

    expect(fetchMock).toHaveBeenNthCalledWith(1, "http://ai-api/sklearn-tools");
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://ai-api/ml-data/iris%20set.csv"
    );
  });

  it("returns null on non-ok/empty PCA responses and builds chart payload on success", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: "ok", result: null }) })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: "ok",
          result: {
            transformed: [
              [1.25, -0.5],
              [0.5, 1.1],
            ],
            explained_variance_ratio: [0.6, 0.3, 0.1],
          },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: "ok",
          result: {
            transformed: [[]],
            explained_variance_ratio: [0.9, 0.1],
          },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: "ok",
          result: {
            transformed: undefined,
            explained_variance_ratio: [0.8, 0.2],
          },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: "ok",
          result: {
            transformed: [[5]],
            explained_variance_ratio: undefined,
          },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: "ok",
          result: {
            transformed: [[2, 3]],
            explained_variance_ratio: [0.7],
          },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: "ok",
          result: {
            transformed: [],
            explained_variance_ratio: [0.5, 0.5],
          },
        }),
      });

    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchPcaChartSpec({ data: [[1, 2]] })).resolves.toBeNull();
    await expect(fetchPcaChartSpec({ data: [[1, 2]] })).resolves.toBeNull();

    await expect(
      fetchPcaChartSpec({
        data: [[1, 2]],
        n_components: 3,
        feature_names: ["a", "b"],
        dataset_id: "iris.csv",
        dataset_meta: { source: "test" },
      })
    ).resolves.toEqual({
      id: "agentic-research-pca",
      title: "PCA Projection",
      description: "Explained variance: PC1 60.0%, PC2 30.0%, PC3 10.0%",
      type: "scatter",
      xKey: "pc1",
      yKeys: ["pc2"],
      xLabel: "PC1",
      yLabel: "PC2",
      data: [
        { id: "pca-1", pc1: 1.25, pc2: -0.5 },
        { id: "pca-2", pc1: 0.5, pc2: 1.1 },
      ],
    });

    await expect(fetchPcaChartSpec({ data: [[1, 2]] })).resolves.toEqual({
      id: "agentic-research-pca",
      title: "PCA Projection",
      description: "Explained variance: PC1 90.0%, PC2 10.0%",
      type: "scatter",
      xKey: "pc1",
      yKeys: ["pc2"],
      xLabel: "PC1",
      yLabel: "PC2",
      data: [{ id: "pca-1", pc1: 0, pc2: 0 }],
    });

    await expect(fetchPcaChartSpec({ data: [[1, 2]] })).resolves.toBeNull();

    await expect(fetchPcaChartSpec({ data: [[1, 2]] })).resolves.toEqual({
      id: "agentic-research-pca",
      title: "PCA Projection",
      description: undefined,
      type: "scatter",
      xKey: "pc1",
      yKeys: ["pc2"],
      xLabel: "PC1",
      yLabel: "PC2",
      data: [{ id: "pca-1", pc1: 5, pc2: 0 }],
    });

    await expect(fetchPcaChartSpec({ data: [[1, 2]] })).resolves.toEqual({
      id: "agentic-research-pca",
      title: "PCA Projection",
      description: undefined,
      type: "scatter",
      xKey: "pc1",
      yKeys: ["pc2"],
      xLabel: "PC1",
      yLabel: "PC2",
      data: [{ id: "pca-1", pc1: 2, pc2: 3 }],
    });

    await expect(fetchPcaChartSpec({ data: [[1, 2]] })).resolves.toBeNull();

    expect(fetchMock).toHaveBeenCalledWith("http://ai-api/llm/ds", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "Run PCA and return the transformed points.",
        tool_name: "pca_transform",
        tool_args: {
          data: [[1, 2]],
          n_components: 2,
          feature_names: undefined,
          dataset_id: undefined,
          dataset_meta: undefined,
        },
      }),
    });
  });
});
