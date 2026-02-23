import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchDatasetManifest,
  fetchDatasetRows,
  fetchPcaChartSpec,
  fetchSklearnTools,
} from "@/features/agentic-research/api/agenticResearchApi";

const originalFetch = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("agenticResearchApi", () => {
  it("fetchDatasetManifest returns datasets on success", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ datasets: [{ id: "d1", label: "D1" }] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    ) as any;

    const datasets = await fetchDatasetManifest();
    expect(datasets).toEqual([{ id: "d1", label: "D1" }]);
  });

  it("fetchDatasetManifest throws user-safe error on non-ok", async () => {
    global.fetch = vi.fn(async () => new Response("{}", { status: 500 })) as any;
    await expect(fetchDatasetManifest()).rejects.toThrow("Failed to load dataset manifest.");
  });

  it("fetchDatasetManifest returns empty array when payload omits datasets", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ status: "ok" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    ) as any;

    const datasets = await fetchDatasetManifest();
    expect(datasets).toEqual([]);
  });

  it("fetchSklearnTools returns tools on success", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ tools: ["pca_transform", "linear_regression"] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    ) as any;

    const tools = await fetchSklearnTools();
    expect(tools).toEqual(["pca_transform", "linear_regression"]);
  });

  it("fetchSklearnTools throws user-safe error on non-ok", async () => {
    global.fetch = vi.fn(async () => new Response("{}", { status: 500 })) as any;
    await expect(fetchSklearnTools()).rejects.toThrow("Failed to load sklearn tools.");
  });

  it("fetchSklearnTools returns empty array when payload omits tools", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ status: "ok" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    ) as any;

    const tools = await fetchSklearnTools();
    expect(tools).toEqual([]);
  });

  it("fetchDatasetRows returns row payload on success", async () => {
    global.fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({
          rows: [{ a: 1, b: 2 }],
          columns: ["a", "b"],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    ) as any;

    const rows = await fetchDatasetRows("dataset-a");
    expect(rows.rows).toEqual([{ a: 1, b: 2 }]);
    expect(rows.columns).toEqual(["a", "b"]);
  });

  it("fetchDatasetRows throws user-safe error on non-ok", async () => {
    global.fetch = vi.fn(async () => new Response("{}", { status: 404 })) as any;
    await expect(fetchDatasetRows("missing")).rejects.toThrow("Failed to load dataset file.");
  });

  it("fetchPcaChartSpec returns null on non-ok response", async () => {
    global.fetch = vi.fn(async () => new Response("{}", { status: 500 })) as any;
    const result = await fetchPcaChartSpec({ data: [[1, 2]] });
    expect(result).toBeNull();
  });

  it("fetchPcaChartSpec builds scatter chart when PCA result exists", async () => {
    global.fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({
          result: {
            transformed: [
              [0.1, 0.2],
              [0.3, 0.4],
            ],
            explained_variance_ratio: [0.6, 0.3],
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    ) as any;

    const result = await fetchPcaChartSpec({ data: [[1, 2], [3, 4]], n_components: 2 });

    expect(result?.type).toBe("scatter");
    expect(result?.id).toBe("agentic-research-pca");
    expect(result?.data).toHaveLength(2);
    expect(result?.xKey).toBe("pc1");
    expect(result?.yKeys).toEqual(["pc2"]);
  });

  it("fetchPcaChartSpec returns null when result payload is missing", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ status: "ok", result: null }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    ) as any;

    const result = await fetchPcaChartSpec({ data: [[1, 2]] });
    expect(result).toBeNull();
  });

  it("fetchPcaChartSpec returns null when transformed points are empty", async () => {
    global.fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({
          result: {
            transformed: [],
            explained_variance_ratio: [0.9, 0.1],
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    ) as any;

    const result = await fetchPcaChartSpec({ data: [[1, 2]] });
    expect(result).toBeNull();
  });

  it("fetchPcaChartSpec omits variance description when fewer than two variance values exist", async () => {
    global.fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({
          result: {
            transformed: [[0.1, 0.2]],
            explained_variance_ratio: [0.9],
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    ) as any;

    const result = await fetchPcaChartSpec({ data: [[1, 2]] });
    expect(result?.description).toBeUndefined();
  });

  it("fetchPcaChartSpec falls back missing PCA coordinate values to zero", async () => {
    global.fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({
          result: {
            transformed: [[0.1], []],
            explained_variance_ratio: [0.8, 0.2],
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    ) as any;

    const result = await fetchPcaChartSpec({ data: [[1, 2]] });
    expect(result?.data).toEqual([
      { id: "pca-1", pc1: 0.1, pc2: 0 },
      { id: "pca-2", pc1: 0, pc2: 0 },
    ]);
  });

  it("fetchPcaChartSpec returns null when transformed field is missing", async () => {
    global.fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({
          result: {
            explained_variance_ratio: [0.8, 0.2],
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    ) as any;

    const result = await fetchPcaChartSpec({ data: [[1, 2]] });
    expect(result).toBeNull();
  });

  it("fetchPcaChartSpec handles missing explained_variance_ratio", async () => {
    global.fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({
          result: {
            transformed: [[0.1, 0.2]],
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    ) as any;

    const result = await fetchPcaChartSpec({ data: [[1, 2]] });
    expect(result?.description).toBeUndefined();
  });
});
