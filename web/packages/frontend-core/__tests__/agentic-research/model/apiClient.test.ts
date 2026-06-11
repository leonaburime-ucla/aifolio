import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchAgenticDatasetManifest,
  fetchAgenticSklearnTools,
  fetchAgenticDatasetRows,
  fetchAgenticPcaChartSpec,
} from "../../../src/agentic-research/model/apiClient";

function mockFetch(body: unknown, ok = true) {
  return vi.fn().mockResolvedValue({
    ok,
    status: ok ? 200 : 500,
    json: async () => body,
  });
}

const BASE_URL = "http://test-api";
const runtimeDeps = (fetchImpl: typeof fetch) => ({
  runtimeDeps: { fetchImpl, resolveBaseUrl: () => BASE_URL },
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchAgenticDatasetManifest", () => {
  it("maps backend datasets to manifest entries", async () => {
    const fetchImpl = mockFetch({
      datasets: [
        { id: "iris.csv", label: "Iris", format: "csv" },
        { id: "wine.csv" },
      ],
    });

    const result = await fetchAgenticDatasetManifest({}, runtimeDeps(fetchImpl));
    expect(fetchImpl).toHaveBeenCalledWith(`${BASE_URL}/ml-data`);
    expect(result).toEqual([
      { id: "iris.csv", label: "Iris", description: "CSV dataset from backend/data/ml_data" },
      { id: "wine.csv", label: "wine.csv", description: "Dataset from backend/data/ml_data" },
    ]);
  });

  it("returns empty array when datasets key is missing", async () => {
    const fetchImpl = mockFetch({});
    const result = await fetchAgenticDatasetManifest({}, runtimeDeps(fetchImpl));
    expect(result).toEqual([]);
  });

  it("throws on non-ok response", async () => {
    const fetchImpl = mockFetch({}, false);
    await expect(fetchAgenticDatasetManifest({}, runtimeDeps(fetchImpl))).rejects.toThrow(
      "Failed to load dataset manifest."
    );
  });
});

describe("fetchAgenticSklearnTools", () => {
  it("uses global fetch and an empty base URL when runtime deps are omitted", async () => {
    const fetchImpl = mockFetch({ tools: ["pca"] });
    vi.stubGlobal("fetch", fetchImpl);

    const result = await fetchAgenticSklearnTools();

    expect(fetchImpl).toHaveBeenCalledWith("/sklearn-tools");
    expect(result).toEqual(["pca"]);
  });

  it("returns tools array from backend", async () => {
    const fetchImpl = mockFetch({ tools: ["pca", "svd", "nmf"] });
    const result = await fetchAgenticSklearnTools({}, runtimeDeps(fetchImpl));
    expect(fetchImpl).toHaveBeenCalledWith(`${BASE_URL}/sklearn-tools`);
    expect(result).toEqual(["pca", "svd", "nmf"]);
  });

  it("returns empty array when tools key is missing", async () => {
    const fetchImpl = mockFetch({});
    const result = await fetchAgenticSklearnTools({}, runtimeDeps(fetchImpl));
    expect(result).toEqual([]);
  });

  it("throws on non-ok response", async () => {
    const fetchImpl = mockFetch({}, false);
    await expect(fetchAgenticSklearnTools({}, runtimeDeps(fetchImpl))).rejects.toThrow(
      "Failed to load sklearn tools."
    );
  });
});

describe("fetchAgenticDatasetRows", () => {
  it("fetches rows for a given dataset id", async () => {
    const body = { rows: [{ a: 1 }], columns: ["a"] };
    const fetchImpl = mockFetch(body);
    const result = await fetchAgenticDatasetRows(
      { datasetId: "iris.csv" },
      runtimeDeps(fetchImpl)
    );
    expect(fetchImpl).toHaveBeenCalledWith(`${BASE_URL}/ml-data/iris.csv`);
    expect(result).toEqual(body);
  });

  it("encodes dataset id in URL", async () => {
    const fetchImpl = mockFetch({ rows: [], columns: [] });
    await fetchAgenticDatasetRows(
      { datasetId: "file with spaces.csv" },
      runtimeDeps(fetchImpl)
    );
    expect(fetchImpl).toHaveBeenCalledWith(
      `${BASE_URL}/ml-data/file%20with%20spaces.csv`
    );
  });

  it("throws on non-ok response", async () => {
    const fetchImpl = mockFetch({}, false);
    await expect(
      fetchAgenticDatasetRows({ datasetId: "x" }, runtimeDeps(fetchImpl))
    ).rejects.toThrow("Failed to load dataset file.");
  });
});

describe("fetchAgenticPcaChartSpec", () => {
  it("returns null on non-ok response", async () => {
    const fetchImpl = mockFetch({}, false);
    const result = await fetchAgenticPcaChartSpec(
      { data: [[1, 2]], feature_names: ["a", "b"] },
      runtimeDeps(fetchImpl)
    );
    expect(result).toBeNull();
  });

  it("returns null when result is missing", async () => {
    const fetchImpl = mockFetch({ status: "ok", result: null });
    const result = await fetchAgenticPcaChartSpec(
      { data: [[1, 2]], feature_names: ["a", "b"] },
      runtimeDeps(fetchImpl)
    );
    expect(result).toBeNull();
  });

  it("sends correct POST payload", async () => {
    const fetchImpl = mockFetch({ status: "ok", result: null });
    await fetchAgenticPcaChartSpec(
      { data: [[1, 2]], feature_names: ["a", "b"], n_components: 3, dataset_id: "d1" },
      runtimeDeps(fetchImpl)
    );
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${BASE_URL}/llm/ds`);
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body);
    expect(body.tool_args.n_components).toBe(3);
    expect(body.tool_args.dataset_id).toBe("d1");
    expect(body.tool_args.feature_names).toEqual(["a", "b"]);
  });

  it("returns chart spec on valid PCA result", async () => {
    const fetchImpl = mockFetch({
      status: "ok",
      result: {
        transformed: [[0.5, -0.3], [1.2, 0.8]],
        explained_variance_ratio: [0.6, 0.3],
        feature_names: ["a", "b"],
      },
    });
    const result = await fetchAgenticPcaChartSpec(
      { data: [[1, 2], [3, 4]], feature_names: ["a", "b"] },
      runtimeDeps(fetchImpl)
    );
    expect(result).not.toBeNull();
    expect(result!.type).toBe("scatter");
  });
});
