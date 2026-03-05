import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_ML_DATASET_ID,
  fetchMlDatasetOptions,
  fetchMlDatasetRows,
} from "@/features/ml/typescript/api/mlDataApi";

describe("mlDataApi", () => {
  it("exposes the expected default dataset id", () => {
    expect(DEFAULT_ML_DATASET_ID).toBe("customer_churn_telco.csv");
  });

  it("fetches and maps dataset options", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        datasets: [
          { id: "d1.csv", label: "Dataset 1", format: "csv" },
          { id: "d2.xlsx", format: "xlsx" },
        ],
      }),
    }));

    const options = await fetchMlDatasetOptions({
      fetchImpl: fetchImpl as unknown as typeof fetch,
      resolveBaseUrl: () => "http://ml-api",
    });

    expect(fetchImpl).toHaveBeenCalledWith("http://ml-api/ml-data", {
      cache: "no-store",
    });
    expect(options).toEqual([
      { id: "d1.csv", label: "Dataset 1", description: "CSV dataset from backend/data/ml_data" },
      { id: "d2.xlsx", label: "d2.xlsx", description: "XLSX dataset from backend/data/ml_data" },
    ]);
  });

  it("throws on non-ok dataset options response", async () => {
    const fetchImpl = vi.fn(async () => ({ ok: false }));
    await expect(
      fetchMlDatasetOptions({
        fetchImpl: fetchImpl as unknown as typeof fetch,
        resolveBaseUrl: () => "http://ml-api",
      })
    ).rejects.toThrow("Failed to load ML datasets.");
  });

  it("fetches dataset rows by id", async () => {
    const payload = { columns: ["a"], rows: [{ a: 1 }], rowCount: 1, totalRowCount: 1 };
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => payload,
    }));

    const rows = await fetchMlDatasetRows({ datasetId: "d1.csv" }, {
      fetchImpl: fetchImpl as unknown as typeof fetch,
      resolveBaseUrl: () => "http://ml-api",
    });

    expect(fetchImpl).toHaveBeenCalledWith("http://ml-api/ml-data/d1.csv");
    expect(rows).toEqual(payload);
  });

  it("maps manifest entries without format and throws on rows fetch failure", async () => {
    const manifestFetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        datasets: [{ id: "d3.bin", label: undefined, format: undefined }],
      }),
    }));
    const options = await fetchMlDatasetOptions({
      fetchImpl: manifestFetch as unknown as typeof fetch,
      resolveBaseUrl: () => "http://ml-api",
    });

    expect(options).toEqual([
      { id: "d3.bin", label: "d3.bin", description: "Dataset from backend/data/ml_data" },
    ]);

    const rowsFetch = vi.fn(async () => ({ ok: false }));
    await expect(
      fetchMlDatasetRows({ datasetId: "d3.bin" }, {
        fetchImpl: rowsFetch as unknown as typeof fetch,
        resolveBaseUrl: () => "http://ml-api",
      })
    ).rejects.toThrow("Failed to load dataset rows.");
  });

  it("returns an empty dataset option list when manifest omits datasets", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({}),
    }));

    const options = await fetchMlDatasetOptions({
      fetchImpl: fetchImpl as unknown as typeof fetch,
      resolveBaseUrl: () => "http://ml-api",
    });

    expect(options).toEqual([]);
  });
});
