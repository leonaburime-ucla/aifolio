import { describe, it, expect, vi, beforeEach } from "vitest";
import { flushPromises } from "@vue/test-utils";
import { withSetup } from "./helpers/withSetup";
import { useTrainingScreenOrchestrator } from "~/features/ml/orchestrator";
import type { DatasetLoaderApi } from "~/features/ml/model";
import type { MlTrainingApi } from "~/features/ml/api";

vi.mock("~/features/ml/model/useDatasetLoader", async (importOriginal) => {
  const actual = await importOriginal<typeof import("~/features/ml/model/useDatasetLoader")>();
  return {
    ...actual,
    createDatasetLoaderApi: vi.fn(() => ({
      loadManifest: vi.fn().mockResolvedValue([]),
      loadDataset: vi.fn().mockResolvedValue({ rows: [], columns: [] }),
    })),
  };
});

function createMockDatasetApi(): DatasetLoaderApi {
  return {
    fetchManifest: vi.fn().mockResolvedValue({
      datasets: [{ id: "churn.csv", label: "churn.csv" }],
    }),
    fetchDataset: vi.fn().mockResolvedValue({
      rows: [{ tenure: 12 }],
      columns: ["tenure"],
    }),
  };
}

function createMockTrainingApi(): MlTrainingApi {
  return {
    trainPytorch: vi.fn().mockResolvedValue({ status: "ok", metrics: {} }),
    trainTensorflow: vi.fn().mockResolvedValue({ status: "ok", metrics: {} }),
  };
}

describe("useTrainingScreenOrchestrator", () => {
  let datasetApi: DatasetLoaderApi;
  let trainingApi: MlTrainingApi;

  beforeEach(() => {
    datasetApi = createMockDatasetApi();
    trainingApi = createMockTrainingApi();
  });

  it("loads manifest on mount", async () => {
    const [result] = withSetup(() =>
      useTrainingScreenOrchestrator({
        baseUrl: "/api/ai",
        framework: "pytorch",
        defaultTrainingMode: "linear_glm_baseline",
        datasetApi,
        trainingApi,
      })
    );

    await flushPromises();

    expect(datasetApi.fetchManifest).toHaveBeenCalledOnce();
    expect(result.datasetOptions.value).toHaveLength(1);
    expect(result.datasetOptions.value[0].id).toBe("churn.csv");
  });

  it("does not expose loadManifest to the component", () => {
    const [result] = withSetup(() =>
      useTrainingScreenOrchestrator({
        baseUrl: "/api/ai",
        framework: "pytorch",
        defaultTrainingMode: "linear_glm_baseline",
        datasetApi,
        trainingApi,
      })
    );

    expect(result).not.toHaveProperty("loadManifest");
  });

  it("exposes training form state", () => {
    const [result] = withSetup(() =>
      useTrainingScreenOrchestrator({
        baseUrl: "/api/ai",
        framework: "tensorflow",
        defaultTrainingMode: "wide_and_deep",
        datasetApi,
        trainingApi,
      })
    );

    expect(result.trainingMode.value).toBe("wide_and_deep");
    expect(result.epochValues.value).toBe("60");
  });
});
