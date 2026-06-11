import { describe, it, expect, vi, beforeEach } from "vitest";
import { useTrainingOrchestrator, formatMetric } from "~/features/ml/model";
import type { MlTrainingApi } from "~/features/ml/api";

vi.mock("~/features/ml/api", async (importOriginal) => {
  const orig = await importOriginal<typeof import("~/features/ml/api")>();
  return {
    ...orig,
    createMlTrainingApi: vi.fn(() => ({
      trainPytorch: vi.fn().mockResolvedValue({ status: "ok", metrics: { test_metric: 0.7 } }),
      trainTensorflow: vi.fn().mockResolvedValue({ status: "ok", metrics: { test_metric: 0.7 } }),
    })),
  };
});

function createMockApi(overrides: Partial<MlTrainingApi> = {}): MlTrainingApi {
  return {
    trainPytorch: vi.fn().mockResolvedValue({
      status: "ok",
      metrics: { test_metric: 0.95 },
      model_id: "model-123",
    }),
    trainTensorflow: vi.fn().mockResolvedValue({
      status: "ok",
      metrics: { test_metric: 0.92 },
      model_id: "model-456",
    }),
    ...overrides,
  };
}

describe("useTrainingOrchestrator", () => {
  let api: MlTrainingApi;
  let orchestrator: ReturnType<typeof useTrainingOrchestrator>;

  beforeEach(() => {
    api = createMockApi();
    orchestrator = useTrainingOrchestrator({
      baseUrl: "/api/ai",
      framework: "pytorch",
      api,
    });
  });

  describe("buildCombos()", () => {
    it("builds cartesian product of hyperparams", () => {
      const combos = orchestrator.buildCombos({
        epochs: [10, 20],
        batches: [32, 64],
        learningRates: [0.001],
        testSizes: [0.2],
        hiddenDims: [128],
        numHiddenLayers: [2],
        dropouts: [0.1],
      });

      expect(combos).toHaveLength(4);
      expect(combos[0]).toEqual({
        epochs: 10,
        batch_size: 32,
        learning_rate: 0.001,
        test_size: 0.2,
        hidden_dim: 128,
        num_hidden_layers: 2,
        dropout: 0.1,
      });
    });

    it("uses defaults for empty hidden/layer/dropout arrays", () => {
      const combos = orchestrator.buildCombos({
        epochs: [10],
        batches: [32],
        learningRates: [0.001],
        testSizes: [0.2],
        hiddenDims: [],
        numHiddenLayers: [],
        dropouts: [],
      });

      expect(combos[0].hidden_dim).toBe(128);
      expect(combos[0].num_hidden_layers).toBe(2);
      expect(combos[0].dropout).toBe(0.1);
    });
  });

  describe("train()", () => {
    it("runs all combos and populates trainingRuns", async () => {
      const combos = orchestrator.buildCombos({
        epochs: [10, 20],
        batches: [32],
        learningRates: [0.001],
        testSizes: [0.2],
        hiddenDims: [128],
        numHiddenLayers: [2],
        dropouts: [0.1],
      });

      await orchestrator.train({
        datasetId: "churn.csv",
        targetColumn: "Churn",
        trainingMode: "mlp_dense",
        task: "classification",
        excludeColumns: ["customerID"],
        dateColumns: [],
        combos,
      });

      expect(orchestrator.trainingRuns.value).toHaveLength(2);
      expect(orchestrator.trainingRuns.value[0].status).toBe("completed");
      expect(orchestrator.trainingRuns.value[0].metric).toBe(0.95);
      expect(api.trainPytorch).toHaveBeenCalledTimes(2);
    });

    it("uses trainTensorflow for tensorflow framework", async () => {
      const tfOrch = useTrainingOrchestrator({
        baseUrl: "/api/ai",
        framework: "tensorflow",
        api,
      });
      const combos = tfOrch.buildCombos({
        epochs: [10],
        batches: [32],
        learningRates: [0.001],
        testSizes: [0.2],
        hiddenDims: [128],
        numHiddenLayers: [2],
        dropouts: [0.1],
      });

      await tfOrch.train({
        datasetId: "churn.csv",
        targetColumn: "Churn",
        trainingMode: "wide_and_deep",
        task: "classification",
        excludeColumns: [],
        dateColumns: [],
        combos,
      });

      expect(api.trainTensorflow).toHaveBeenCalledTimes(1);
      expect(api.trainPytorch).not.toHaveBeenCalled();
    });

    it("handles failed training response", async () => {
      const failApi = createMockApi({
        trainPytorch: vi.fn().mockResolvedValue({
          status: "error",
          error: "Out of memory",
        }),
      });
      const orch = useTrainingOrchestrator({
        baseUrl: "/api/ai",
        framework: "pytorch",
        api: failApi,
      });

      await orch.train({
        datasetId: "churn.csv",
        targetColumn: "Churn",
        trainingMode: "mlp_dense",
        task: "auto",
        excludeColumns: [],
        dateColumns: [],
        combos: [{ epochs: 10, batch_size: 32, learning_rate: 0.001, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1 }],
      });

      expect(orch.trainingRuns.value[0].status).toBe("failed");
      expect(orch.trainingRuns.value[0].error).toBe("Out of memory");
    });

    it("handles network error", async () => {
      const failApi = createMockApi({
        trainPytorch: vi.fn().mockRejectedValue(new Error("ECONNREFUSED")),
      });
      const orch = useTrainingOrchestrator({
        baseUrl: "/api/ai",
        framework: "pytorch",
        api: failApi,
      });

      await orch.train({
        datasetId: "churn.csv",
        targetColumn: "Churn",
        trainingMode: "mlp_dense",
        task: "auto",
        excludeColumns: [],
        dateColumns: [],
        combos: [{ epochs: 10, batch_size: 32, learning_rate: 0.001, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1 }],
      });

      expect(orch.trainingRuns.value[0].status).toBe("failed");
      expect(orch.trainingRuns.value[0].error).toBe("ECONNREFUSED");
    });

    it("handles non-Error throw", async () => {
      const failApi = createMockApi({
        trainPytorch: vi.fn().mockRejectedValue(42),
      });
      const orch = useTrainingOrchestrator({
        baseUrl: "/api/ai",
        framework: "pytorch",
        api: failApi,
      });

      await orch.train({
        datasetId: "d",
        targetColumn: "t",
        trainingMode: "m",
        task: "auto",
        excludeColumns: [],
        dateColumns: [],
        combos: [{ epochs: 1, batch_size: 1, learning_rate: 0.1, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1 }],
      });

      expect(orch.trainingRuns.value[0].error).toBe("Request failed");
    });

    it("updates progress during training", async () => {
      const progressValues: number[] = [];
      const orch = useTrainingOrchestrator({
        baseUrl: "/api/ai",
        framework: "pytorch",
        api: createMockApi({
          trainPytorch: vi.fn().mockImplementation(async () => {
            progressValues.push(orch.trainingProgress.value.current);
            return { status: "ok", metrics: { test_metric: 0.9 } };
          }),
        }),
      });

      await orch.train({
        datasetId: "d",
        targetColumn: "t",
        trainingMode: "m",
        task: "auto",
        excludeColumns: [],
        dateColumns: [],
        combos: [
          { epochs: 10, batch_size: 32, learning_rate: 0.001, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1 },
          { epochs: 20, batch_size: 32, learning_rate: 0.001, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1 },
        ],
      });

      expect(progressValues).toEqual([1, 2]);
    });

    it("resets state at start of training", async () => {
      orchestrator.trainingRuns.value = [{ epochs: 1, batch_size: 1, learning_rate: 0.1, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1, status: "old", metric: 0.5 }];
      orchestrator.trainingError.value = "stale";

      await orchestrator.train({
        datasetId: "d",
        targetColumn: "t",
        trainingMode: "m",
        task: "auto",
        excludeColumns: [],
        dateColumns: [],
        combos: [{ epochs: 10, batch_size: 32, learning_rate: 0.001, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1 }],
      });

      expect(orchestrator.trainingRuns.value).toHaveLength(1);
      expect(orchestrator.trainingError.value).toBeNull();
    });

    it("sets isTraining false after completion", async () => {
      await orchestrator.train({
        datasetId: "d",
        targetColumn: "t",
        trainingMode: "m",
        task: "auto",
        excludeColumns: [],
        dateColumns: [],
        combos: [{ epochs: 10, batch_size: 32, learning_rate: 0.001, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1 }],
      });

      expect(orchestrator.isTraining.value).toBe(false);
    });

    it("uses accuracy as fallback metric", async () => {
      const accApi = createMockApi({
        trainPytorch: vi.fn().mockResolvedValue({
          status: "ok",
          metrics: { accuracy: 0.88 },
        }),
      });
      const orch = useTrainingOrchestrator({
        baseUrl: "/api/ai",
        framework: "pytorch",
        api: accApi,
      });

      await orch.train({
        datasetId: "d",
        targetColumn: "t",
        trainingMode: "m",
        task: "auto",
        excludeColumns: [],
        dateColumns: [],
        combos: [{ epochs: 10, batch_size: 32, learning_rate: 0.001, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1 }],
      });

      expect(orch.trainingRuns.value[0].metric).toBe(0.88);
    });
  });

  describe("stop()", () => {
    it("stops training loop", async () => {
      let callCount = 0;
      const slowApi = createMockApi({
        trainPytorch: vi.fn().mockImplementation(async () => {
          callCount++;
          if (callCount === 1) orchestrator.stop();
          return { status: "ok", metrics: { test_metric: 0.9 } };
        }),
      });

      orchestrator = useTrainingOrchestrator({
        baseUrl: "/api/ai",
        framework: "pytorch",
        api: slowApi,
      });

      await orchestrator.train({
        datasetId: "d",
        targetColumn: "t",
        trainingMode: "m",
        task: "auto",
        excludeColumns: [],
        dateColumns: [],
        combos: [
          { epochs: 10, batch_size: 32, learning_rate: 0.001, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1 },
          { epochs: 20, batch_size: 32, learning_rate: 0.001, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1 },
          { epochs: 30, batch_size: 32, learning_rate: 0.001, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1 },
        ],
      });

      expect(orchestrator.trainingRuns.value).toHaveLength(1);
    });
  });

  describe("clearRuns()", () => {
    it("empties training runs", () => {
      orchestrator.trainingRuns.value = [{ epochs: 1, batch_size: 1, learning_rate: 0.1, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1, status: "completed", metric: 0.9 }];
      orchestrator.clearRuns();
      expect(orchestrator.trainingRuns.value).toHaveLength(0);
    });
  });

  describe("api fallback", () => {
    it("uses createMlTrainingApi when no api option provided", async () => {
      const orch = useTrainingOrchestrator({
        baseUrl: "/api/ai",
        framework: "pytorch",
      });
      await orch.train({
        datasetId: "d",
        targetColumn: "t",
        trainingMode: "m",
        task: "auto",
        excludeColumns: [],
        dateColumns: [],
        combos: [{ epochs: 1, batch_size: 1, learning_rate: 0.1, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1 }],
      });
      expect(orch.trainingRuns.value[0].status).toBe("completed");
    });
  });

  describe("copyResults()", () => {
    it("returns JSON string of runs", () => {
      orchestrator.trainingRuns.value = [{ epochs: 10, batch_size: 32, learning_rate: 0.001, test_size: 0.2, hidden_dim: 128, num_hidden_layers: 2, dropout: 0.1, status: "completed", metric: 0.95 }];
      const json = orchestrator.copyResults();
      const parsed = JSON.parse(json);
      expect(parsed).toHaveLength(1);
      expect(parsed[0].metric).toBe(0.95);
    });
  });
});

describe("formatMetric", () => {
  it("formats numbers to 4 decimals", () => {
    expect(formatMetric(0.9512345)).toBe("0.9512");
  });

  it("formats zero", () => {
    expect(formatMetric(0)).toBe("0.0000");
  });

  it("returns dash for null", () => {
    expect(formatMetric(null)).toBe("-");
  });

  it("returns dash for undefined", () => {
    expect(formatMetric(undefined)).toBe("-");
  });

  it("stringifies non-number values", () => {
    expect(formatMetric("hello")).toBe("hello");
  });
});
