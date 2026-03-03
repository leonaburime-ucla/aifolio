import { describe, expect, it, vi } from "vitest";
import {
  runTensorflowDistillation,
  runTensorflowTraining,
} from "@/features/ml/typescript/react/orchestrators/tensorflowTraining.orchestrator";

describe("tensorflowTraining.orchestrator", () => {
  const baseProblem = {
    datasetId: "d1.csv",
    targetColumn: "target",
    task: "classification" as const,
    trainingMode: "wide_and_deep" as const,
    isLinearBaselineMode: false,
    excludeColumns: [],
    dateColumns: [],
    combinations: [
      {
        epochs: 60,
        testSize: 0.2,
        learningRate: 0.001,
        batchSize: 64,
        hiddenDim: 128,
        numHiddenLayers: 2,
        dropout: 0.1,
      },
    ],
  };

  it("stops early when shouldContinue is false", async () => {
    const result = await runTensorflowTraining(
      {
        datasetId: "d1.csv",
        targetColumn: "target",
        task: "classification",
        trainingMode: "wide_and_deep",
        isLinearBaselineMode: false,
        excludeColumns: [],
        dateColumns: [],
        combinations: [
          {
            epochs: 60,
            testSize: 0.2,
            learningRate: 0.001,
            batchSize: 64,
            hiddenDim: 128,
            numHiddenLayers: 2,
            dropout: 0.1,
          },
        ],
      },
      {
        trainModel: vi.fn(async () => ({ status: "ok" })),
        prependTrainingRun: vi.fn(),
        onProgress: vi.fn(),
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
        shouldContinue: () => false,
      }
    );

    expect(result.stopped).toBe(true);
    expect(result.completed).toBe(0);
  });

  it("records failed and completed tensorflow runs", async () => {
    const prependTrainingRun = vi.fn();
    const onProgress = vi.fn();
    const result = await runTensorflowTraining(
      {
        ...baseProblem,
        combinations: [
          baseProblem.combinations[0],
          { ...baseProblem.combinations[0], epochs: 80 },
        ],
      },
      {
        trainModel: vi
          .fn()
          .mockResolvedValueOnce({
            status: "error",
            error: "tf fail",
          })
          .mockResolvedValueOnce({
            status: "ok",
            run_id: "run-ok",
            metrics: { test_metric_name: "accuracy", test_metric_value: 0.87 },
          }),
        prependTrainingRun,
        onProgress,
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
      }
    );

    expect(result.failedRuns).toBe(1);
    expect(result.completedTeacherRuns).toHaveLength(1);
    expect(result.firstFailureMessage).toBe("tf fail");
    expect(onProgress).toHaveBeenNthCalledWith(2, 2, 2);
    expect(prependTrainingRun).toHaveBeenCalledTimes(2);
  });

  it("uses default failure text and n/a baseline values", async () => {
    const prependTrainingRun = vi.fn();
    const result = await runTensorflowTraining(
      {
        ...baseProblem,
        isLinearBaselineMode: true,
      },
      {
        trainModel: vi.fn(async () => ({ status: "error" })),
        prependTrainingRun,
        onProgress: vi.fn(),
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
      }
    );

    expect(result.firstFailureMessage).toBe("Training failed.");
    expect(prependTrainingRun).toHaveBeenCalledWith(
      expect.objectContaining({
        hidden_dim: "n/a",
        num_hidden_layers: "n/a",
        dropout: "n/a",
        error: "Training failed.",
      })
    );
  });

  it("builds successful distillation output", async () => {
    const result = await runTensorflowDistillation(
      {
        datasetId: "d1.csv",
        targetColumn: "target",
        task: "classification",
        trainingMode: "wide_and_deep",
        saveDistilledModel: false,
        excludeColumns: [],
        dateColumns: [],
        teacher: {
          hidden: 128,
          layers: 2,
          dropout: 0.1,
          epochs: 60,
          batch: 64,
          learningRate: 0.001,
          testSize: 0.2,
        },
      },
      {
        distillModel: vi.fn(async () => ({
          status: "ok",
          run_id: "distill-1",
          model_id: "student-1",
          model_path: "/tmp/student-1",
          metrics: { test_metric_name: "accuracy", test_metric_value: 0.88 },
        })),
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
      }
    );

    expect(result.status).toBe("ok");
    if (result.status === "ok") {
      expect(result.runId).toBe("distill-1");
      expect(result.distilledRun.result).toBe("distilled");
    }
  });

  it("returns default distillation error and clamps student config", async () => {
    const err = await runTensorflowDistillation(
      {
        datasetId: "d1.csv",
        targetColumn: "target",
        task: "classification",
        trainingMode: "wide_and_deep",
        saveDistilledModel: false,
        excludeColumns: [],
        dateColumns: [],
        teacher: {
          hidden: 10,
          layers: 25,
          dropout: 0.9,
          epochs: 5,
          batch: 0.4,
          learningRate: 0.001,
          testSize: 0.2,
        },
      },
      {
        distillModel: vi.fn(async () => ({ status: "error" })),
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
      }
    );
    expect(err).toEqual({ status: "error", error: "Distillation failed." });

    const distillModel = vi.fn(async () => ({ status: "ok", metrics: {} }));
    const ok = await runTensorflowDistillation(
      {
        datasetId: "d1.csv",
        targetColumn: "target",
        task: "classification",
        trainingMode: "wide_and_deep",
        saveDistilledModel: false,
        excludeColumns: [],
        dateColumns: [],
        teacher: {
          hidden: 10,
          layers: 25,
          dropout: 0.9,
          epochs: 5,
          batch: 0.4,
          learningRate: 0.001,
          testSize: 0.2,
        },
      },
      {
        distillModel,
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
      }
    );
    expect(distillModel).toHaveBeenCalledWith(
      expect.objectContaining({
        epochs: 8,
        batch_size: 1,
        student_hidden_dim: 16,
        student_num_hidden_layers: 15,
        student_dropout: 0.5,
      })
    );
    expect(ok.status).toBe("ok");
    if (ok.status === "ok") {
      expect(ok.distilledRun.metric_name).toBe("n/a");
      expect(ok.distilledRun.model_id).toBe("n/a");
    }
  });
});
