import { describe, expect, it, vi } from "vitest";
import {
  runPytorchDistillation,
  runPytorchTraining,
} from "@/features/ml/typescript/react/orchestrators/pytorchTraining.orchestrator";

describe("pytorchTraining.orchestrator", () => {
  const baseProblem = {
    datasetId: "d1.csv",
    targetColumn: "target",
    task: "classification" as const,
    trainingMode: "mlp_dense" as const,
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

  it("runs training combinations and records completed + failed runs", async () => {
    const prependTrainingRun = vi.fn();
    const onProgress = vi.fn();
    const trainModel = vi
      .fn()
      .mockResolvedValueOnce({
        status: "ok",
        run_id: "run-1",
        model_id: "model-1",
        model_path: "/tmp/model-1",
        metrics: { test_metric_name: "accuracy", test_metric_value: 0.91 },
      })
      .mockResolvedValueOnce({
        status: "error",
        error: "boom",
      });

    const result = await runPytorchTraining(
      {
        datasetId: "d1.csv",
        targetColumn: "target",
        task: "classification",
        trainingMode: "mlp_dense",
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
          {
            epochs: 80,
            testSize: 0.25,
            learningRate: 0.002,
            batchSize: 32,
            hiddenDim: 64,
            numHiddenLayers: 3,
            dropout: 0.2,
          },
        ],
      },
      {
        trainModel,
        prependTrainingRun,
        onProgress,
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
      }
    );

    expect(result.stopped).toBe(false);
    expect(result.completed).toBe(2);
    expect(result.failedRuns).toBe(1);
    expect(prependTrainingRun).toHaveBeenCalledTimes(2);
    expect(onProgress).toHaveBeenNthCalledWith(1, 1, 2);
  });

  it("uses default training failure message when backend error is absent", async () => {
    const prependTrainingRun = vi.fn();
    const result = await runPytorchTraining(
      {
        ...baseProblem,
        combinations: [...baseProblem.combinations, { ...baseProblem.combinations[0], epochs: 80 }],
      },
      {
        trainModel: vi
          .fn()
          .mockResolvedValueOnce({ status: "error" })
          .mockResolvedValueOnce({ status: "error" }),
        prependTrainingRun,
        onProgress: vi.fn(),
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
      }
    );

    expect(result.failedRuns).toBe(2);
    expect(result.firstFailureMessage).toBe("Training failed.");
    expect(prependTrainingRun).toHaveBeenCalledWith(
      expect.objectContaining({
        error: "Training failed.",
      })
    );
  });

  it("maps baseline mode fields to n/a and captures completed teacher runs", async () => {
    const prependTrainingRun = vi.fn();
    const result = await runPytorchTraining(
      {
        ...baseProblem,
        isLinearBaselineMode: true,
      },
      {
        trainModel: vi.fn(async () => ({
          status: "ok",
          metrics: {},
        })),
        prependTrainingRun,
        onProgress: vi.fn(),
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
      }
    );

    expect(result.completedTeacherRuns).toHaveLength(1);
    expect(prependTrainingRun).toHaveBeenCalledWith(
      expect.objectContaining({
        hidden_dim: "n/a",
        num_hidden_layers: "n/a",
        dropout: "n/a",
        metric_name: "n/a",
        model_id: "n/a",
      })
    );
  });

  it("maps failed baseline-mode runs with n/a architecture fields", async () => {
    const prependTrainingRun = vi.fn();
    await runPytorchTraining(
      {
        ...baseProblem,
        isLinearBaselineMode: true,
      },
      {
        trainModel: vi.fn(async () => ({
          status: "error",
          error: "baseline failed",
        })),
        prependTrainingRun,
        onProgress: vi.fn(),
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
      }
    );

    expect(prependTrainingRun).toHaveBeenCalledWith(
      expect.objectContaining({
        hidden_dim: "n/a",
        num_hidden_layers: "n/a",
        dropout: "n/a",
        error: "baseline failed",
      })
    );
  });

  it("stops early when shouldContinue switches false", async () => {
    const prependTrainingRun = vi.fn();
    let calls = 0;
    const result = await runPytorchTraining(
      {
        ...baseProblem,
        combinations: [
          baseProblem.combinations[0],
          { ...baseProblem.combinations[0], epochs: 70 },
        ],
      },
      {
        trainModel: vi.fn(async () => ({ status: "ok" })),
        prependTrainingRun,
        onProgress: vi.fn(),
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
        shouldContinue: () => {
          calls += 1;
          return calls === 1;
        },
      }
    );

    expect(result.stopped).toBe(true);
    expect(result.completed).toBe(1);
    expect(prependTrainingRun).toHaveBeenCalledTimes(1);
  });

  it("returns error status when distillation fails", async () => {
    const result = await runPytorchDistillation(
      {
        datasetId: "d1.csv",
        targetColumn: "target",
        task: "classification",
        trainingMode: "mlp_dense",
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
        distillModel: vi.fn(async () => ({ status: "error", error: "distill failed" })),
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
      }
    );

    expect(result).toEqual({ status: "error", error: "distill failed" });
  });

  it("returns default distillation error when backend omits error", async () => {
    const result = await runPytorchDistillation(
      {
        datasetId: "d1.csv",
        targetColumn: "target",
        task: "classification",
        trainingMode: "mlp_dense",
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
        distillModel: vi.fn(async () => ({ status: "error" })),
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
      }
    );

    expect(result).toEqual({ status: "error", error: "Distillation failed." });
  });

  it("builds successful distillation output with clamped student params", async () => {
    const distillModel = vi.fn(async () => ({
      status: "ok",
      run_id: "distill-1",
      model_id: "student-1",
      model_path: "/tmp/student-1",
      metrics: { test_metric_name: "auc", test_metric_value: 0.92 },
      teacher_input_dim: 10,
      teacher_output_dim: 2,
      student_input_dim: 10,
      student_output_dim: 2,
      teacher_model_size_bytes: 2000,
      student_model_size_bytes: 900,
      size_saved_bytes: 1100,
      size_saved_percent: 55,
      teacher_param_count: 10000,
      student_param_count: 4800,
      param_saved_count: 5200,
      param_saved_percent: 52,
    }));

    const result = await runPytorchDistillation(
      {
        datasetId: "d1.csv",
        targetColumn: "target",
        task: "classification",
        trainingMode: "mlp_dense",
        saveDistilledModel: false,
        excludeColumns: [],
        dateColumns: [],
        teacher: {
          hidden: 20,
          layers: 20,
          dropout: 0.9,
          epochs: 20,
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
        epochs: 30,
        batch_size: 1,
        student_hidden_dim: 16,
        student_num_hidden_layers: 15,
        student_dropout: 0.5,
      })
    );
    expect(result.status).toBe("ok");
    if (result.status === "ok") {
      expect(result.distilledRun.metric_name).toBe("auc");
      expect(result.teacherParamCount).toBe(10000);
      expect(result.paramSavedPercent).toBe(52);
    }
  });

  it("fills null/default distillation output fields when backend omits optionals", async () => {
    const result = await runPytorchDistillation(
      {
        datasetId: "d1.csv",
        targetColumn: "target",
        task: "classification",
        trainingMode: "mlp_dense",
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
          metrics: {},
        })),
        formatCompletedAt: () => "01/01/26 00:00:00",
        formatMetricNumber: ({ value }) => String(value ?? "n/a"),
      }
    );

    expect(result.status).toBe("ok");
    if (result.status !== "ok") return;
    expect(result.modelId).toBeNull();
    expect(result.modelPath).toBeNull();
    expect(result.runId).toBeNull();
    expect(result.teacherModelSizeBytes).toBeNull();
    expect(result.studentModelSizeBytes).toBeNull();
    expect(result.distilledRun.metric_name).toBe("n/a");
    expect(result.distilledRun.metric_score).toBe("n/a");
    expect(result.distilledRun.model_id).toBe("n/a");
    expect(result.distilledRun.model_path).toBe("n/a");
    expect(result.distilledRun.run_id).toBe("n/a");
  });
});
