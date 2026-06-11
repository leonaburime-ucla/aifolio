import { describe, expect, it, vi } from "vitest";
import {
  runPytorchDistillation,
  runPytorchTraining,
} from "../../../src/ml-training";
import type {
  RunPytorchDistillationProblem,
  RunPytorchTrainingProblem,
} from "@aifolio/contracts/entities/ml-training";

const formatCompletedAt = () => "2026-06-10T00:00:00.000Z";
const formatMetricNumber = ({ value }: { value: unknown }) =>
  typeof value === "number" ? String(value) : "n/a";

describe("ML training orchestrator core", () => {
  it("runs PyTorch training combinations and records completed teacher rows", async () => {
    const problem: RunPytorchTrainingProblem = {
      datasetId: "d1.csv",
      targetColumn: "target",
      task: "classification",
      trainingMode: "mlp_dense",
      isLinearBaselineMode: false,
      excludeColumns: [],
      dateColumns: [],
      combinations: [
        {
          epochs: 10,
          testSize: 0.2,
          learningRate: 0.001,
          batchSize: 32,
          hiddenDim: 64,
          numHiddenLayers: 2,
          dropout: 0.1,
        },
      ],
    };
    const prependTrainingRun = vi.fn();

    const result = await runPytorchTraining(problem, {
      trainModel: vi.fn().mockResolvedValue({
        status: "ok",
        run_id: "run-1",
        model_id: "model-1",
        model_path: "/models/model-1",
        metrics: {
          test_metric_name: "accuracy",
          test_metric_value: 0.91,
          train_loss: 0.2,
          test_loss: 0.3,
        },
      }),
      prependTrainingRun,
      onProgress: vi.fn(),
      formatCompletedAt,
      formatMetricNumber,
    });

    expect(result).toMatchObject({
      stopped: false,
      completed: 1,
      total: 1,
      failedRuns: 0,
    });
    expect(result.completedTeacherRuns).toHaveLength(1);
    expect(prependTrainingRun).toHaveBeenCalledWith(
      expect.objectContaining({
        result: "completed",
        run_id: "run-1",
        metric_name: "accuracy",
        metric_score: "0.91",
      })
    );
  });

  it("stops a training sweep before the next model call when cancellation is requested", async () => {
    const problem: RunPytorchTrainingProblem = {
      datasetId: "d1.csv",
      targetColumn: "target",
      task: "classification",
      trainingMode: "mlp_dense",
      isLinearBaselineMode: false,
      excludeColumns: [],
      dateColumns: [],
      combinations: [
        {
          epochs: 10,
          testSize: 0.2,
          learningRate: 0.001,
          batchSize: 32,
          hiddenDim: 64,
          numHiddenLayers: 2,
          dropout: 0.1,
        },
      ],
    };
    const trainModel = vi.fn();

    const result = await runPytorchTraining(problem, {
      trainModel,
      prependTrainingRun: vi.fn(),
      onProgress: vi.fn(),
      shouldContinue: vi.fn(() => false),
      formatCompletedAt,
      formatMetricNumber,
    });

    expect(result).toEqual({
      stopped: true,
      completed: 0,
      total: 1,
      completedTeacherRuns: [],
      failedRuns: 0,
      firstFailureMessage: null,
    });
    expect(trainModel).not.toHaveBeenCalled();
  });

  it("records linear-baseline failed runs with default error fallbacks", async () => {
    const problem: RunPytorchTrainingProblem = {
      datasetId: "d1.csv",
      targetColumn: "target",
      task: "classification",
      trainingMode: "linear_glm_baseline",
      isLinearBaselineMode: true,
      excludeColumns: ["id"],
      dateColumns: ["created_at"],
      combinations: [
        {
          epochs: 10,
          testSize: 0.2,
          learningRate: 0.001,
          batchSize: 32,
          hiddenDim: 64,
          numHiddenLayers: 3,
          dropout: 0.3,
        },
      ],
    };
    const trainModel = vi.fn().mockResolvedValue({ status: "error" });
    const prependTrainingRun = vi.fn();

    const result = await runPytorchTraining(problem, {
      trainModel,
      prependTrainingRun,
      onProgress: vi.fn(),
      formatCompletedAt,
      formatMetricNumber,
    });

    expect(trainModel).toHaveBeenCalledWith(
      expect.objectContaining({
        hidden_dim: 128,
        num_hidden_layers: 2,
        dropout: 0.1,
      })
    );
    expect(result).toMatchObject({
      stopped: false,
      completed: 1,
      failedRuns: 1,
      firstFailureMessage: "Training failed.",
    });
    expect(prependTrainingRun).toHaveBeenCalledWith(
      expect.objectContaining({
        result: "failed",
        hidden_dim: "n/a",
        num_hidden_layers: "n/a",
        dropout: "n/a",
        error: "Training failed.",
      })
    );
  });

  it("records completed runs with metric and model fallbacks when API fields are omitted", async () => {
    const problem: RunPytorchTrainingProblem = {
      datasetId: "d1.csv",
      targetColumn: "target",
      task: "classification",
      trainingMode: "mlp_dense",
      isLinearBaselineMode: false,
      excludeColumns: [],
      dateColumns: [],
      combinations: [
        {
          epochs: 10,
          testSize: 0.2,
          learningRate: 0.001,
          batchSize: 32,
          hiddenDim: 64,
          numHiddenLayers: 2,
          dropout: 0.1,
        },
      ],
    };
    const prependTrainingRun = vi.fn();

    const result = await runPytorchTraining(problem, {
      trainModel: vi.fn().mockResolvedValue({ status: "ok", metrics: {} }),
      prependTrainingRun,
      onProgress: vi.fn(),
      formatCompletedAt,
      formatMetricNumber,
    });

    expect(result.completedTeacherRuns).toHaveLength(1);
    expect(prependTrainingRun).toHaveBeenCalledWith(
      expect.objectContaining({
        result: "completed",
        metric_name: "n/a",
        metric_score: "n/a",
        train_loss: "n/a",
        test_loss: "n/a",
        model_id: "n/a",
        model_path: "n/a",
        run_id: "n/a",
      })
    );
  });

  it("records linear-baseline completed rows with n/a deep-model dimensions", async () => {
    const problem: RunPytorchTrainingProblem = {
      datasetId: "d1.csv",
      targetColumn: "target",
      task: "classification",
      trainingMode: "linear_glm_baseline",
      isLinearBaselineMode: true,
      excludeColumns: [],
      dateColumns: [],
      combinations: [
        {
          epochs: 10,
          testSize: 0.2,
          learningRate: 0.001,
          batchSize: 32,
          hiddenDim: 64,
          numHiddenLayers: 2,
          dropout: 0.1,
        },
      ],
    };
    const prependTrainingRun = vi.fn();

    await runPytorchTraining(problem, {
      trainModel: vi.fn().mockResolvedValue({
        status: "ok",
        metrics: { test_metric_name: "accuracy", test_metric_value: 0.91 },
      }),
      prependTrainingRun,
      onProgress: vi.fn(),
      formatCompletedAt,
      formatMetricNumber,
    });

    expect(prependTrainingRun).toHaveBeenCalledWith(
      expect.objectContaining({
        result: "completed",
        hidden_dim: "n/a",
        num_hidden_layers: "n/a",
        dropout: "n/a",
      })
    );
  });

  it("derives bounded PyTorch student distillation payload fields", async () => {
    const problem: RunPytorchDistillationProblem = {
      datasetId: "d1.csv",
      targetColumn: "target",
      task: "classification",
      trainingMode: "tabresnet",
      saveDistilledModel: true,
      excludeColumns: ["id"],
      dateColumns: [],
      teacher: {
        hidden: 64,
        layers: 2,
        dropout: 0.1,
        epochs: 12,
        batch: 31.6,
        learningRate: 0.001,
        testSize: 0.2,
        runId: "teacher-run",
      },
    };
    const distillModel = vi.fn().mockResolvedValue({
      status: "ok",
      run_id: "student-run",
      model_id: "student-model",
      model_path: "/models/student",
      metrics: { test_metric_name: "accuracy", test_metric_value: 0.89 },
    });

    const result = await runPytorchDistillation(problem, {
      distillModel,
      formatCompletedAt,
      formatMetricNumber,
    });

    const distillPayload = distillModel.mock.calls[0][0];
    expect(distillPayload).toMatchObject({
      teacher_run_id: "teacher-run",
      epochs: 30,
      batch_size: 32,
      student_hidden_dim: 32,
      student_num_hidden_layers: 1,
    });
    expect(distillPayload.student_dropout).toBeCloseTo(0.15);
    expect(result).toMatchObject({
      status: "ok",
      runId: "student-run",
      modelId: "student-model",
    });
  });

  it("returns a default distillation error when the adapter omits its error message", async () => {
    const problem: RunPytorchDistillationProblem = {
      datasetId: "d1.csv",
      targetColumn: "target",
      task: "classification",
      trainingMode: "tabresnet",
      saveDistilledModel: true,
      excludeColumns: [],
      dateColumns: [],
      teacher: {
        hidden: 20,
        layers: 1,
        dropout: 0.49,
        epochs: 12,
        batch: 0,
        learningRate: 0.001,
        testSize: 0.2,
        runId: "teacher-run",
      },
    };

    const result = await runPytorchDistillation(problem, {
      distillModel: vi.fn().mockResolvedValue({ status: "error" }),
      formatCompletedAt,
      formatMetricNumber,
    });

    expect(result).toEqual({
      status: "error",
      error: "Distillation failed.",
    });
  });

  it("returns null metadata fallbacks for successful distillation with omitted optional fields", async () => {
    const problem: RunPytorchDistillationProblem = {
      datasetId: "d1.csv",
      targetColumn: "target",
      task: "classification",
      trainingMode: "tabresnet",
      saveDistilledModel: false,
      excludeColumns: [],
      dateColumns: [],
      teacher: {
        hidden: 20,
        layers: 20,
        dropout: 0.49,
        epochs: 12,
        batch: 0,
        learningRate: 0.001,
        testSize: 0.2,
        runId: "teacher-run",
      },
    };
    const distillModel = vi.fn().mockResolvedValue({
      status: "ok",
      metrics: {},
    });

    const result = await runPytorchDistillation(problem, {
      distillModel,
      formatCompletedAt,
      formatMetricNumber,
    });

    expect(distillModel).toHaveBeenCalledWith(
      expect.objectContaining({
        batch_size: 1,
        student_hidden_dim: 16,
        student_num_hidden_layers: 15,
        student_dropout: 0.5,
      })
    );
    expect(result).toMatchObject({
      status: "ok",
      modelId: null,
      modelPath: null,
      runId: null,
      teacherModelSizeBytes: null,
      studentModelSizeBytes: null,
      teacherParamCount: null,
      studentParamCount: null,
    });
    if (result.status === "ok") {
      expect(result.distilledRun).toEqual(
        expect.objectContaining({
          metric_name: "n/a",
          metric_score: "n/a",
          model_id: "n/a",
          model_path: "n/a",
          run_id: "n/a",
        })
      );
    }
  });
});
