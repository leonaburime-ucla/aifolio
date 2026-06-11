import { describe, expect, it, vi } from "vitest";
import {
  runTensorflowDistillation,
  runTensorflowTraining,
} from "../../../src/ml-training";
import type {
  RunTensorflowDistillationProblem,
  RunTensorflowTrainingProblem,
} from "@aifolio/contracts/entities/ml-training";

const formatCompletedAt = () => "2026-06-10T00:00:00.000Z";
const formatMetricNumber = ({ value }: { value: unknown }) =>
  typeof value === "number" ? String(value) : "n/a";

describe("runTensorflowTraining", () => {
  it("runs TensorFlow training combinations and records completed rows", async () => {
    const problem: RunTensorflowTrainingProblem = {
      datasetId: "d1.csv",
      targetColumn: "price",
      task: "regression",
      trainingMode: "wide_and_deep",
      isLinearBaselineMode: false,
      excludeColumns: [],
      dateColumns: [],
      combinations: [
        {
          epochs: 15,
          testSize: 0.25,
          learningRate: 0.01,
          batchSize: 64,
          hiddenDim: 128,
          numHiddenLayers: 3,
          dropout: 0.2,
        },
      ],
    };
    const prependTrainingRun = vi.fn();

    const result = await runTensorflowTraining(problem, {
      trainModel: vi.fn().mockResolvedValue({
        status: "ok",
        run_id: "tf-run-1",
        model_id: "tf-model-1",
        model_path: "/models/tf-model-1",
        metrics: {
          test_metric_name: "r2_score",
          test_metric_value: 0.85,
          train_loss: 0.15,
          test_loss: 0.22,
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
        run_id: "tf-run-1",
        metric_name: "r2_score",
        metric_score: "0.85",
      })
    );
  });

  it("records failed runs when trainModel returns error status", async () => {
    const problem: RunTensorflowTrainingProblem = {
      datasetId: "d1.csv",
      targetColumn: "price",
      task: "regression",
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

    const result = await runTensorflowTraining(problem, {
      trainModel: vi.fn().mockResolvedValue({
        status: "error",
        error: "Training diverged",
      }),
      prependTrainingRun,
      onProgress: vi.fn(),
      formatCompletedAt,
      formatMetricNumber,
    });

    expect(result.failedRuns).toBe(1);
    expect(result.completedTeacherRuns).toHaveLength(0);
    expect(prependTrainingRun).toHaveBeenCalledWith(
      expect.objectContaining({ result: "failed" })
    );
  });
});

describe("runTensorflowDistillation", () => {
  it("derives TensorFlow student distillation payload with clamped epochs", async () => {
    const problem: RunTensorflowDistillationProblem = {
      datasetId: "d1.csv",
      targetColumn: "target",
      task: "classification",
      trainingMode: "wide_and_deep",
      saveDistilledModel: true,
      excludeColumns: [],
      dateColumns: [],
      teacher: {
        hidden: 128,
        layers: 3,
        dropout: 0.2,
        epochs: 50,
        batch: 64,
        learningRate: 0.01,
        testSize: 0.25,
        runId: "tf-teacher-run",
      },
    };
    const distillModel = vi.fn().mockResolvedValue({
      status: "ok",
      run_id: "tf-student-run",
      model_id: "tf-student-model",
      model_path: "/models/tf-student",
      metrics: { test_metric_name: "accuracy", test_metric_value: 0.87 },
    });

    const result = await runTensorflowDistillation(problem, {
      distillModel,
      formatCompletedAt,
      formatMetricNumber,
    });

    const payload = distillModel.mock.calls[0][0];
    expect(payload).toMatchObject({
      teacher_run_id: "tf-teacher-run",
      batch_size: 64,
    });
    // TF resolves epochs as min(24, max(8, round(teacherEpochs * 0.4)))
    // 50 * 0.4 = 20 → min(24, max(8, 20)) = 20
    expect(payload.epochs).toBe(20);
    expect(payload.student_hidden_dim).toBe(64); // teacher/2
    expect(payload.student_num_hidden_layers).toBe(2); // teacher-1
    expect(result).toMatchObject({
      status: "ok",
      runId: "tf-student-run",
      modelId: "tf-student-model",
    });
  });

  it("clamps distillation epochs to minimum 8", async () => {
    const problem: RunTensorflowDistillationProblem = {
      datasetId: "d1.csv",
      targetColumn: "target",
      task: "regression",
      trainingMode: "mlp_dense",
      saveDistilledModel: false,
      excludeColumns: [],
      dateColumns: [],
      teacher: {
        hidden: 64,
        layers: 2,
        dropout: 0.1,
        epochs: 10,
        batch: 32,
        learningRate: 0.001,
        testSize: 0.2,
        runId: "tf-teacher-2",
      },
    };
    const distillModel = vi.fn().mockResolvedValue({
      status: "ok",
      run_id: "student-2",
      model_id: "model-2",
      model_path: "/models/2",
      metrics: { test_metric_name: "r2_score", test_metric_value: 0.8 },
    });

    await runTensorflowDistillation(problem, {
      distillModel,
      formatCompletedAt,
      formatMetricNumber,
    });

    const payload = distillModel.mock.calls[0][0];
    // 10 * 0.4 = 4 → min(24, max(8, 4)) = 8
    expect(payload.epochs).toBe(8);
  });

  it("clamps distillation epochs to maximum 24", async () => {
    const problem: RunTensorflowDistillationProblem = {
      datasetId: "d1.csv",
      targetColumn: "target",
      task: "regression",
      trainingMode: "mlp_dense",
      saveDistilledModel: false,
      excludeColumns: [],
      dateColumns: [],
      teacher: {
        hidden: 256,
        layers: 4,
        dropout: 0.3,
        epochs: 100,
        batch: 128,
        learningRate: 0.01,
        testSize: 0.3,
        runId: "tf-teacher-3",
      },
    };
    const distillModel = vi.fn().mockResolvedValue({
      status: "ok",
      run_id: "student-3",
      model_id: "model-3",
      model_path: "/models/3",
      metrics: { test_metric_name: "r2_score", test_metric_value: 0.82 },
    });

    await runTensorflowDistillation(problem, {
      distillModel,
      formatCompletedAt,
      formatMetricNumber,
    });

    const payload = distillModel.mock.calls[0][0];
    // 100 * 0.4 = 40 → min(24, max(8, 40)) = 24
    expect(payload.epochs).toBe(24);
  });
});
