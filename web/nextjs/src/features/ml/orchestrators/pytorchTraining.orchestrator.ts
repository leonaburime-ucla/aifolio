import type { PytorchDistillRequest, PytorchTrainRequest } from "@/features/ml/api/pytorchApi";
import type { TrainingMetrics, TrainingRunRow } from "@/features/ml/utils/trainingRuns.util";

type PytorchTrainCombo = {
  epochs: number;
  batchSize: number;
  learningRate: number;
  testSize: number;
  hiddenDim: number;
  numHiddenLayers: number;
  dropout: number;
};

export type RunPytorchTrainingProblem = {
  datasetId: string;
  targetColumn: string;
  task: "classification" | "regression" | "auto";
  excludeColumns: string[];
  dateColumns: string[];
  combinations: PytorchTrainCombo[];
};

export type RunPytorchTrainingDeps = {
  trainModel: (payload: PytorchTrainRequest) => Promise<{
    status: "ok" | "error";
    model_id?: string;
    model_path?: string;
    metrics?: unknown;
    error?: string;
  }>;
  prependTrainingRun: (row: TrainingRunRow) => void;
  onProgress: (current: number, total: number) => void;
  formatCompletedAt: () => string;
  formatMetricNumber: (value: unknown) => string;
};

export async function runPytorchTraining(
  problem: RunPytorchTrainingProblem,
  deps: RunPytorchTrainingDeps
): Promise<void> {
  const total = problem.combinations.length;
  for (let i = 0; i < total; i += 1) {
    const combo = problem.combinations[i];
    const result = await deps.trainModel({
      dataset_id: problem.datasetId,
      target_column: problem.targetColumn,
      save_model: false,
      exclude_columns: problem.excludeColumns,
      date_columns: problem.dateColumns,
      task: problem.task,
      epochs: combo.epochs,
      batch_size: combo.batchSize,
      learning_rate: combo.learningRate,
      test_size: combo.testSize,
      hidden_dim: combo.hiddenDim,
      num_hidden_layers: combo.numHiddenLayers,
      dropout: combo.dropout,
    });
    deps.onProgress(i + 1, total);

    if (result.status === "error") {
      deps.prependTrainingRun({
        result: "failed",
        completed_at: deps.formatCompletedAt(),
        epochs: combo.epochs,
        learning_rate: deps.formatMetricNumber(combo.learningRate),
        test_size: deps.formatMetricNumber(combo.testSize),
        batch_size: combo.batchSize,
        hidden_dim: combo.hiddenDim,
        num_hidden_layers: combo.numHiddenLayers,
        dropout: deps.formatMetricNumber(combo.dropout),
        task: problem.task,
        target_column: problem.targetColumn,
        dataset_id: problem.datasetId,
        metric_name: "n/a",
        metric_score: "n/a",
        train_loss: "n/a",
        test_loss: "n/a",
        model_id: "n/a",
        model_path: "n/a",
        error: result.error ?? "Training failed.",
      });
      continue;
    }

    const metrics = (result.metrics ?? {}) as TrainingMetrics;
    deps.prependTrainingRun({
      result: "completed",
      completed_at: deps.formatCompletedAt(),
      epochs: combo.epochs,
      learning_rate: deps.formatMetricNumber(combo.learningRate),
      test_size: deps.formatMetricNumber(combo.testSize),
      batch_size: combo.batchSize,
      hidden_dim: combo.hiddenDim,
      num_hidden_layers: combo.numHiddenLayers,
      dropout: deps.formatMetricNumber(combo.dropout),
      task: problem.task,
      target_column: problem.targetColumn,
      dataset_id: problem.datasetId,
      metric_name: metrics.test_metric_name ?? "n/a",
      metric_score: deps.formatMetricNumber(metrics.test_metric_value),
      train_loss: deps.formatMetricNumber(metrics.train_loss),
      test_loss: deps.formatMetricNumber(metrics.test_loss),
      model_id: result.model_id ?? "n/a",
      model_path: result.model_path ?? "n/a",
    });
  }
}

export type PytorchTeacherConfig = {
  hidden: number;
  layers: number;
  dropout: number;
  epochs: number;
  batch: number;
  learningRate: number;
  testSize: number;
  modelId?: string;
  modelPath?: string;
};

export type RunPytorchDistillationProblem = {
  datasetId: string;
  targetColumn: string;
  task: "classification" | "regression" | "auto";
  saveDistilledModel: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  teacher: PytorchTeacherConfig;
};

export type RunPytorchDistillationDeps = {
  distillModel: (payload: PytorchDistillRequest) => Promise<{
    status: "ok" | "error";
    model_id?: string;
    model_path?: string;
    metrics?: unknown;
    error?: string;
  }>;
  formatCompletedAt: () => string;
  formatMetricNumber: (value: unknown) => string;
};

export async function runPytorchDistillation(
  problem: RunPytorchDistillationProblem,
  deps: RunPytorchDistillationDeps
): Promise<
  | { status: "error"; error: string }
  | {
      status: "ok";
      metrics: TrainingMetrics;
      modelId: string | null;
      modelPath: string | null;
      distilledRun: TrainingRunRow;
    }
> {
  const result = await deps.distillModel({
    dataset_id: problem.datasetId,
    target_column: problem.targetColumn,
    save_model: problem.saveDistilledModel,
    teacher_model_id: problem.teacher.modelId,
    teacher_model_path: problem.teacher.modelPath,
    exclude_columns: problem.excludeColumns,
    date_columns: problem.dateColumns,
    task: problem.task,
    epochs: Math.max(30, Math.round(problem.teacher.epochs)),
    batch_size: Math.max(1, Math.round(problem.teacher.batch)),
    learning_rate: problem.teacher.learningRate,
    test_size: problem.teacher.testSize,
    temperature: 2.5,
    alpha: 0.5,
    student_hidden_dim: Math.max(16, Math.round(problem.teacher.hidden / 2)),
    student_num_hidden_layers: Math.max(1, Math.min(15, Math.round(problem.teacher.layers - 1))),
    student_dropout: Math.min(0.5, problem.teacher.dropout + 0.05),
  });

  if (result.status === "error") {
    return { status: "error", error: result.error ?? "Distillation failed." };
  }

  const metrics = (result.metrics ?? {}) as TrainingMetrics;
  const distilledRun: TrainingRunRow = {
    result: "distilled",
    completed_at: deps.formatCompletedAt(),
    epochs: Math.max(30, Math.round(problem.teacher.epochs)),
    learning_rate: deps.formatMetricNumber(problem.teacher.learningRate),
    test_size: deps.formatMetricNumber(problem.teacher.testSize),
    batch_size: Math.max(1, Math.round(problem.teacher.batch)),
    hidden_dim: Math.max(16, Math.round(problem.teacher.hidden / 2)),
    num_hidden_layers: Math.max(1, Math.min(15, Math.round(problem.teacher.layers - 1))),
    dropout: deps.formatMetricNumber(Math.min(0.5, problem.teacher.dropout + 0.05)),
    task: problem.task,
    target_column: problem.targetColumn,
    dataset_id: problem.datasetId,
    metric_name: metrics.test_metric_name ?? "n/a",
    metric_score: deps.formatMetricNumber(metrics.test_metric_value),
    train_loss: deps.formatMetricNumber(metrics.train_loss),
    test_loss: deps.formatMetricNumber(metrics.test_loss),
    model_id: result.model_id ?? "n/a",
    model_path: result.model_path ?? "n/a",
    error: "",
  };

  return {
    status: "ok",
    metrics,
    modelId: result.model_id ?? null,
    modelPath: result.model_path ?? null,
    distilledRun,
  };
}
