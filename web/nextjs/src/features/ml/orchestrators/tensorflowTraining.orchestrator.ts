import type {
  TensorflowDistillRequest,
  TensorflowTrainRequest,
} from "@/features/ml/api/tensorflowApi";
import type { TrainingMetrics, TrainingRunRow } from "@/features/ml/utils/trainingRuns.util";

type TensorflowTrainCombo = {
  epochs: number;
  batchSize: number;
  learningRate: number;
  testSize: number;
  hiddenDim: number;
  numHiddenLayers: number;
  dropout: number;
};

export type TensorflowTrainingMode =
  | "mlp_dense"
  | "linear_glm_baseline"
  | "wide_and_deep"
  | "imbalance_aware"
  | "quantile_regression"
  | "calibrated_classifier"
  | "entity_embeddings"
  | "autoencoder_head"
  | "multi_task_learning"
  | "time_aware_tabular";

export type RunTensorflowTrainingProblem = {
  datasetId: string;
  targetColumn: string;
  task: "classification" | "regression" | "auto";
  trainingMode: TensorflowTrainingMode;
  isLinearBaselineMode: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  combinations: TensorflowTrainCombo[];
};

export type RunTensorflowTrainingDeps = {
  trainModel: (payload: TensorflowTrainRequest) => Promise<{
    status: "ok" | "error";
    run_id?: string;
    model_id?: string;
    model_path?: string;
    metrics?: unknown;
    error?: string;
  }>;
  prependTrainingRun: (row: TrainingRunRow) => void;
  onProgress: (current: number, total: number) => void;
  formatCompletedAt: () => string;
  formatMetricNumber: (value: unknown) => string;
  shouldContinue?: () => boolean;
};

export async function runTensorflowTraining(
  problem: RunTensorflowTrainingProblem,
  deps: RunTensorflowTrainingDeps
): Promise<{
  stopped: boolean;
  completed: number;
  total: number;
  completedTeacherRuns: TrainingRunRow[];
  failedRuns: number;
  firstFailureMessage: string | null;
}> {
  const total = problem.combinations.length;
  let completed = 0;
  let failedRuns = 0;
  let firstFailureMessage: string | null = null;
  const completedTeacherRuns: TrainingRunRow[] = [];
  for (let i = 0; i < total; i += 1) {
    if (deps.shouldContinue && !deps.shouldContinue()) {
      return {
        stopped: true,
        completed,
        total,
        completedTeacherRuns,
        failedRuns,
        firstFailureMessage,
      };
    }
    const combo = problem.combinations[i];
    const result = await deps.trainModel({
      dataset_id: problem.datasetId,
      target_column: problem.targetColumn,
      training_mode: problem.trainingMode,
      save_model: false,
      exclude_columns: problem.excludeColumns,
      date_columns: problem.dateColumns,
      task: problem.task,
      epochs: combo.epochs,
      batch_size: combo.batchSize,
      learning_rate: combo.learningRate,
      test_size: combo.testSize,
      hidden_dim: problem.isLinearBaselineMode ? 128 : combo.hiddenDim,
      num_hidden_layers: problem.isLinearBaselineMode ? 2 : combo.numHiddenLayers,
      dropout: problem.isLinearBaselineMode ? 0.1 : combo.dropout,
    });
    completed += 1;
    deps.onProgress(i + 1, total);

    if (result.status === "error") {
      failedRuns += 1;
      if (!firstFailureMessage) {
        firstFailureMessage = result.error ?? "Training failed.";
      }
      deps.prependTrainingRun({
        result: "failed",
        completed_at: deps.formatCompletedAt(),
        epochs: combo.epochs,
        learning_rate: deps.formatMetricNumber(combo.learningRate),
        test_size: deps.formatMetricNumber(combo.testSize),
        batch_size: combo.batchSize,
        hidden_dim: problem.isLinearBaselineMode ? "n/a" : combo.hiddenDim,
        num_hidden_layers: problem.isLinearBaselineMode ? "n/a" : combo.numHiddenLayers,
        dropout: problem.isLinearBaselineMode ? "n/a" : deps.formatMetricNumber(combo.dropout),
        task: problem.task,
        training_mode: problem.trainingMode,
        target_column: problem.targetColumn,
        dataset_id: problem.datasetId,
        metric_name: "n/a",
        metric_score: "n/a",
        train_loss: "n/a",
        test_loss: "n/a",
        model_id: "n/a",
        model_path: "n/a",
        run_id: "n/a",
        error: result.error ?? "Training failed.",
      });
      continue;
    }

    const metrics = (result.metrics ?? {}) as TrainingMetrics;
    const completedRun: TrainingRunRow = {
      result: "completed",
      completed_at: deps.formatCompletedAt(),
      epochs: combo.epochs,
      learning_rate: deps.formatMetricNumber(combo.learningRate),
      test_size: deps.formatMetricNumber(combo.testSize),
      batch_size: combo.batchSize,
      hidden_dim: problem.isLinearBaselineMode ? "n/a" : combo.hiddenDim,
      num_hidden_layers: problem.isLinearBaselineMode ? "n/a" : combo.numHiddenLayers,
      dropout: problem.isLinearBaselineMode ? "n/a" : deps.formatMetricNumber(combo.dropout),
      task: problem.task,
      training_mode: problem.trainingMode,
      target_column: problem.targetColumn,
      dataset_id: problem.datasetId,
      metric_name: metrics.test_metric_name ?? "n/a",
      metric_score: deps.formatMetricNumber(metrics.test_metric_value),
      train_loss: deps.formatMetricNumber(metrics.train_loss),
      test_loss: deps.formatMetricNumber(metrics.test_loss),
      model_id: result.model_id ?? "n/a",
      model_path: result.model_path ?? "n/a",
      run_id: result.run_id ?? "n/a",
    };
    deps.prependTrainingRun(completedRun);
    completedTeacherRuns.push(completedRun);
  }
  return {
    stopped: false,
    completed,
    total,
    completedTeacherRuns,
    failedRuns,
    firstFailureMessage,
  };
}

export type TensorflowTeacherConfig = {
  hidden: number;
  layers: number;
  dropout: number;
  epochs: number;
  batch: number;
  learningRate: number;
  testSize: number;
  runId?: string;
  modelId?: string;
  modelPath?: string;
};

export type RunTensorflowDistillationProblem = {
  datasetId: string;
  targetColumn: string;
  task: "classification" | "regression" | "auto";
  trainingMode: TensorflowTrainingMode;
  saveDistilledModel: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  teacher: TensorflowTeacherConfig;
};

export type RunTensorflowDistillationDeps = {
  distillModel: (payload: TensorflowDistillRequest) => Promise<{
    status: "ok" | "error";
    run_id?: string;
    model_id?: string;
    model_path?: string;
    metrics?: unknown;
    teacher_input_dim?: number | null;
    teacher_output_dim?: number | null;
    student_input_dim?: number | null;
    student_output_dim?: number | null;
    teacher_model_size_bytes?: number | null;
    student_model_size_bytes?: number | null;
    size_saved_bytes?: number | null;
    size_saved_percent?: number | null;
    teacher_param_count?: number | null;
    student_param_count?: number | null;
    param_saved_count?: number | null;
    param_saved_percent?: number | null;
    error?: string;
  }>;
  formatCompletedAt: () => string;
  formatMetricNumber: (value: unknown) => string;
};

export async function runTensorflowDistillation(
  problem: RunTensorflowDistillationProblem,
  deps: RunTensorflowDistillationDeps
): Promise<
  | { status: "error"; error: string }
  | {
      status: "ok";
      metrics: TrainingMetrics;
      modelId: string | null;
      modelPath: string | null;
      runId: string | null;
      teacherModelSizeBytes: number | null;
      studentModelSizeBytes: number | null;
      teacherInputDim: number | null;
      teacherOutputDim: number | null;
      studentInputDim: number | null;
      studentOutputDim: number | null;
      sizeSavedBytes: number | null;
      sizeSavedPercent: number | null;
      teacherParamCount: number | null;
      studentParamCount: number | null;
      paramSavedCount: number | null;
      paramSavedPercent: number | null;
      distilledRun: TrainingRunRow;
    }
> {
  // Keep distillation faster than full teacher training for interactive runs.
  const distilledEpochs = Math.min(
    24,
    Math.max(8, Math.round(problem.teacher.epochs * 0.4))
  );

  const result = await deps.distillModel({
    dataset_id: problem.datasetId,
    target_column: problem.targetColumn,
    training_mode: problem.trainingMode,
    save_model: problem.saveDistilledModel,
    teacher_run_id: problem.teacher.runId,
    teacher_model_id: problem.teacher.modelId,
    teacher_model_path: problem.teacher.modelPath,
    exclude_columns: problem.excludeColumns,
    date_columns: problem.dateColumns,
    task: problem.task,
    epochs: distilledEpochs,
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
    epochs: distilledEpochs,
    learning_rate: deps.formatMetricNumber(problem.teacher.learningRate),
    test_size: deps.formatMetricNumber(problem.teacher.testSize),
    batch_size: Math.max(1, Math.round(problem.teacher.batch)),
    hidden_dim: Math.max(16, Math.round(problem.teacher.hidden / 2)),
    num_hidden_layers: Math.max(1, Math.min(15, Math.round(problem.teacher.layers - 1))),
    dropout: deps.formatMetricNumber(Math.min(0.5, problem.teacher.dropout + 0.05)),
    task: problem.task,
    training_mode: problem.trainingMode,
    target_column: problem.targetColumn,
    dataset_id: problem.datasetId,
    metric_name: metrics.test_metric_name ?? "n/a",
    metric_score: deps.formatMetricNumber(metrics.test_metric_value),
    train_loss: deps.formatMetricNumber(metrics.train_loss),
    test_loss: deps.formatMetricNumber(metrics.test_loss),
    model_id: result.model_id ?? "n/a",
    model_path: result.model_path ?? "n/a",
    run_id: result.run_id ?? "n/a",
    error: "",
  };

  return {
    status: "ok",
    metrics,
    modelId: result.model_id ?? null,
    modelPath: result.model_path ?? null,
    runId: result.run_id ?? null,
    teacherModelSizeBytes: result.teacher_model_size_bytes ?? null,
    studentModelSizeBytes: result.student_model_size_bytes ?? null,
    teacherInputDim: result.teacher_input_dim ?? null,
    teacherOutputDim: result.teacher_output_dim ?? null,
    studentInputDim: result.student_input_dim ?? null,
    studentOutputDim: result.student_output_dim ?? null,
    sizeSavedBytes: result.size_saved_bytes ?? null,
    sizeSavedPercent: result.size_saved_percent ?? null,
    teacherParamCount: result.teacher_param_count ?? null,
    studentParamCount: result.student_param_count ?? null,
    paramSavedCount: result.param_saved_count ?? null,
    paramSavedPercent: result.param_saved_percent ?? null,
    distilledRun,
  };
}
