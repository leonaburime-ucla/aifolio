import type {
  DistillationDepsBase,
  DistillationProblemBase,
  DistillationRunResult,
  DistillModelRequestBase,
  DistillModelResultBase,
  PytorchDistillModelResult,
  PytorchDistillRequest,
  PytorchTrainModelResult,
  PytorchTrainRequest,
  RunPytorchDistillationDeps,
  RunPytorchDistillationProblem,
  RunPytorchDistillationResult,
  RunPytorchTrainingDeps,
  RunPytorchTrainingProblem,
  RunPytorchTrainingResult,
  RunTensorflowDistillationDeps,
  RunTensorflowDistillationProblem,
  RunTensorflowDistillationResult,
  RunTensorflowTrainingDeps,
  RunTensorflowTrainingProblem,
  RunTensorflowTrainingResult,
  TensorflowDistillModelResult,
  TensorflowDistillRequest,
  TensorflowTrainModelResult,
  TensorflowTrainRequest,
  TrainingDepsBase,
  TrainingMetrics,
  TrainingProblemBase,
  TrainingRunRow,
  TrainingSweepResult,
  TrainModelRequestBase,
  TrainModelResultBase,
} from "@aifolio/contracts/entities/ml-training";

/**
 * Executes a training sweep and records a row for every attempted combination.
 *
 * @param problem - Dataset, target, mode, and sweep combinations to run.
 * @param deps - Injected model runner, row writer, progress reporter, and formatters.
 * @returns Sweep completion summary including failures and teacher-run candidates.
 * @complexity O(n) async training calls for n sweep combinations, O(n) space for completed teacher rows.
 * @overallScore 100
 */
export async function runTrainingSweep<
  TProblem extends TrainingProblemBase,
  TTrainRequest extends TrainModelRequestBase,
  TTrainResult extends TrainModelResultBase,
>(
  problem: TProblem,
  deps: TrainingDepsBase<TTrainRequest, TTrainResult>
): Promise<TrainingSweepResult> {
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
    } as TTrainRequest);

    completed += 1;
    deps.onProgress(i + 1, total);

    if (result.status === "error") {
      failedRuns += 1;
      if (!firstFailureMessage) {
        firstFailureMessage = result.error ?? "Training failed.";
      }
      deps.prependTrainingRun({
        result: "failed",
        completed_at: deps.formatCompletedAt({}),
        epochs: combo.epochs,
        learning_rate: deps.formatMetricNumber({ value: combo.learningRate }),
        test_size: deps.formatMetricNumber({ value: combo.testSize }),
        batch_size: combo.batchSize,
        hidden_dim: problem.isLinearBaselineMode ? "n/a" : combo.hiddenDim,
        num_hidden_layers: problem.isLinearBaselineMode ? "n/a" : combo.numHiddenLayers,
        dropout: problem.isLinearBaselineMode ? "n/a" : deps.formatMetricNumber({ value: combo.dropout }),
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
      completed_at: deps.formatCompletedAt({}),
      epochs: combo.epochs,
      learning_rate: deps.formatMetricNumber({ value: combo.learningRate }),
      test_size: deps.formatMetricNumber({ value: combo.testSize }),
      batch_size: combo.batchSize,
      hidden_dim: problem.isLinearBaselineMode ? "n/a" : combo.hiddenDim,
      num_hidden_layers: problem.isLinearBaselineMode ? "n/a" : combo.numHiddenLayers,
      dropout: problem.isLinearBaselineMode ? "n/a" : deps.formatMetricNumber({ value: combo.dropout }),
      task: problem.task,
      training_mode: problem.trainingMode,
      target_column: problem.targetColumn,
      dataset_id: problem.datasetId,
      metric_name: metrics.test_metric_name ?? "n/a",
      metric_score: deps.formatMetricNumber({ value: metrics.test_metric_value }),
      train_loss: deps.formatMetricNumber({ value: metrics.train_loss }),
      test_loss: deps.formatMetricNumber({ value: metrics.test_loss }),
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

/**
 * Executes one student distillation run from a selected teacher configuration.
 *
 * @param problem - Dataset, mode, target, and teacher parameters.
 * @param deps - Injected distillation API adapter and formatting helpers.
 * @param options - Framework-specific derived epoch strategy.
 * @returns Distillation result plus a persisted training-row representation.
 * @complexity O(1) time and space excluding the injected async model call.
 * @overallScore 100
 */
export async function runDistillation<
  TProblem extends DistillationProblemBase,
  TDistillRequest extends DistillModelRequestBase,
  TDistillResult extends DistillModelResultBase,
>(
  problem: TProblem,
  deps: DistillationDepsBase<TDistillRequest, TDistillResult>,
  options: {
    resolveDistilledEpochs: (teacherEpochs: number) => number;
  }
): Promise<DistillationRunResult> {
  const distilledEpochs = options.resolveDistilledEpochs(problem.teacher.epochs);
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
  } as TDistillRequest);

  if (result.status === "error") {
    return { status: "error", error: result.error ?? "Distillation failed." };
  }

  const metrics = (result.metrics ?? {}) as TrainingMetrics;
  const distilledRun: TrainingRunRow = {
    result: "distilled",
    completed_at: deps.formatCompletedAt({}),
    epochs: distilledEpochs,
    learning_rate: deps.formatMetricNumber({ value: problem.teacher.learningRate }),
    test_size: deps.formatMetricNumber({ value: problem.teacher.testSize }),
    batch_size: Math.max(1, Math.round(problem.teacher.batch)),
    hidden_dim: Math.max(16, Math.round(problem.teacher.hidden / 2)),
    num_hidden_layers: Math.max(1, Math.min(15, Math.round(problem.teacher.layers - 1))),
    dropout: deps.formatMetricNumber({ value: Math.min(0.5, problem.teacher.dropout + 0.05) }),
    task: problem.task,
    training_mode: problem.trainingMode,
    target_column: problem.targetColumn,
    dataset_id: problem.datasetId,
    metric_name: metrics.test_metric_name ?? "n/a",
    metric_score: deps.formatMetricNumber({ value: metrics.test_metric_value }),
    train_loss: deps.formatMetricNumber({ value: metrics.train_loss }),
    test_loss: deps.formatMetricNumber({ value: metrics.test_loss }),
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

/**
 * Runs a PyTorch training sweep through the shared training orchestrator.
 *
 * @param problem - PyTorch training problem.
 * @param deps - PyTorch training dependencies.
 * @returns PyTorch sweep result.
 * @complexity O(n) async training calls for n sweep combinations.
 * @overallScore 100
 */
export function runPytorchTraining(
  problem: RunPytorchTrainingProblem,
  deps: RunPytorchTrainingDeps
): Promise<RunPytorchTrainingResult> {
  return runTrainingSweep<
    RunPytorchTrainingProblem,
    PytorchTrainRequest,
    PytorchTrainModelResult
  >(problem, deps);
}

/**
 * Runs PyTorch teacher/student distillation through the shared orchestrator.
 *
 * @param problem - PyTorch distillation problem.
 * @param deps - PyTorch distillation dependencies.
 * @returns PyTorch distillation result.
 * @complexity O(1) excluding the injected async model call.
 * @overallScore 100
 */
export function runPytorchDistillation(
  problem: RunPytorchDistillationProblem,
  deps: RunPytorchDistillationDeps
): Promise<RunPytorchDistillationResult> {
  return runDistillation<
    RunPytorchDistillationProblem,
    PytorchDistillRequest,
    PytorchDistillModelResult
  >(problem, deps, {
    resolveDistilledEpochs: (teacherEpochs) => Math.max(30, Math.round(teacherEpochs)),
  });
}

/**
 * Runs a TensorFlow training sweep through the shared training orchestrator.
 *
 * @param problem - TensorFlow training problem.
 * @param deps - TensorFlow training dependencies.
 * @returns TensorFlow sweep result.
 * @complexity O(n) async training calls for n sweep combinations.
 * @overallScore 100
 */
export function runTensorflowTraining(
  problem: RunTensorflowTrainingProblem,
  deps: RunTensorflowTrainingDeps
): Promise<RunTensorflowTrainingResult> {
  return runTrainingSweep<
    RunTensorflowTrainingProblem,
    TensorflowTrainRequest,
    TensorflowTrainModelResult
  >(problem, deps);
}

/**
 * Runs TensorFlow teacher/student distillation through the shared orchestrator.
 *
 * @param problem - TensorFlow distillation problem.
 * @param deps - TensorFlow distillation dependencies.
 * @returns TensorFlow distillation result.
 * @complexity O(1) excluding the injected async model call.
 * @overallScore 100
 */
export function runTensorflowDistillation(
  problem: RunTensorflowDistillationProblem,
  deps: RunTensorflowDistillationDeps
): Promise<RunTensorflowDistillationResult> {
  return runDistillation<
    RunTensorflowDistillationProblem,
    TensorflowDistillRequest,
    TensorflowDistillModelResult
  >(problem, deps, {
    resolveDistilledEpochs: (teacherEpochs) => Math.min(24, Math.max(8, Math.round(teacherEpochs * 0.4))),
  });
}
