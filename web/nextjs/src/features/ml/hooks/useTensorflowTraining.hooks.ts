import { useMemo, useRef, useState } from "react";
import { useMlDatasetOrchestrator } from "@/features/ml/orchestrators/mlDatasetOrchestrator";
import { distillTensorflowModel, trainTensorflowModel } from "@/features/ml/api/tensorflowApi";
import { getTrainingDefaults } from "@/features/ml/config/datasetTrainingDefaults";
import { useMlTrainingRunsAdapter } from "@/features/ml/state/adapters/mlTrainingRuns.adapter";
import {
  buildSweepCombinations,
  validateBatchSizes,
  validateDropouts,
  validateEpochValues,
  validateHiddenDims,
  validateLearningRates,
  validateNumHiddenLayers,
  validateTestSizes,
} from "@/features/ml/validators/trainingSweep.validators";
import {
  formatCompletedAt,
  formatMetricNumber,
  type DistillComparison,
  type TrainingMetrics,
  type TrainingRunRow,
} from "@/features/ml/utils/trainingRuns.util";
import { toast } from "react-hot-toast";
import {
  metricHigherIsBetter,
  parseNumericValue,
} from "@/features/ml/utils/trainingUiShared";
import { useMlTrainingUiBaseState } from "@/features/ml/hooks/ml.hooks.base";
import {
  createReloadSweepValuesHandler,
  createToggleRunSweepHandler,
  handleApplyOptimalParams,
  handleCopyTrainingRuns,
  handleFindOptimalParams,
} from "@/features/ml/hooks/logic/trainingShared.logic";
import {
  runTensorflowDistillation,
  runTensorflowTraining,
  type TensorflowTrainingMode,
} from "@/features/ml/orchestrators/tensorflowTraining.orchestrator";

export type { TensorflowTrainingMode };
const TENSORFLOW_DISTILL_SUPPORTED_MODES: TensorflowTrainingMode[] = [
  "mlp_dense",
  "linear_glm_baseline",
  "wide_and_deep",
];

export function useTensorflowUiState() {
  const baseState = useMlTrainingUiBaseState();
  const [trainingMode, setTrainingMode] = useState<TensorflowTrainingMode>("wide_and_deep");
  return {
    ...baseState,
    trainingMode,
    setTrainingMode,
  };
}

type TensorflowLogicArgs = {
  dataset: ReturnType<typeof useMlDatasetOrchestrator>;
  trainingRuns: TrainingRunRow[];
  prependTrainingRun: (row: TrainingRunRow) => void;
  ui: ReturnType<typeof useTensorflowUiState>;
  runTraining: typeof runTensorflowTraining;
  runDistillation: typeof runTensorflowDistillation;
};

export function useTensorflowLogic({
  dataset,
  trainingRuns,
  prependTrainingRun,
  ui,
  runTraining,
  runDistillation,
}: TensorflowLogicArgs) {
  const defaults = getTrainingDefaults(dataset.selectedDatasetId);
  const isLinearBaselineMode = ui.trainingMode === "linear_glm_baseline";
  const [isStopRequested, setIsStopRequested] = useState(false);
  const stopRequestedRef = useRef(false);
  const [distillingTeacherKey, setDistillingTeacherKey] = useState<string | null>(null);
  const [distilledByTeacher, setDistilledByTeacher] = useState<Record<string, string>>({});
  const [distilledSnapshotsByTeacher, setDistilledSnapshotsByTeacher] = useState<
    Record<string, { metrics: TrainingMetrics; modelId: string | null; modelPath: string | null; comparison: DistillComparison }>
  >({});
  const resolvedExcludeColumnsInput =
    ui.excludeColumnsInput === null
      ? defaults.excludeColumns.join(",")
      : ui.excludeColumnsInput;
  const resolvedDateColumnsInput =
    ui.dateColumnsInput === null ? defaults.dateColumns.join(",") : ui.dateColumnsInput;
  const isDistillationSupported = TENSORFLOW_DISTILL_SUPPORTED_MODES.includes(ui.trainingMode);

  const epochsValidation = useMemo(
    () => validateEpochValues(ui.epochValuesInput),
    [ui.epochValuesInput]
  );
  const testSizesValidation = useMemo(
    () => validateTestSizes(ui.testSizesInput),
    [ui.testSizesInput]
  );
  const learningRatesValidation = useMemo(
    () => validateLearningRates(ui.learningRatesInput),
    [ui.learningRatesInput]
  );
  const batchSizesValidation = useMemo(
    () => validateBatchSizes(ui.batchSizesInput),
    [ui.batchSizesInput]
  );
  const hiddenDimsValidation = useMemo(
    () => validateHiddenDims(ui.hiddenDimsInput),
    [ui.hiddenDimsInput]
  );
  const numHiddenLayersValidation = useMemo(
    () => validateNumHiddenLayers(ui.numHiddenLayersInput),
    [ui.numHiddenLayersInput]
  );
  const dropoutsValidation = useMemo(
    () => validateDropouts(ui.dropoutsInput),
    [ui.dropoutsInput]
  );

  const plannedRunCount = useMemo(() => {
    if (
      !epochsValidation.ok ||
      !testSizesValidation.ok ||
      !learningRatesValidation.ok ||
      !batchSizesValidation.ok ||
      (!isLinearBaselineMode && !hiddenDimsValidation.ok) ||
      (!isLinearBaselineMode && !numHiddenLayersValidation.ok) ||
      (!isLinearBaselineMode && !dropoutsValidation.ok)
    ) {
      return 0;
    }
    return (
      epochsValidation.values.length *
      testSizesValidation.values.length *
      learningRatesValidation.values.length *
      batchSizesValidation.values.length *
      (isLinearBaselineMode ? 1 : (hiddenDimsValidation.ok ? hiddenDimsValidation.values.length : 0)) *
      (isLinearBaselineMode ? 1 : (numHiddenLayersValidation.ok ? numHiddenLayersValidation.values.length : 0)) *
      (isLinearBaselineMode ? 1 : (dropoutsValidation.ok ? dropoutsValidation.values.length : 0))
    );
  }, [
    batchSizesValidation,
    dropoutsValidation,
    epochsValidation,
    hiddenDimsValidation,
    isLinearBaselineMode,
    learningRatesValidation,
    numHiddenLayersValidation,
    testSizesValidation,
  ]);

  const completedRuns = useMemo(() => {
    return trainingRuns.filter((run) => {
      if (String(run.result ?? "") !== "completed") return false;
      if (String(run.training_mode ?? "") !== ui.trainingMode) return false;
      const metric = String(run.metric_name ?? "").toLowerCase();
      return metric !== "n/a" && parseNumericValue(run.metric_score) !== null;
    });
  }, [trainingRuns, ui.trainingMode]);

  function onDatasetChange(nextDatasetId: string | null) {
    dataset.setSelectedDatasetId(nextDatasetId);
    const nextDefaults = getTrainingDefaults(nextDatasetId);
    ui.setTargetColumn(nextDefaults.targetColumn);
    ui.setExcludeColumnsInput(null);
    ui.setTask(nextDefaults.task);
    ui.setEpochValuesInput(String(nextDefaults.epochs));
    ui.setTestSizesInput("0.2");
    ui.setLearningRatesInput("0.001");
    ui.setBatchSizesInput("64");
    ui.setHiddenDimsInput("128");
    ui.setNumHiddenLayersInput("2");
    ui.setDropoutsInput("0.1");
    ui.setRunSweepEnabled(false);
    ui.setSavedNumericInputs(null);
    ui.setSavedSweepInputs(null);
    ui.setTrainingError(null);
    ui.setDateColumnsInput(null);
  }

  const toggleRunSweep = createToggleRunSweepHandler({
    ui,
    defaultEpochs: defaults.epochs,
  });

  const reloadSweepValues = createReloadSweepValuesHandler({ ui });

  async function onTrainClick() {
    if (!dataset.selectedDatasetId) {
      ui.setTrainingError("Please select a dataset first.");
      return;
    }
    const resolvedTargetColumn =
      ui.targetColumn.trim() || defaults.targetColumn || dataset.tableColumns[0] || "";
    const excludeColumns = resolvedExcludeColumnsInput
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const dateColumns = resolvedDateColumnsInput
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    if (excludeColumns.includes(resolvedTargetColumn.trim())) {
      ui.setTrainingError("Target column cannot also be in excluded columns.");
      return;
    }
    if (dateColumns.includes(resolvedTargetColumn.trim())) {
      ui.setTrainingError("Target column cannot also be in date columns.");
      return;
    }
    const overlap = dateColumns.find((col) => excludeColumns.includes(col));
    if (overlap) {
      ui.setTrainingError(`Column '${overlap}' cannot be in both excluded and date columns.`);
      return;
    }
    if (!resolvedTargetColumn.trim()) {
      ui.setTrainingError("Please provide a target column.");
      return;
    }
    if (!epochsValidation.ok) {
      ui.setTrainingError(epochsValidation.error);
      return;
    }
    if (!testSizesValidation.ok) {
      ui.setTrainingError(testSizesValidation.error);
      return;
    }
    if (!learningRatesValidation.ok) {
      ui.setTrainingError(learningRatesValidation.error);
      return;
    }
    if (!batchSizesValidation.ok) {
      ui.setTrainingError(batchSizesValidation.error);
      return;
    }
    if (!isLinearBaselineMode && !hiddenDimsValidation.ok) {
      ui.setTrainingError(hiddenDimsValidation.error);
      return;
    }
    if (!isLinearBaselineMode && !numHiddenLayersValidation.ok) {
      ui.setTrainingError(numHiddenLayersValidation.error);
      return;
    }
    if (!isLinearBaselineMode && !dropoutsValidation.ok) {
      ui.setTrainingError(dropoutsValidation.error);
      return;
    }

    ui.setIsTraining(true);
    stopRequestedRef.current = false;
    setIsStopRequested(false);
    ui.setTrainingError(null);
    const combinations = buildSweepCombinations({
      epochs: epochsValidation.values,
      testSizes: testSizesValidation.values,
      learningRates: learningRatesValidation.values,
      batchSizes: batchSizesValidation.values,
      hiddenDims: isLinearBaselineMode ? [0] : (hiddenDimsValidation.ok ? hiddenDimsValidation.values : [0]),
      numHiddenLayers: isLinearBaselineMode ? [0] : (numHiddenLayersValidation.ok ? numHiddenLayersValidation.values : [0]),
      dropouts: isLinearBaselineMode ? [0] : (dropoutsValidation.ok ? dropoutsValidation.values : [0]),
    });
    ui.setTrainingProgress({ current: 0, total: combinations.length });

    const outcome = await runTraining(
      {
        datasetId: dataset.selectedDatasetId,
        targetColumn: resolvedTargetColumn.trim(),
        task: ui.task,
        trainingMode: ui.trainingMode,
        isLinearBaselineMode,
        excludeColumns,
        dateColumns,
        combinations,
      },
      {
        trainModel: trainTensorflowModel,
        prependTrainingRun,
        onProgress: (current, total) =>
          ui.setTrainingProgress({ current, total }),
        formatCompletedAt,
        formatMetricNumber,
        shouldContinue: () => !stopRequestedRef.current,
      }
    );

    if (outcome.stopped) {
      ui.setTrainingError(`Training stopped after ${outcome.completed}/${outcome.total} run(s).`);
    } else {
      // If we didn't stop manually, but there are runs that failed, pop a toast for the user
      const failedRuns = outcome.completedTeacherRuns.filter((r) => r.result === "failed");
      if (failedRuns.length > 0) {
        toast.error(failedRuns[0].error || "A training run failed.");
      }
      toast.success("Training sequence completed.");
      ui.setTrainingError(null);
    }
    ui.setIsTraining(false);
    ui.setTrainingProgress({ current: 0, total: 0 });
    stopRequestedRef.current = false;
    setIsStopRequested(false);

    if (ui.autoDistillEnabled && outcome.completedTeacherRuns.length > 0) {
      if (!isDistillationSupported) {
        ui.setDistillStatus(
          `Auto-distill skipped: '${ui.trainingMode}' distillation is not supported yet.`
        );
        setTimeout(() => ui.setDistillStatus(null), 3500);
        return;
      }
      for (const run of outcome.completedTeacherRuns) {
        const teacherKey =
          String(run.run_id ?? "") ||
          String(run.model_id ?? "") ||
          String(run.model_path ?? "") ||
          String(run.completed_at ?? "run");
        // Sequential distillation keeps API load predictable and state updates ordered.
        await runDistillationFromTeacher(run, teacherKey);
      }
    }
  }

  function onStopTrainingRuns() {
    if (!ui.isTraining) return;
    stopRequestedRef.current = true;
    setIsStopRequested(true);
    ui.setTrainingError("Stop requested. Current run will finish, then remaining runs will be skipped.");
  }

  function onFindOptimalParamsClick() {
    handleFindOptimalParams({ trainingRuns: completedRuns, ui });
  }

  function onApplyOptimalParams() {
    handleApplyOptimalParams({ ui });
  }

  async function runDistillationFromTeacher(
    teacher: TrainingRunRow,
    teacherKey: string
  ) {
    if (!isDistillationSupported) {
      ui.setTrainingError(
        `Distillation is not supported for '${ui.trainingMode}' yet. Switch to wide & deep.`
      );
      return;
    }
    if (!dataset.selectedDatasetId) return;
    const teacherRunId = String(teacher.run_id ?? "").trim();
    const teacherModelId = String(teacher.model_id ?? "").trim();
    const teacherModelPath = String(teacher.model_path ?? "").trim();
    const hasTeacherModel =
      (teacherRunId && teacherRunId !== "n/a") ||
      (teacherModelId && teacherModelId !== "n/a") ||
      (teacherModelPath && teacherModelPath !== "n/a");
    if (!hasTeacherModel) {
      ui.setTrainingError("This run has no teacher model reference to distill from.");
      return;
    }

    const resolvedTargetColumn =
      ui.targetColumn.trim() || defaults.targetColumn || dataset.tableColumns[0] || "";

    const excludeColumns = resolvedExcludeColumnsInput
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const dateColumns = resolvedDateColumnsInput
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);

    ui.setIsDistilling(true);
    setDistillingTeacherKey(teacherKey);
    ui.setTrainingError(null);
    ui.setDistillStatus("Running distillation...");

    const result = await runDistillation(
      {
        datasetId: dataset.selectedDatasetId,
        targetColumn: resolvedTargetColumn.trim(),
        task: ui.task,
        trainingMode: ui.trainingMode,
        saveDistilledModel: false,
        excludeColumns,
        dateColumns,
        teacher: {
          hidden: parseNumericValue(teacher.hidden_dim) ?? 128,
          layers: parseNumericValue(teacher.num_hidden_layers) ?? 2,
          dropout: parseNumericValue(teacher.dropout) ?? 0.1,
          epochs: parseNumericValue(teacher.epochs) ?? 60,
          batch: parseNumericValue(teacher.batch_size) ?? 64,
          learningRate: parseNumericValue(teacher.learning_rate) ?? 1e-3,
          testSize: parseNumericValue(teacher.test_size) ?? 0.2,
          runId: teacherRunId && teacherRunId !== "n/a" ? teacherRunId : undefined,
          modelId: teacherModelId && teacherModelId !== "n/a" ? teacherModelId : undefined,
          modelPath: teacherModelPath && teacherModelPath !== "n/a" ? teacherModelPath : undefined,
        },
      },
      {
        distillModel: distillTensorflowModel,
        formatCompletedAt,
        formatMetricNumber,
      }
    );

    if (result.status === "error") {
      ui.setTrainingError(result.error);
      ui.setDistillStatus("Distillation failed.");
      ui.setIsDistilling(false);
      setDistillingTeacherKey(null);
      return;
    }

    const teacherMetricName = String(
      teacher.metric_name ?? result.metrics.test_metric_name ?? "accuracy"
    );
    const teacherMetricValue = parseNumericValue(teacher.metric_score);
    const studentMetricValue =
      typeof result.metrics.test_metric_value === "number"
        ? result.metrics.test_metric_value
        : null;
    const higherIsBetter = metricHigherIsBetter(teacherMetricName);
    const qualityDelta =
      teacherMetricValue !== null && studentMetricValue !== null
        ? higherIsBetter
          ? studentMetricValue - teacherMetricValue
          : teacherMetricValue - studentMetricValue
        : null;
    const comparison: DistillComparison = {
      metricName: teacherMetricName,
      teacherMetricValue,
      studentMetricValue,
      qualityDelta,
      higherIsBetter,
      teacherTrainingMode: String(teacher.training_mode ?? "") || null,
      studentTrainingMode: String(result.distilledRun.training_mode ?? "") || null,
      teacherHiddenDim: parseNumericValue(teacher.hidden_dim),
      studentHiddenDim: parseNumericValue(result.distilledRun.hidden_dim),
      teacherNumHiddenLayers: parseNumericValue(teacher.num_hidden_layers),
      studentNumHiddenLayers: parseNumericValue(result.distilledRun.num_hidden_layers),
      teacherInputDim: result.teacherInputDim,
      studentInputDim: result.studentInputDim,
      teacherOutputDim: result.teacherOutputDim,
      studentOutputDim: result.studentOutputDim,
      teacherModelSizeBytes: result.teacherModelSizeBytes,
      studentModelSizeBytes: result.studentModelSizeBytes,
      sizeSavedBytes: result.sizeSavedBytes,
      sizeSavedPercent: result.sizeSavedPercent,
      teacherParamCount: result.teacherParamCount,
      studentParamCount: result.studentParamCount,
      paramSavedCount: result.paramSavedCount,
      paramSavedPercent: result.paramSavedPercent,
    };
    const enrichedDistilledRun: TrainingRunRow = {
      ...result.distilledRun,
      teacher_ref_key: teacherKey,
      distill_teacher_metric_name: teacherMetricName,
      distill_teacher_metric_value: teacherMetricValue ?? "n/a",
      distill_student_metric_value: studentMetricValue ?? "n/a",
      distill_quality_delta: qualityDelta ?? "n/a",
      distill_higher_is_better: higherIsBetter ? "1" : "0",
      distill_teacher_training_mode: comparison.teacherTrainingMode ?? "n/a",
      distill_student_training_mode: comparison.studentTrainingMode ?? "n/a",
      distill_teacher_hidden_dim: comparison.teacherHiddenDim ?? "n/a",
      distill_student_hidden_dim: comparison.studentHiddenDim ?? "n/a",
      distill_teacher_num_hidden_layers: comparison.teacherNumHiddenLayers ?? "n/a",
      distill_student_num_hidden_layers: comparison.studentNumHiddenLayers ?? "n/a",
      distill_teacher_input_dim: comparison.teacherInputDim ?? "n/a",
      distill_student_input_dim: comparison.studentInputDim ?? "n/a",
      distill_teacher_output_dim: comparison.teacherOutputDim ?? "n/a",
      distill_student_output_dim: comparison.studentOutputDim ?? "n/a",
      distill_teacher_model_size_bytes: comparison.teacherModelSizeBytes ?? "n/a",
      distill_student_model_size_bytes: comparison.studentModelSizeBytes ?? "n/a",
      distill_size_saved_bytes: comparison.sizeSavedBytes ?? "n/a",
      distill_size_saved_percent: comparison.sizeSavedPercent ?? "n/a",
      distill_teacher_param_count: comparison.teacherParamCount ?? "n/a",
      distill_student_param_count: comparison.studentParamCount ?? "n/a",
      distill_param_saved_count: comparison.paramSavedCount ?? "n/a",
      distill_param_saved_percent: comparison.paramSavedPercent ?? "n/a",
    };

    ui.setDistillMetrics(result.metrics);
    ui.setDistillModelId(result.modelId ?? result.runId);
    ui.setDistillModelPath(result.modelPath);
    ui.setDistillComparison(comparison);
    ui.setIsDistillMetricsModalOpen(true);
    prependTrainingRun(enrichedDistilledRun);
    setDistilledByTeacher((prev) => ({
      ...prev,
      [teacherKey]: result.runId ?? result.modelId ?? result.modelPath ?? "ready",
    }));
    setDistilledSnapshotsByTeacher((prev) => ({
      ...prev,
      [teacherKey]: {
        metrics: result.metrics,
        modelId: result.modelId,
        modelPath: result.modelPath,
        comparison,
      },
    }));
    ui.setDistillStatus("Distilled student model created.");
    setTimeout(() => ui.setDistillStatus(null), 2500);
    ui.setIsDistilling(false);
    setDistillingTeacherKey(null);
  }

  async function onDistillFromRun(run: TrainingRunRow) {
    const teacherKey =
      String(run.run_id ?? "") ||
      String(run.model_id ?? "") ||
      String(run.model_path ?? "") ||
      String(run.completed_at ?? "run");
    await runDistillationFromTeacher(run, teacherKey);
  }

  function onSeeDistilledFromRun(run: TrainingRunRow) {
    const teacherKey =
      String(run.run_id ?? "") ||
      String(run.model_id ?? "") ||
      String(run.model_path ?? "") ||
      String(run.completed_at ?? "run");
    const snapshot = distilledSnapshotsByTeacher[teacherKey];
    if (snapshot) {
      ui.setDistillMetrics(snapshot.metrics);
      ui.setDistillModelId(snapshot.modelId);
      ui.setDistillModelPath(snapshot.modelPath);
      ui.setDistillComparison(snapshot.comparison);
      ui.setIsDistillMetricsModalOpen(true);
      return;
    }
    const fallbackDistilled = trainingRuns.find(
      (candidate) =>
        String(candidate.result ?? "") === "distilled" &&
        String(candidate.teacher_ref_key ?? "") === teacherKey
    );
    if (!fallbackDistilled) {
      ui.setTrainingError("No distilled result found yet for this teacher run.");
      return;
    }
    const fallbackComparison: DistillComparison = {
      metricName: String(fallbackDistilled.distill_teacher_metric_name ?? fallbackDistilled.metric_name ?? "accuracy"),
      teacherMetricValue: parseNumericValue(fallbackDistilled.distill_teacher_metric_value),
      studentMetricValue: parseNumericValue(fallbackDistilled.distill_student_metric_value ?? fallbackDistilled.metric_score),
      qualityDelta: parseNumericValue(fallbackDistilled.distill_quality_delta),
      higherIsBetter: String(fallbackDistilled.distill_higher_is_better ?? "1") === "1",
      teacherTrainingMode: String(fallbackDistilled.distill_teacher_training_mode ?? "n/a"),
      studentTrainingMode: String(fallbackDistilled.distill_student_training_mode ?? "n/a"),
      teacherHiddenDim: parseNumericValue(fallbackDistilled.distill_teacher_hidden_dim),
      studentHiddenDim: parseNumericValue(fallbackDistilled.distill_student_hidden_dim),
      teacherNumHiddenLayers: parseNumericValue(fallbackDistilled.distill_teacher_num_hidden_layers),
      studentNumHiddenLayers: parseNumericValue(fallbackDistilled.distill_student_num_hidden_layers),
      teacherInputDim: parseNumericValue(fallbackDistilled.distill_teacher_input_dim),
      studentInputDim: parseNumericValue(fallbackDistilled.distill_student_input_dim),
      teacherOutputDim: parseNumericValue(fallbackDistilled.distill_teacher_output_dim),
      studentOutputDim: parseNumericValue(fallbackDistilled.distill_student_output_dim),
      teacherModelSizeBytes: parseNumericValue(fallbackDistilled.distill_teacher_model_size_bytes),
      studentModelSizeBytes: parseNumericValue(fallbackDistilled.distill_student_model_size_bytes),
      sizeSavedBytes: parseNumericValue(fallbackDistilled.distill_size_saved_bytes),
      sizeSavedPercent: parseNumericValue(fallbackDistilled.distill_size_saved_percent),
      teacherParamCount: parseNumericValue(fallbackDistilled.distill_teacher_param_count),
      studentParamCount: parseNumericValue(fallbackDistilled.distill_student_param_count),
      paramSavedCount: parseNumericValue(fallbackDistilled.distill_param_saved_count),
      paramSavedPercent: parseNumericValue(fallbackDistilled.distill_param_saved_percent),
    };
    ui.setDistillMetrics({
      task: String(fallbackDistilled.task ?? "auto"),
      train_loss: parseNumericValue(fallbackDistilled.train_loss) ?? undefined,
      test_loss: parseNumericValue(fallbackDistilled.test_loss) ?? undefined,
      test_metric_name: String(fallbackDistilled.metric_name ?? fallbackComparison.metricName),
      test_metric_value: parseNumericValue(fallbackDistilled.metric_score) ?? undefined,
    });
    ui.setDistillModelId(String(fallbackDistilled.model_id ?? "n/a"));
    ui.setDistillModelPath(String(fallbackDistilled.model_path ?? "n/a"));
    ui.setDistillComparison(fallbackComparison);
    ui.setIsDistillMetricsModalOpen(true);
  }

  async function onCopyTrainingRuns() {
    await handleCopyTrainingRuns({
      trainingRuns,
      setCopyRunsStatus: ui.setCopyRunsStatus,
    });
  }

  return {
    defaults,
    isLinearBaselineMode,
    autoDistillEnabled: ui.autoDistillEnabled,
    setAutoDistillEnabled: ui.setAutoDistillEnabled,
    isStopRequested,
    distillingTeacherKey,
    distilledByTeacher,
    resolvedExcludeColumnsInput,
    resolvedDateColumnsInput,
    epochsValidation,
    testSizesValidation,
    learningRatesValidation,
    batchSizesValidation,
    hiddenDimsValidation,
    numHiddenLayersValidation,
    dropoutsValidation,
    plannedRunCount,
    completedRuns,
    onDatasetChange,
    toggleRunSweep,
    reloadSweepValues,
    onTrainClick,
    onFindOptimalParamsClick,
    onApplyOptimalParams,
    onStopTrainingRuns,
    onDistillFromRun,
    onSeeDistilledFromRun,
    onCopyTrainingRuns,
  };
}

export type TensorflowIntegrationArgs = {
  useTrainingRunsState?: typeof useMlTrainingRunsAdapter;
  runTraining?: typeof runTensorflowTraining;
  runDistillation?: typeof runTensorflowDistillation;
};

export function useTensorflowTrainingIntegration({
  useTrainingRunsState = useMlTrainingRunsAdapter,
  runTraining = runTensorflowTraining,
  runDistillation = runTensorflowDistillation,
}: TensorflowIntegrationArgs = {}) {
  const dataset = useMlDatasetOrchestrator();
  const { trainingRuns, prependTrainingRun, clearTrainingRuns } = useTrainingRunsState();
  const ui = useTensorflowUiState();
  const logic = useTensorflowLogic({
    dataset,
    trainingRuns,
    prependTrainingRun,
    ui,
    runTraining,
    runDistillation,
  });

  return {
    ...dataset,
    ...ui,
    ...logic,
    trainingRuns,
    clearTrainingRuns,
  };
}
