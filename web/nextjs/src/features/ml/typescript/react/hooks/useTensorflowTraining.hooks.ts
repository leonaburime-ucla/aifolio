import { useMemo, useRef, useState } from "react";
import { useMlDatasetOrchestrator } from "@/features/ml/typescript/react/orchestrators/mlDatasetOrchestrator";
import { distillTensorflowModel, trainTensorflowModel } from "@/features/ml/typescript/api/tensorflowApi";
import { getTrainingDefaults } from "@/features/ml/typescript/config/datasetTrainingDefaults";
import { useMlTrainingRunsAdapter } from "@/features/ml/typescript/react/state/adapters/mlTrainingRuns.adapter";
import {
  buildSweepCombinations,
  validateBatchSizes,
  validateDropouts,
  validateEpochValues,
  validateHiddenDims,
  validateLearningRates,
  validateNumHiddenLayers,
  validateTestSizes,
} from "@/features/ml/typescript/validators/trainingSweep.validators";
import {
  formatCompletedAt,
  formatMetricNumber,
  type DistillComparison,
  type TrainingMetrics,
  type TrainingRunRow,
} from "@/features/ml/typescript/utils/trainingRuns.util";
import { toast } from "react-hot-toast";
import {
  parseNumericValue,
} from "@/features/ml/typescript/utils/trainingUiShared";
import {
  resolveTargetColumn,
  resolveTeacherRunKey,
  splitColumnInput,
  validateTrainingSetup,
} from "@/features/ml/typescript/react/logic/trainingInputValidation.logic";
import {
  calculatePlannedRunCount,
  hasTeacherModelReference,
  isCompletedRunForMode,
} from "@/features/ml/typescript/react/logic/trainingHookDecisions.logic";
import { useMlTrainingUiBaseState } from "@/features/ml/typescript/react/hooks/ml.hooks.base";
import {
  createReloadSweepValuesHandler,
  createToggleRunSweepHandler,
  handleApplyOptimalParams,
  handleCopyTrainingRuns,
  handleFindOptimalParams,
} from "@/features/ml/typescript/react/hooks/logic/trainingShared.logic";
import {
  buildDistillationComparison,
  buildEnrichedDistilledRun,
  resolveDistilledModalPayload,
} from "@/features/ml/typescript/react/hooks/logic/distillationView.logic";
import {
  runTensorflowDistillation,
  runTensorflowTraining,
  type TensorflowTrainingMode,
} from "@/features/ml/typescript/react/orchestrators/tensorflowTraining.orchestrator";
import type {
  TensorflowIntegrationArgs,
  TensorflowLogicArgs,
  TensorflowRuntimeDeps,
} from "@/features/ml/__types__/typescript/react/hooks/tensorflowTraining.types";
export type {
  TensorflowIntegrationArgs,
  TensorflowLogicArgs,
  TensorflowRuntimeDeps,
} from "@/features/ml/__types__/typescript/react/hooks/tensorflowTraining.types";

export type { TensorflowTrainingMode };
const TENSORFLOW_DISTILL_SUPPORTED_MODES: TensorflowTrainingMode[] = [
  "mlp_dense",
  "linear_glm_baseline",
  "wide_and_deep",
];

function isTensorflowDistillSupportedMode(mode: string): mode is TensorflowTrainingMode {
  return TENSORFLOW_DISTILL_SUPPORTED_MODES.includes(mode as TensorflowTrainingMode);
}

export function useTensorflowUiState() {
  const baseState = useMlTrainingUiBaseState();
  const [trainingMode, setTrainingMode] = useState<TensorflowTrainingMode>("wide_and_deep");
  return {
    ...baseState,
    trainingMode,
    setTrainingMode,
  };
}

const DEFAULT_TENSORFLOW_RUNTIME: TensorflowRuntimeDeps = {
  notifySuccess: (message) => toast.success(message),
  notifyError: (message) => toast.error(message),
  schedule: (callback, delayMs) => {
    setTimeout(callback, delayMs);
  },
  writeClipboardText: async (text) => {
    const clipboard = globalThis.navigator?.clipboard;
    if (!clipboard || typeof clipboard.writeText !== "function") {
      throw new Error("Clipboard API unavailable.");
    }
    await clipboard.writeText(text);
  },
};

export function useTensorflowLogic({
  dataset,
  trainingRuns,
  prependTrainingRun,
  ui,
  runTraining,
  runDistillation,
  runtime,
}: TensorflowLogicArgs) {
  const runtimeDeps: TensorflowRuntimeDeps = { ...DEFAULT_TENSORFLOW_RUNTIME, ...runtime };
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
  const isDistillationSupported = isTensorflowDistillSupportedMode(ui.trainingMode);

  const epochsValidation = useMemo(
    () => validateEpochValues({ raw: ui.epochValuesInput }),
    [ui.epochValuesInput]
  );
  const testSizesValidation = useMemo(
    () => validateTestSizes({ raw: ui.testSizesInput }),
    [ui.testSizesInput]
  );
  const learningRatesValidation = useMemo(
    () => validateLearningRates({ raw: ui.learningRatesInput }),
    [ui.learningRatesInput]
  );
  const batchSizesValidation = useMemo(
    () => validateBatchSizes({ raw: ui.batchSizesInput }),
    [ui.batchSizesInput]
  );
  const hiddenDimsValidation = useMemo(
    () => validateHiddenDims({ raw: ui.hiddenDimsInput }),
    [ui.hiddenDimsInput]
  );
  const numHiddenLayersValidation = useMemo(
    () => validateNumHiddenLayers({ raw: ui.numHiddenLayersInput }),
    [ui.numHiddenLayersInput]
  );
  const dropoutsValidation = useMemo(
    () => validateDropouts({ raw: ui.dropoutsInput }),
    [ui.dropoutsInput]
  );

  const plannedRunCount = useMemo(() => {
    return calculatePlannedRunCount({
      isLinearBaselineMode,
      validations: {
        epochsValidation,
        testSizesValidation,
        learningRatesValidation,
        batchSizesValidation,
        hiddenDimsValidation,
        numHiddenLayersValidation,
        dropoutsValidation,
      },
    });
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
    return trainingRuns.filter((run) =>
      isCompletedRunForMode({ run, mode: ui.trainingMode })
    );
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
    const resolvedTargetColumn = resolveTargetColumn({
      targetColumn: ui.targetColumn,
      defaultTargetColumn: defaults.targetColumn,
      tableColumns: dataset.tableColumns,
    });
    const excludeColumns = splitColumnInput({ value: resolvedExcludeColumnsInput });
    const dateColumns = splitColumnInput({ value: resolvedDateColumnsInput });

    const trainingSetupError = validateTrainingSetup({
      selectedDatasetId: dataset.selectedDatasetId,
      resolvedTargetColumn,
      excludeColumns,
      dateColumns,
      isLinearBaselineMode,
      validations: {
        epochsValidation,
        testSizesValidation,
        learningRatesValidation,
        batchSizesValidation,
        hiddenDimsValidation,
        numHiddenLayersValidation,
        dropoutsValidation,
      },
    });
    if (trainingSetupError) {
      ui.setTrainingError(trainingSetupError);
      return;
    }

    ui.setIsTraining(true);
    stopRequestedRef.current = false;
    setIsStopRequested(false);
    ui.setTrainingError(null);
    const combinations = buildSweepCombinations({
      config: {
        epochs: epochsValidation.values,
        testSizes: testSizesValidation.values,
        learningRates: learningRatesValidation.values,
        batchSizes: batchSizesValidation.values,
        hiddenDims: isLinearBaselineMode ? [0] : (hiddenDimsValidation.ok ? hiddenDimsValidation.values : [0]),
        numHiddenLayers: isLinearBaselineMode ? [0] : (numHiddenLayersValidation.ok ? numHiddenLayersValidation.values : [0]),
        dropouts: isLinearBaselineMode ? [0] : (dropoutsValidation.ok ? dropoutsValidation.values : [0]),
      },
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
      if (outcome.failedRuns > 0) {
        runtimeDeps.notifyError(
          outcome.firstFailureMessage ??
            `${outcome.failedRuns} training run(s) failed in the sequence.`
        );
        if (outcome.failedRuns < outcome.completed) {
          runtimeDeps.notifySuccess(
            `Training sequence completed with partial success (${outcome.completed - outcome.failedRuns}/${outcome.completed}).`
          );
        }
      } else {
        runtimeDeps.notifySuccess("Training sequence completed.");
      }
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
        runtimeDeps.schedule(() => ui.setDistillStatus(null), 3500);
        return;
      }
      for (const run of outcome.completedTeacherRuns) {
        const teacherKey = resolveTeacherRunKey({ run });
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
    handleFindOptimalParams({ trainingRuns: completedRuns, ui }, {
      runtime: {
        schedule: runtimeDeps.schedule,
        writeClipboardText: runtimeDeps.writeClipboardText,
      },
    });
  }

  function onApplyOptimalParams() {
    handleApplyOptimalParams({ ui }, {
      runtime: {
        schedule: runtimeDeps.schedule,
        writeClipboardText: runtimeDeps.writeClipboardText,
      },
    });
  }

  async function runDistillationFromTeacher(
    teacher: TrainingRunRow,
    teacherKey: string
  ) {
    const teacherTrainingMode = String(teacher.training_mode ?? ui.trainingMode);
    if (!isTensorflowDistillSupportedMode(teacherTrainingMode)) {
      ui.setTrainingError(
        `Distillation is not supported for '${teacherTrainingMode}' yet.`
      );
      return;
    }
    if (!dataset.selectedDatasetId) return;
    const teacherRunId = String(teacher.run_id ?? "").trim();
    const teacherModelId = String(teacher.model_id ?? "").trim();
    const teacherModelPath = String(teacher.model_path ?? "").trim();
    const hasTeacherModel = hasTeacherModelReference({
      runId: teacherRunId,
      modelId: teacherModelId,
      modelPath: teacherModelPath,
    });
    if (!hasTeacherModel) {
      ui.setTrainingError("This run has no teacher model reference to distill from.");
      return;
    }

    const resolvedTargetColumn = resolveTargetColumn({
      targetColumn: ui.targetColumn,
      defaultTargetColumn: defaults.targetColumn,
      tableColumns: dataset.tableColumns,
    });
    const excludeColumns = splitColumnInput({ value: resolvedExcludeColumnsInput });
    const dateColumns = splitColumnInput({ value: resolvedDateColumnsInput });

    ui.setIsDistilling(true);
    setDistillingTeacherKey(teacherKey);
    ui.setTrainingError(null);
    ui.setDistillStatus("Running distillation...");

    const result = await runDistillation(
      {
        datasetId: dataset.selectedDatasetId,
        targetColumn: resolvedTargetColumn.trim(),
        task: ui.task,
        trainingMode: teacherTrainingMode as TensorflowTrainingMode,
        saveDistilledModel: false,
        excludeColumns,
        dateColumns,
        teacher: {
          hidden: parseNumericValue({ value: teacher.hidden_dim }) ?? 128,
          layers: parseNumericValue({ value: teacher.num_hidden_layers }) ?? 2,
          dropout: parseNumericValue({ value: teacher.dropout }) ?? 0.1,
          epochs: parseNumericValue({ value: teacher.epochs }) ?? 60,
          batch: parseNumericValue({ value: teacher.batch_size }) ?? 64,
          learningRate: parseNumericValue({ value: teacher.learning_rate }) ?? 1e-3,
          testSize: parseNumericValue({ value: teacher.test_size }) ?? 0.2,
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

    const {
      comparison,
      teacherMetricName,
      teacherMetricValue,
      studentMetricValue,
      qualityDelta,
    } = buildDistillationComparison({
      teacher,
      result,
    });
    const enrichedDistilledRun = buildEnrichedDistilledRun({
      distilledRun: result.distilledRun,
      teacherKey,
      comparison,
      teacherMetricName,
      teacherMetricValue,
      studentMetricValue,
      qualityDelta,
    });

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
    runtimeDeps.schedule(() => ui.setDistillStatus(null), 2500);
    ui.setIsDistilling(false);
    setDistillingTeacherKey(null);
  }

  async function onDistillFromRun(run: TrainingRunRow) {
    const teacherKey = resolveTeacherRunKey({ run });
    await runDistillationFromTeacher(run, teacherKey);
  }

  function onSeeDistilledFromRun(run: TrainingRunRow) {
    const teacherKey = resolveTeacherRunKey({ run });
    const payload = resolveDistilledModalPayload({
      teacherKey,
      snapshotsByTeacher: distilledSnapshotsByTeacher,
      trainingRuns,
    });
    if (payload.status === "missing") {
      ui.setTrainingError("No distilled result found yet for this teacher run.");
      return;
    }
    ui.setDistillMetrics(payload.metrics);
    ui.setDistillModelId(payload.modelId);
    ui.setDistillModelPath(payload.modelPath);
    ui.setDistillComparison(payload.comparison);
    ui.setIsDistillMetricsModalOpen(true);
  }

  async function onCopyTrainingRuns() {
    await handleCopyTrainingRuns({
      trainingRuns,
      setCopyRunsStatus: ui.setCopyRunsStatus,
    }, {
      runtime: {
        schedule: runtimeDeps.schedule,
        writeClipboardText: runtimeDeps.writeClipboardText,
      },
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
    isDistillationSupportedForRun: (run: TrainingRunRow) =>
      isTensorflowDistillSupportedMode(String(run.training_mode ?? "")),
  };
}

export function useTensorflowTrainingIntegration({
  useTrainingRunsState = useMlTrainingRunsAdapter,
  runTraining = runTensorflowTraining,
  runDistillation = runTensorflowDistillation,
  runtime,
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
    runtime,
  });

  return {
    ...dataset,
    ...ui,
    ...logic,
    trainingRuns,
    clearTrainingRuns,
  };
}
