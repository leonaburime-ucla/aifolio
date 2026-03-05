import { useMemo, useRef, useState } from "react";
import type { TrainingRunRow } from "@/features/ml/typescript/utils/trainingRuns.util";
import {
  DEFAULT_TRAINING_FRAMEWORK_LOGIC_DEPS,
  DEFAULT_TRAINING_FRAMEWORK_UI_DEPS,
} from "@/features/ml/typescript/react/hooks/trainingFramework.dependencies";
import type {
  TrainingFrameworkLogicDeps,
  TrainingFrameworkUiDeps,
} from "@/features/ml/__types__/typescript/react/hooks/trainingFramework.dependencies.types";
import type {
  FrameworkDistilledSnapshot,
  FrameworkTrainingUiState,
  UseTrainingFrameworkLogicArgs,
} from "@/features/ml/__types__/typescript/react/hooks/trainingFramework.types";
import type { ValidationResult } from "@/features/ml/__types__/typescript/validators/trainingSweep.types";

function getValidationValues(validation: ValidationResult<number>): number[] {
  return validation.ok ? validation.values : [];
}

/**
 * Shared framework-agnostic UI state hook for ML training pages.
 *
 * @param initialMode Initial training mode token for the framework wrapper.
 * @param deps Optional dependency overrides for base UI-state hook.
 * @returns Training UI state with framework mode controls.
 */
export function useTrainingFrameworkUiState<TMode extends string>(
  initialMode: TMode,
  deps?: Partial<TrainingFrameworkUiDeps>
): FrameworkTrainingUiState<TMode> {
  const { useBaseUiState } = { ...DEFAULT_TRAINING_FRAMEWORK_UI_DEPS, ...(deps ?? {}) };
  const baseState = useBaseUiState();
  const [trainingMode, setTrainingMode] = useState<TMode>(initialMode);
  return {
    ...baseState,
    trainingMode,
    setTrainingMode,
  };
}

/**
 * Shared framework-agnostic ML training logic hook.
 * Framework wrappers inject adapters for supported modes and API payload functions.
 *
 * @param args Composed dataset/ui/run state plus framework adapters.
 * @param deps Optional dependency overrides for pure logic/helper collaborators.
 * @returns Unified training behavior model consumed by framework hook wrappers.
 */
export function useTrainingFrameworkLogic<
  TMode extends string,
  TTrainPayload,
  TTrainResult,
  TDistillPayload,
  TDistillResult,
>({
  dataset,
  trainingRuns,
  prependTrainingRun,
  ui,
  runTraining,
  runDistillation,
  runtime,
  framework,
}: UseTrainingFrameworkLogicArgs<
  TMode,
  TTrainPayload,
  TTrainResult,
  TDistillPayload,
  TDistillResult
>,
deps?: Partial<TrainingFrameworkLogicDeps>) {
  const injected = useMemo(
    () => ({ ...DEFAULT_TRAINING_FRAMEWORK_LOGIC_DEPS, ...(deps ?? {}) }),
    [deps]
  );
  const runtimeDeps = { ...injected.createDefaultTrainingRuntime(), ...runtime };
  const defaults = injected.getTrainingDefaults(dataset.selectedDatasetId);
  const isLinearBaselineMode = ui.trainingMode === "linear_glm_baseline";
  const [isStopRequested, setIsStopRequested] = useState(false);
  const stopRequestedRef = useRef(false);
  const [distillingTeacherKey, setDistillingTeacherKey] = useState<string | null>(null);
  const [distilledByTeacher, setDistilledByTeacher] = useState<Record<string, string>>({});
  const [distilledSnapshotsByTeacher, setDistilledSnapshotsByTeacher] =
    useState<FrameworkDistilledSnapshot>({});

  const resolvedExcludeColumnsInput =
    ui.excludeColumnsInput === null ? defaults.excludeColumns.join(",") : ui.excludeColumnsInput;
  const resolvedDateColumnsInput =
    ui.dateColumnsInput === null ? defaults.dateColumns.join(",") : ui.dateColumnsInput;
  const isDistillationSupported = framework.isDistillationSupportedMode(ui.trainingMode);

  const epochsValidation = useMemo(
    () => injected.validateEpochValues({ raw: ui.epochValuesInput }),
    [injected, ui.epochValuesInput]
  );
  const testSizesValidation = useMemo(
    () => injected.validateTestSizes({ raw: ui.testSizesInput }),
    [injected, ui.testSizesInput]
  );
  const learningRatesValidation = useMemo(
    () => injected.validateLearningRates({ raw: ui.learningRatesInput }),
    [injected, ui.learningRatesInput]
  );
  const batchSizesValidation = useMemo(
    () => injected.validateBatchSizes({ raw: ui.batchSizesInput }),
    [injected, ui.batchSizesInput]
  );
  const hiddenDimsValidation = useMemo(
    () => injected.validateHiddenDims({ raw: ui.hiddenDimsInput }),
    [injected, ui.hiddenDimsInput]
  );
  const numHiddenLayersValidation = useMemo(
    () => injected.validateNumHiddenLayers({ raw: ui.numHiddenLayersInput }),
    [injected, ui.numHiddenLayersInput]
  );
  const dropoutsValidation = useMemo(
    () => injected.validateDropouts({ raw: ui.dropoutsInput }),
    [injected, ui.dropoutsInput]
  );

  const plannedRunCount = useMemo(() => {
    return injected.calculatePlannedRunCount({
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
    injected,
  ]);

  const completedRuns = useMemo(() => {
    return trainingRuns.filter((run) => injected.isCompletedRunForMode({ run, mode: ui.trainingMode }));
  }, [trainingRuns, ui.trainingMode, injected]);

  function onDatasetChange(nextDatasetId: string | null) {
    dataset.setSelectedDatasetId(nextDatasetId);
    const nextDefaults = injected.getTrainingDefaults(nextDatasetId);
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

  const toggleRunSweep = injected.createToggleRunSweepHandler({
    ui,
    defaultEpochs: defaults.epochs,
  });
  const reloadSweepValues = injected.createReloadSweepValuesHandler({ ui });

  function resolveDeepSweepValues() {
    if (isLinearBaselineMode) {
      return { hiddenDims: [0], numHiddenLayers: [0], dropouts: [0] };
    }
    return {
      hiddenDims: getValidationValues(hiddenDimsValidation),
      numHiddenLayers: getValidationValues(numHiddenLayersValidation),
      dropouts: getValidationValues(dropoutsValidation),
    };
  }

  async function onTrainClick() {
    const resolvedTargetColumn = injected.resolveTargetColumn({
      targetColumn: ui.targetColumn,
      defaultTargetColumn: defaults.targetColumn,
      tableColumns: dataset.tableColumns,
    });
    const excludeColumns = injected.splitColumnInput({ value: resolvedExcludeColumnsInput });
    const dateColumns = injected.splitColumnInput({ value: resolvedDateColumnsInput });

    const trainingSetupError = injected.validateTrainingSetup({
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
    const deepSweepValues = resolveDeepSweepValues();
    const combinations = injected.buildSweepCombinations({
      config: {
        epochs: getValidationValues(epochsValidation),
        testSizes: getValidationValues(testSizesValidation),
        learningRates: getValidationValues(learningRatesValidation),
        batchSizes: getValidationValues(batchSizesValidation),
        hiddenDims: deepSweepValues.hiddenDims,
        numHiddenLayers: deepSweepValues.numHiddenLayers,
        dropouts: deepSweepValues.dropouts,
      },
    });
    ui.setTrainingProgress({ current: 0, total: combinations.length });

    let outcome: Awaited<ReturnType<typeof runTraining>> | null = null;
    try {
      outcome = await runTraining(
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
          trainModel: framework.trainModel,
          prependTrainingRun,
          onProgress: (current, total) => ui.setTrainingProgress({ current, total }),
          formatCompletedAt: injected.formatCompletedAt,
          formatMetricNumber: injected.formatMetricNumber,
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
    } catch (error) {
      const message = error instanceof Error ? error.message : "Training failed unexpectedly.";
      ui.setTrainingError(message);
      runtimeDeps.notifyError(message);
      return;
    } finally {
      ui.setIsTraining(false);
      ui.setTrainingProgress({ current: 0, total: 0 });
      stopRequestedRef.current = false;
      setIsStopRequested(false);
    }

    if (!outcome) {
      return;
    }

    if (ui.autoDistillEnabled && outcome.completedTeacherRuns.length > 0) {
      if (!isDistillationSupported) {
        ui.setDistillStatus(
          `Auto-distill skipped: '${ui.trainingMode}' distillation is not supported yet.`
        );
        runtimeDeps.schedule(() => ui.setDistillStatus(null), 3500);
        return;
      }

      for (const run of outcome.completedTeacherRuns) {
        const teacherKey = injected.resolveTeacherRunKey({ run });
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
    injected.handleFindOptimalParams(
      { trainingRuns: completedRuns, ui },
      {
        runtime: {
          schedule: runtimeDeps.schedule,
          writeClipboardText: runtimeDeps.writeClipboardText,
        },
      }
    );
  }

  function onApplyOptimalParams() {
    injected.handleApplyOptimalParams(
      { ui },
      {
        runtime: {
          schedule: runtimeDeps.schedule,
          writeClipboardText: runtimeDeps.writeClipboardText,
        },
      }
    );
  }

  async function runDistillationFromTeacher(teacher: TrainingRunRow, teacherKey: string) {
    const teacherTrainingMode = String(teacher.training_mode ?? ui.trainingMode);
    if (!framework.isDistillationSupportedMode(teacherTrainingMode)) {
      ui.setTrainingError(`Distillation is not supported for '${teacherTrainingMode}' yet.`);
      return;
    }
    if (!dataset.selectedDatasetId) return;

    const teacherRunId = String(teacher.run_id ?? "").trim();
    const teacherModelId = String(teacher.model_id ?? "").trim();
    const teacherModelPath = String(teacher.model_path ?? "").trim();
    const hasTeacherModel = injected.hasTeacherModelReference({
      runId: teacherRunId,
      modelId: teacherModelId,
      modelPath: teacherModelPath,
    });
    if (!hasTeacherModel) {
      ui.setTrainingError("This run has no teacher model reference to distill from.");
      return;
    }

    const resolvedTargetColumn = injected.resolveTargetColumn({
      targetColumn: ui.targetColumn,
      defaultTargetColumn: defaults.targetColumn,
      tableColumns: dataset.tableColumns,
    });
    const excludeColumns = injected.splitColumnInput({ value: resolvedExcludeColumnsInput });
    const dateColumns = injected.splitColumnInput({ value: resolvedDateColumnsInput });

    ui.setIsDistilling(true);
    setDistillingTeacherKey(teacherKey);
    ui.setTrainingError(null);
    ui.setDistillStatus("Running distillation...");

    try {
      const result = await runDistillation(
        {
          datasetId: dataset.selectedDatasetId,
          targetColumn: resolvedTargetColumn.trim(),
          task: ui.task,
          trainingMode: teacherTrainingMode as TMode,
          saveDistilledModel: false,
          excludeColumns,
          dateColumns,
          teacher: {
            hidden: injected.parseNumericValue({ value: teacher.hidden_dim }) ?? 128,
            layers: injected.parseNumericValue({ value: teacher.num_hidden_layers }) ?? 2,
            dropout: injected.parseNumericValue({ value: teacher.dropout }) ?? 0.1,
            epochs: injected.parseNumericValue({ value: teacher.epochs }) ?? 60,
            batch: injected.parseNumericValue({ value: teacher.batch_size }) ?? 64,
            learningRate: injected.parseNumericValue({ value: teacher.learning_rate }) ?? 1e-3,
            testSize: injected.parseNumericValue({ value: teacher.test_size }) ?? 0.2,
            runId: teacherRunId && teacherRunId !== "n/a" ? teacherRunId : undefined,
            modelId: teacherModelId && teacherModelId !== "n/a" ? teacherModelId : undefined,
            modelPath: teacherModelPath && teacherModelPath !== "n/a" ? teacherModelPath : undefined,
          },
        },
        {
          distillModel: framework.distillModel,
          formatCompletedAt: injected.formatCompletedAt,
          formatMetricNumber: injected.formatMetricNumber,
        }
      );

      if (result.status === "error") {
        ui.setTrainingError(result.error);
        ui.setDistillStatus("Distillation failed.");
        return;
      }

      const { comparison, teacherMetricName, teacherMetricValue, studentMetricValue, qualityDelta } =
        injected.buildDistillationComparison({ teacher, result });
      const enrichedDistilledRun = injected.buildEnrichedDistilledRun({
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
    } catch (error) {
      const message = error instanceof Error ? error.message : "Distillation failed unexpectedly.";
      ui.setTrainingError(message);
      ui.setDistillStatus("Distillation failed.");
    } finally {
      ui.setIsDistilling(false);
      setDistillingTeacherKey(null);
    }
  }

  async function onDistillFromRun(run: TrainingRunRow) {
    const teacherKey = injected.resolveTeacherRunKey({ run });
    await runDistillationFromTeacher(run, teacherKey);
  }

  function onSeeDistilledFromRun(run: TrainingRunRow) {
    const teacherKey = injected.resolveTeacherRunKey({ run });
    const payload = injected.resolveDistilledModalPayload({
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
    await injected.handleCopyTrainingRuns(
      {
        trainingRuns,
        setCopyRunsStatus: ui.setCopyRunsStatus,
      },
      {
        runtime: {
          schedule: runtimeDeps.schedule,
          writeClipboardText: runtimeDeps.writeClipboardText,
        },
      }
    );
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
      framework.isDistillationSupportedMode(String(run.training_mode ?? "")),
  };
}
