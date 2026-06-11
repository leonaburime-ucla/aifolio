import { ref, computed } from "vue";
import { useDatasetLoader } from "./useDatasetLoader";
import { useTrainingOrchestrator } from "./useTrainingOrchestrator";
import type { DatasetLoaderApi } from "./useDatasetLoader";
import { createMlTrainingApi } from "../api";
import type { MlTrainingApi } from "../api";
import type { Framework } from "./useTrainingOrchestrator";
import type { TrainingMetrics, TrainingRunRow } from "@aifolio/contracts/entities/ml-training";
import {
  validateEpochValues,
  validateBatchSizes,
  validateLearningRates,
  validateTestSizes,
  validateHiddenDims,
  validateNumHiddenLayers,
  validateDropouts,
  calculatePlannedRunCount,
  getTrainingDefaults,
  createToggleRunSweepHandler,
  createReloadSweepValuesHandler,
  handleFindOptimalParams,
  handleApplyOptimalParams,
  handleCopyTrainingRuns,
  resolveTeacherRunKey,
  splitColumnInput,
  validateTrainingSetup,
  buildDistillationComparison,
  buildEnrichedDistilledRun,
  resolveDistilledModalPayload,
} from "@aifolio/frontend-core/ml-training";

export type UseTrainingScreenOptions = {
  baseUrl: string;
  framework: Framework;
  defaultTrainingMode: string;
  defaultExcludeColumns?: string;
  datasetApi?: DatasetLoaderApi;
  trainingApi?: MlTrainingApi;
};

type DistillationComparisonInput = {
  metrics: TrainingMetrics;
  distilledRun: TrainingRunRow;
  teacherInputDim: number | null;
  teacherOutputDim: number | null;
  studentInputDim: number | null;
  studentOutputDim: number | null;
  teacherModelSizeBytes: number | null;
  studentModelSizeBytes: number | null;
  sizeSavedBytes: number | null;
  sizeSavedPercent: number | null;
  teacherParamCount: number | null;
  studentParamCount: number | null;
  paramSavedCount: number | null;
  paramSavedPercent: number | null;
};

function buildFallbackDistilledRun({
  result,
  payload,
}: {
  result: Awaited<ReturnType<MlTrainingApi["distillPytorch"]>>;
  payload: {
    epochs: number;
    learning_rate: number;
    test_size: number;
    batch_size: number;
    student_hidden_dim: number;
    student_num_hidden_layers: number;
    student_dropout: number;
    task: string;
    training_mode: string;
    target_column: string;
    dataset_id: string;
  };
}): TrainingRunRow {
  return {
    result: "distilled",
    completed_at: new Date().toISOString(),
    epochs: payload.epochs,
    learning_rate: payload.learning_rate.toString(),
    test_size: payload.test_size.toString(),
    batch_size: payload.batch_size,
    hidden_dim: payload.student_hidden_dim,
    num_hidden_layers: payload.student_num_hidden_layers,
    dropout: payload.student_dropout.toString(),
    task: payload.task,
    training_mode: payload.training_mode,
    target_column: payload.target_column,
    dataset_id: payload.dataset_id,
    metric_name: result.metrics?.test_metric_name ?? "n/a",
    metric_score: String(result.metrics?.test_metric_value ?? "n/a"),
    train_loss: String(result.metrics?.train_loss ?? "n/a"),
    test_loss: String(result.metrics?.test_loss ?? "n/a"),
    model_id: result.model_id ?? "n/a",
    model_path: result.model_path ?? "n/a",
    run_id: result.run_id ?? "n/a",
  };
}

function toDistillationComparisonInput({
  result,
  payload,
}: {
  result: Awaited<ReturnType<MlTrainingApi["distillPytorch"]>>;
  payload: Parameters<typeof buildFallbackDistilledRun>[0]["payload"];
}): DistillationComparisonInput {
  return {
    metrics: result.metrics ?? {},
    distilledRun: buildFallbackDistilledRun({ result, payload }),
    teacherInputDim: result.teacher_input_dim ?? null,
    teacherOutputDim: result.teacher_output_dim ?? null,
    studentInputDim: result.student_input_dim ?? null,
    studentOutputDim: result.student_output_dim ?? null,
    teacherModelSizeBytes: result.teacher_model_size_bytes ?? null,
    studentModelSizeBytes: result.student_model_size_bytes ?? null,
    sizeSavedBytes: result.size_saved_bytes ?? null,
    sizeSavedPercent: result.size_saved_percent ?? null,
    teacherParamCount: result.teacher_param_count ?? null,
    studentParamCount: result.student_param_count ?? null,
    paramSavedCount: result.param_saved_count ?? null,
    paramSavedPercent: result.param_saved_percent ?? null,
  };
}

function nullableTeacherReference(value: unknown): string | undefined {
  const text = String(value ?? "").trim();
  return text && text !== "n/a" ? text : undefined;
}

function numericTeacherValue(value: unknown, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function resolveDistilledEpochs(framework: Framework, teacherEpochs: number): number {
  if (framework === "tensorflow") {
    return Math.min(24, Math.max(8, Math.round(teacherEpochs * 0.4)));
  }
  return Math.max(30, Math.round(teacherEpochs));
}

export function useTrainingScreen(options: UseTrainingScreenOptions) {
  const api = options.trainingApi ?? createMlTrainingApi({ baseUrl: options.baseUrl });

  const dataset = useDatasetLoader({
    baseUrl: options.baseUrl,
    api: options.datasetApi,
  });

  const orchestrator = useTrainingOrchestrator({
    baseUrl: options.baseUrl,
    framework: options.framework,
    api,
  });

  // Modal open states
  const isModelPreviewOpen = ref(false);
  const isOptimalModalOpen = ref(false);
  const isDistillMetricsModalOpen = ref(false);

  // Bayesian suggestions
  const pendingOptimalParams = ref<any>(null);
  const pendingOptimalPrediction = ref<any>(null);
  const optimizerStatus = ref<string | null>(null);

  // Distillation states
  const distillMetrics = ref<any>(null);
  const distillModelId = ref<string | null>(null);
  const distillModelPath = ref<string | null>(null);
  const distillComparison = ref<any>(null);

  const distillingTeacherKey = ref<string | null>(null);
  const distilledByTeacher = ref<Record<string, string>>({});
  const distilledSnapshotsByTeacher = ref<Record<string, any>>({});
  const distillStatus = ref<string | null>(null);

  const copyRunsStatus = ref<string | null>(null);

  // Form states
  const trainingMode = ref(options.defaultTrainingMode);
  const task = ref("auto");
  const epochValues = ref("60");
  const batchSizes = ref("64");
  const learningRates = ref("0.001");
  const testSizes = ref("0.2");
  const hiddenDims = ref("128");
  const numHiddenLayers = ref("2");
  const dropouts = ref("0.1");
  const excludeColumns = ref(options.defaultExcludeColumns ?? "");
  const dateColumns = ref("");
  const sweepEnabled = ref(false);
  const autoDistill = ref(false);

  const isLinearBaseline = computed(() => trainingMode.value === "linear_glm_baseline");

  // Validations
  const epochsValidation = computed(() => validateEpochValues({ raw: epochValues.value }));
  const batchSizesValidation = computed(() => validateBatchSizes({ raw: batchSizes.value }));
  const learningRatesValidation = computed(() => validateLearningRates({ raw: learningRates.value }));
  const testSizesValidation = computed(() => validateTestSizes({ raw: testSizes.value }));
  const hiddenDimsValidation = computed(() => validateHiddenDims({ raw: hiddenDims.value }));
  const numHiddenLayersValidation = computed(() => validateNumHiddenLayers({ raw: numHiddenLayers.value }));
  const dropoutsValidation = computed(() => validateDropouts({ raw: dropouts.value }));

  const defaults = computed(() => getTrainingDefaults(dataset.selectedDatasetId.value));

  const plannedRunCount = computed(() => {
    return calculatePlannedRunCount({
      isLinearBaselineMode: isLinearBaseline.value,
      validations: {
        epochsValidation: epochsValidation.value,
        testSizesValidation: testSizesValidation.value,
        learningRatesValidation: learningRatesValidation.value,
        batchSizesValidation: batchSizesValidation.value,
        hiddenDimsValidation: hiddenDimsValidation.value,
        numHiddenLayersValidation: numHiddenLayersValidation.value,
        dropoutsValidation: dropoutsValidation.value,
      },
    });
  });

  const isTrainDisabled = computed(
    () => orchestrator.isTraining.value || !dataset.selectedDatasetId.value || plannedRunCount.value === 0
  );

  const completedRuns = computed(() => {
    return orchestrator.trainingRuns.value.filter(
      (run) => run.result === "completed" || run.status === "completed"
    );
  });

  const uiStateObj = {
    get epochValuesInput() { return epochValues.value; },
    setEpochValuesInput(v: string) { epochValues.value = v; },
    get batchSizesInput() { return batchSizes.value; },
    setBatchSizesInput(v: string) { batchSizes.value = v; },
    get learningRatesInput() { return learningRates.value; },
    setLearningRatesInput(v: string) { learningRates.value = v; },
    get testSizesInput() { return testSizes.value; },
    setTestSizesInput(v: string) { testSizes.value = v; },
    get hiddenDimsInput() { return hiddenDims.value; },
    setHiddenDimsInput(v: string) { hiddenDims.value = v; },
    get numHiddenLayersInput() { return numHiddenLayers.value; },
    setNumHiddenLayersInput(v: string) { numHiddenLayers.value = v; },
    get dropoutsInput() { return dropouts.value; },
    setDropoutsInput(v: string) { dropouts.value = v; },

    get runSweepEnabled() { return sweepEnabled.value; },
    setRunSweepEnabled(v: boolean) { sweepEnabled.value = v; },

    get savedSweepInputs() { return null; },
    setSavedSweepInputs(v: any) {},
    get savedNumericInputs() { return null; },
    setSavedNumericInputs(v: any) {},
  };

  const toggleRunSweep = createToggleRunSweepHandler({
    ui: uiStateObj as any,
    defaultEpochs: defaults.value?.epochs ?? 60,
  });

  const reloadSweepValues = createReloadSweepValuesHandler({
    ui: uiStateObj as any,
  });

  function onDatasetChange(nextDatasetId: string | null) {
    if (nextDatasetId) {
      dataset.onDatasetChange(nextDatasetId);
    } else {
      dataset.selectedDatasetId.value = null;
    }
    const nextDefaults = getTrainingDefaults(nextDatasetId);
    dataset.targetColumn.value = nextDefaults.targetColumn;
    excludeColumns.value = nextDefaults.excludeColumns.join(",");
    task.value = nextDefaults.task;
    epochValues.value = String(nextDefaults.epochs);
    testSizes.value = "0.2";
    learningRates.value = "0.001";
    batchSizes.value = "64";
    hiddenDims.value = "128";
    numHiddenLayers.value = "2";
    dropouts.value = "0.1";
    sweepEnabled.value = false;
    orchestrator.trainingError.value = null;
    dateColumns.value = nextDefaults.dateColumns.join(",");
  }

  const scheduleRuntime = {
    schedule: (callback: () => void, delayMs: number) => {
      setTimeout(callback, delayMs);
    },
    writeClipboardText: async (text: string) => {
      await navigator.clipboard.writeText(text);
    },
  };

  const uiForOptimizer = {
    ...uiStateObj,
    setOptimizerStatus(msg: string | null) { optimizerStatus.value = msg; },
    setPendingOptimalParams(p: any) { pendingOptimalParams.value = p; },
    setPendingOptimalPrediction(pred: any) { pendingOptimalPrediction.value = pred; },
    setIsOptimalModalOpen(open: boolean) { isOptimalModalOpen.value = open; },
  };

  function onFindOptimalParamsClick() {
    handleFindOptimalParams(
      { trainingRuns: completedRuns.value as any, ui: uiForOptimizer as any },
      { runtime: scheduleRuntime }
    );
  }

  function onApplyOptimalParams() {
    handleApplyOptimalParams(
      { ui: uiForOptimizer as any },
      { runtime: scheduleRuntime }
    );
  }

  async function onCopyTrainingRuns() {
    await handleCopyTrainingRuns(
      {
        trainingRuns: orchestrator.trainingRuns.value as any,
        setCopyRunsStatus: (status: string | null) => { copyRunsStatus.value = status; },
      },
      { runtime: scheduleRuntime }
    );
  }

  function isDistillationSupportedForRun(run: any) {
    const supported = options.framework === "tensorflow"
      ? ["mlp_dense", "linear_glm_baseline", "wide_and_deep"]
      : ["mlp_dense", "linear_glm_baseline", "tabresnet"];
    return supported.includes(String(run.training_mode ?? ""));
  }

  async function onDistillFromRun(run: any) {
    if (!dataset.selectedDatasetId.value) return;
    const teacherKey = resolveTeacherRunKey({ run });
    distillingTeacherKey.value = teacherKey;
    distillStatus.value = "Running distillation...";
    orchestrator.isTraining.value = true;

    try {
      const distillFn = options.framework === "pytorch"
        ? api.distillPytorch
        : api.distillTensorflow;
      const teacherRunId = nullableTeacherReference(run.run_id);
      const teacherModelId = nullableTeacherReference(run.model_id);
      const teacherModelPath = nullableTeacherReference(run.model_path);

      if (!teacherRunId && !teacherModelId && !teacherModelPath) {
        orchestrator.trainingError.value = "This run has no teacher model reference to distill from.";
        distillStatus.value = "Distillation failed.";
        return;
      }
      const teacherEpochs = numericTeacherValue(run.epochs, 60);
      const teacherBatchSize = numericTeacherValue(run.batch_size, 64);
      const teacherLearningRate = numericTeacherValue(run.learning_rate, 0.001);
      const teacherTestSize = numericTeacherValue(run.test_size, 0.2);
      const teacherHiddenDim = numericTeacherValue(run.hidden_dim, 128);
      const teacherNumHiddenLayers = numericTeacherValue(run.num_hidden_layers, 2);
      const teacherDropout = numericTeacherValue(run.dropout, 0.1);

      const payload = {
        dataset_id: dataset.selectedDatasetId.value,
        target_column: run.target_column || dataset.targetColumn.value,
        training_mode: run.training_mode || trainingMode.value,
        save_model: false,
        teacher_run_id: teacherRunId,
        teacher_model_id: teacherModelId,
        teacher_model_path: teacherModelPath,
        exclude_columns: splitColumnInput({ value: excludeColumns.value }),
        date_columns: splitColumnInput({ value: dateColumns.value }),
        task: run.task || task.value,
        epochs: resolveDistilledEpochs(options.framework, teacherEpochs),
        batch_size: Math.max(1, Math.round(teacherBatchSize)),
        learning_rate: teacherLearningRate,
        test_size: teacherTestSize,
        temperature: 2.5,
        alpha: 0.5,
        student_hidden_dim: Math.max(16, Math.round(teacherHiddenDim / 2)),
        student_num_hidden_layers: Math.max(1, Math.min(15, Math.round(teacherNumHiddenLayers - 1))),
        student_dropout: Math.min(0.5, teacherDropout + 0.05),
      };

      const result = await distillFn(payload);
      if (result.status !== "ok") {
        orchestrator.trainingError.value = result.error ?? "Distillation failed.";
        distillStatus.value = "Distillation failed.";
        return;
      }

      const comparisonInput = toDistillationComparisonInput({ result, payload });
      const { comparison, teacherMetricName, teacherMetricValue, studentMetricValue, qualityDelta } =
        buildDistillationComparison({ teacher: run, result: comparisonInput });

      const enrichedDistilledRun = buildEnrichedDistilledRun({
        distilledRun: comparisonInput.distilledRun,
        teacherKey,
        comparison,
        teacherMetricName,
        teacherMetricValue,
        studentMetricValue,
        qualityDelta,
      });

      distillMetrics.value = result.metrics;
      distillModelId.value = result.model_id ?? result.run_id ?? null;
      distillModelPath.value = result.model_path ?? null;
      distillComparison.value = comparison;
      isDistillMetricsModalOpen.value = true;

      // Add distilled run to table
      orchestrator.trainingRuns.value.unshift(enrichedDistilledRun as any);

      distilledByTeacher.value = {
        ...distilledByTeacher.value,
        [teacherKey]: result.run_id ?? result.model_id ?? result.model_path ?? "ready",
      };

      distilledSnapshotsByTeacher.value = {
        ...distilledSnapshotsByTeacher.value,
        [teacherKey]: {
          metrics: result.metrics,
          modelId: result.model_id,
          modelPath: result.model_path,
          comparison,
        },
      };

      distillStatus.value = "Distilled student model created.";
      setTimeout(() => { distillStatus.value = null; }, 2500);

    } catch (err) {
      orchestrator.trainingError.value = err instanceof Error ? err.message : "Distillation failed.";
      distillStatus.value = "Distillation failed.";
    } finally {
      orchestrator.isTraining.value = false;
      distillingTeacherKey.value = null;
    }
  }

  function onSeeDistilledFromRun(run: any) {
    const teacherKey = resolveTeacherRunKey({ run });
    const payload = resolveDistilledModalPayload({
      teacherKey,
      snapshotsByTeacher: distilledSnapshotsByTeacher.value,
      trainingRuns: orchestrator.trainingRuns.value as any,
    });
    if (payload.status === "missing") {
      orchestrator.trainingError.value = "No distilled result found yet for this teacher run.";
      return;
    }
    distillMetrics.value = payload.metrics;
    distillModelId.value = payload.modelId;
    distillModelPath.value = payload.modelPath;
    distillComparison.value = payload.comparison;
    isDistillMetricsModalOpen.value = true;
  }

  async function onTrain() {
    const epochs = epochsValidation.value.ok ? epochsValidation.value.values : [60];
    const batches = batchSizesValidation.value.ok ? batchSizesValidation.value.values : [64];
    const lrs = learningRatesValidation.value.ok ? learningRatesValidation.value.values : [0.001];
    const tests = testSizesValidation.value.ok ? testSizesValidation.value.values : [0.2];
    const dims = hiddenDimsValidation.value.ok ? hiddenDimsValidation.value.values : [128];
    const layers = numHiddenLayersValidation.value.ok ? numHiddenLayersValidation.value.values : [2];
    const drops = dropoutsValidation.value.ok ? dropoutsValidation.value.values : [0.1];

    const combos = orchestrator.buildCombos({
      epochs,
      batches,
      learningRates: lrs,
      testSizes: tests,
      hiddenDims: dims,
      numHiddenLayers: layers,
      dropouts: drops,
    });

    const targetCol = dataset.targetColumn.value || defaults.value.targetColumn;
    const exclCols = splitColumnInput({ value: excludeColumns.value });
    const dateCols = splitColumnInput({ value: dateColumns.value });

    const setupError = validateTrainingSetup({
      selectedDatasetId: dataset.selectedDatasetId.value,
      resolvedTargetColumn: targetCol,
      excludeColumns: exclCols,
      dateColumns: dateCols,
      isLinearBaselineMode: isLinearBaseline.value,
      validations: {
        epochsValidation: epochsValidation.value,
        testSizesValidation: testSizesValidation.value,
        learningRatesValidation: learningRatesValidation.value,
        batchSizesValidation: batchSizesValidation.value,
        hiddenDimsValidation: hiddenDimsValidation.value,
        numHiddenLayersValidation: numHiddenLayersValidation.value,
        dropoutsValidation: dropoutsValidation.value,
      },
    });

    if (setupError) {
      orchestrator.trainingError.value = setupError;
      return;
    }

    await orchestrator.train({
      datasetId: dataset.selectedDatasetId.value!,
      targetColumn: targetCol,
      trainingMode: trainingMode.value,
      task: task.value,
      excludeColumns: exclCols,
      dateColumns: dateCols,
      combos,
    });

    if (autoDistill.value && orchestrator.trainingRuns.value.length > 0) {
      if (!isDistillationSupportedForRun({ training_mode: trainingMode.value })) {
        distillStatus.value = `Auto-distill skipped: '${trainingMode.value}' distillation is not supported yet.`;
        setTimeout(() => { distillStatus.value = null; }, 3500);
        return;
      }
      for (const run of orchestrator.trainingRuns.value) {
        if (run.result === "completed" || run.status === "completed") {
          await onDistillFromRun(run);
        }
      }
    }
  }

  return {
    // dataset
    datasetOptions: dataset.datasetOptions,
    selectedDatasetId: dataset.selectedDatasetId,
    tableRows: dataset.tableRows,
    tableColumns: dataset.tableColumns,
    targetColumn: dataset.targetColumn,
    datasetError: dataset.datasetError,
    loadManifest: dataset.loadManifest,
    onDatasetChange,

    // orchestrator
    isTraining: orchestrator.isTraining,
    stopTraining: orchestrator.stopTraining,
    trainingError: orchestrator.trainingError,
    trainingRuns: orchestrator.trainingRuns,
    trainingProgress: orchestrator.trainingProgress,

    // form state
    trainingMode,
    task,
    epochValues,
    batchSizes,
    learningRates,
    testSizes,
    hiddenDims,
    numHiddenLayers,
    dropouts,
    excludeColumns,
    dateColumns,
    sweepEnabled,
    autoDistill,

    // validation
    epochsValidation,
    testSizesValidation,
    learningRatesValidation,
    batchSizesValidation,
    hiddenDimsValidation,
    numHiddenLayersValidation,
    dropoutsValidation,
    defaults,

    // computed
    isLinearBaseline,
    plannedRunCount,
    isTrainDisabled,
    completedRuns,

    // Actions & Modal controls
    onTrain,
    onCopyResults: onCopyTrainingRuns,
    isModelPreviewOpen,
    isOptimalModalOpen,
    pendingOptimalParams,
    pendingOptimalPrediction,
    optimizerStatus,
    isDistillMetricsModalOpen,
    distillMetrics,
    distillModelId,
    distillModelPath,
    distillComparison,
    distillingTeacherKey,
    distilledByTeacher,
    distillStatus,
    copyRunsStatus,
    isStopRequested: orchestrator.stopTraining,
    toggleRunSweep,
    reloadSweepValues,
    onFindOptimalParamsClick,
    onApplyOptimalParams,
    onDistillFromRun,
    onSeeDistilledFromRun,
    isDistillationSupportedForRun,
  };
}
