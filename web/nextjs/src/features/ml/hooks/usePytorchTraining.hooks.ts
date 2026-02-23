import { useMemo } from "react";
import { useMlDatasetOrchestrator } from "@/features/ml/orchestrators/mlDatasetOrchestrator";
import { distillPytorchModel, trainPytorchModel } from "@/features/ml/api/pytorchApi";
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
  type TrainingRunRow,
} from "@/features/ml/utils/trainingRuns.util";
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
  runPytorchDistillation,
  runPytorchTraining,
} from "@/features/ml/orchestrators/pytorchTraining.orchestrator";

export function usePytorchUiState() {
  return useMlTrainingUiBaseState();
}

type PytorchLogicArgs = {
  dataset: ReturnType<typeof useMlDatasetOrchestrator>;
  trainingRuns: TrainingRunRow[];
  prependTrainingRun: (row: TrainingRunRow) => void;
  ui: ReturnType<typeof usePytorchUiState>;
  runTraining: typeof runPytorchTraining;
  runDistillation: typeof runPytorchDistillation;
};

export function usePytorchLogic({
  dataset,
  trainingRuns,
  prependTrainingRun,
  ui,
  runTraining,
  runDistillation,
}: PytorchLogicArgs) {
  const defaults = getTrainingDefaults(dataset.selectedDatasetId);
  const resolvedExcludeColumnsInput =
    ui.excludeColumnsInput === null
      ? defaults.excludeColumns.join(",")
      : ui.excludeColumnsInput;
  const resolvedDateColumnsInput =
    ui.dateColumnsInput === null ? defaults.dateColumns.join(",") : ui.dateColumnsInput;

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
      !hiddenDimsValidation.ok ||
      !numHiddenLayersValidation.ok ||
      !dropoutsValidation.ok
    ) {
      return 0;
    }
    return (
      epochsValidation.values.length *
      testSizesValidation.values.length *
      learningRatesValidation.values.length *
      batchSizesValidation.values.length *
      hiddenDimsValidation.values.length *
      numHiddenLayersValidation.values.length *
      dropoutsValidation.values.length
    );
  }, [
    batchSizesValidation,
    dropoutsValidation,
    epochsValidation,
    hiddenDimsValidation,
    learningRatesValidation,
    numHiddenLayersValidation,
    testSizesValidation,
  ]);

  const completedRuns = useMemo(() => {
    return trainingRuns.filter((run) => {
      if (String(run.result ?? "") !== "completed") return false;
      const metric = String(run.metric_name ?? "").toLowerCase();
      return metric !== "n/a" && parseNumericValue(run.metric_score) !== null;
    });
  }, [trainingRuns]);

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
    if (!hiddenDimsValidation.ok) {
      ui.setTrainingError(hiddenDimsValidation.error);
      return;
    }
    if (!numHiddenLayersValidation.ok) {
      ui.setTrainingError(numHiddenLayersValidation.error);
      return;
    }
    if (!dropoutsValidation.ok) {
      ui.setTrainingError(dropoutsValidation.error);
      return;
    }

    ui.setIsTraining(true);
    ui.setTrainingError(null);
    const combinations = buildSweepCombinations({
      epochs: epochsValidation.values,
      testSizes: testSizesValidation.values,
      learningRates: learningRatesValidation.values,
      batchSizes: batchSizesValidation.values,
      hiddenDims: hiddenDimsValidation.values,
      numHiddenLayers: numHiddenLayersValidation.values,
      dropouts: dropoutsValidation.values,
    });
    ui.setTrainingProgress({ current: 0, total: combinations.length });

    await runTraining(
      {
        datasetId: dataset.selectedDatasetId,
        targetColumn: resolvedTargetColumn.trim(),
        task: ui.task,
        excludeColumns,
        dateColumns,
        combinations,
      },
      {
        trainModel: trainPytorchModel,
        prependTrainingRun,
        onProgress: (current, total) =>
          ui.setTrainingProgress({ current, total }),
        formatCompletedAt,
        formatMetricNumber,
      }
    );

    ui.setTrainingError(null);
    ui.setIsTraining(false);
    ui.setTrainingProgress({ current: 0, total: 0 });
  }

  function onFindOptimalParamsClick() {
    handleFindOptimalParams({ trainingRuns, ui });
  }

  function onApplyOptimalParams() {
    handleApplyOptimalParams({ ui });
  }

  async function onDistillClick() {
    if (!dataset.selectedDatasetId) {
      ui.setTrainingError("Please select a dataset first.");
      return;
    }
    const resolvedTargetColumn =
      ui.targetColumn.trim() || defaults.targetColumn || dataset.tableColumns[0] || "";
    if (!resolvedTargetColumn.trim()) {
      ui.setTrainingError("Please provide a target column.");
      return;
    }

    const eligibleTeacherRuns = completedRuns.filter((run) => {
      const modelId = String(run.model_id ?? "");
      const modelPath = String(run.model_path ?? "");
      const datasetMatch = String(run.dataset_id ?? "") === dataset.selectedDatasetId;
      return datasetMatch && (modelId && modelId !== "n/a" || modelPath && modelPath !== "n/a");
    });

    if (eligibleTeacherRuns.length === 0) {
      ui.setTrainingError("Need at least one completed run with a teacher model for this dataset.");
      return;
    }

    const bestTeacher = [...eligibleTeacherRuns].sort((a, b) => {
      const aMetric = parseNumericValue(a.metric_score) ?? Number.NEGATIVE_INFINITY;
      const bMetric = parseNumericValue(b.metric_score) ?? Number.NEGATIVE_INFINITY;
      const metricName = String(a.metric_name ?? b.metric_name ?? "accuracy");
      const higherIsBetter = metricHigherIsBetter(metricName);
      return higherIsBetter ? bMetric - aMetric : aMetric - bMetric;
    })[0];

    const excludeColumns = resolvedExcludeColumnsInput
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const dateColumns = resolvedDateColumnsInput
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);

    ui.setIsDistilling(true);
    ui.setTrainingError(null);
    ui.setDistillStatus("Running distillation...");

    const result = await runDistillation(
      {
        datasetId: dataset.selectedDatasetId,
        targetColumn: resolvedTargetColumn.trim(),
        task: ui.task,
        saveDistilledModel: ui.saveDistilledModel,
        excludeColumns,
        dateColumns,
        teacher: {
          hidden: parseNumericValue(bestTeacher.hidden_dim) ?? 128,
          layers: parseNumericValue(bestTeacher.num_hidden_layers) ?? 2,
          dropout: parseNumericValue(bestTeacher.dropout) ?? 0.1,
          epochs: parseNumericValue(bestTeacher.epochs) ?? 60,
          batch: parseNumericValue(bestTeacher.batch_size) ?? 64,
          learningRate: parseNumericValue(bestTeacher.learning_rate) ?? 1e-3,
          testSize: parseNumericValue(bestTeacher.test_size) ?? 0.2,
          modelId: String(bestTeacher.model_id ?? "") || undefined,
          modelPath: String(bestTeacher.model_path ?? "") || undefined,
        },
      },
      {
        distillModel: distillPytorchModel,
        formatCompletedAt,
        formatMetricNumber,
      }
    );

    if (result.status === "error") {
      ui.setTrainingError(result.error);
      ui.setDistillStatus("Distillation failed.");
      ui.setIsDistilling(false);
      return;
    }

    ui.setDistillMetrics(result.metrics);
    ui.setDistillModelId(result.modelId);
    ui.setDistillModelPath(result.modelPath);
    ui.setIsDistillMetricsModalOpen(true);
    prependTrainingRun(result.distilledRun);
    ui.setDistillStatus("Distilled student model created.");
    setTimeout(() => ui.setDistillStatus(null), 2500);
    ui.setIsDistilling(false);
  }

  async function onCopyTrainingRuns() {
    await handleCopyTrainingRuns({
      trainingRuns,
      setCopyRunsStatus: ui.setCopyRunsStatus,
    });
  }

  return {
    defaults,
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
    onDistillClick,
    onCopyTrainingRuns,
  };
}

export type PytorchIntegrationArgs = {
  useTrainingRunsState?: typeof useMlTrainingRunsAdapter;
  runTraining?: typeof runPytorchTraining;
  runDistillation?: typeof runPytorchDistillation;
};

export function usePytorchTrainingIntegration({
  useTrainingRunsState = useMlTrainingRunsAdapter,
  runTraining = runPytorchTraining,
  runDistillation = runPytorchDistillation,
}: PytorchIntegrationArgs = {}) {
  const dataset = useMlDatasetOrchestrator();
  const { trainingRuns, prependTrainingRun, clearTrainingRuns } = useTrainingRunsState();
  const ui = usePytorchUiState();
  const logic = usePytorchLogic({
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
