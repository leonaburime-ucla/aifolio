"use client";

import { useMemo, useState } from "react";
import CsvDatasetCombobox from "@/core/views/patterns/CsvDatasetCombobox";
import DataTable from "@/core/views/components/Datatable/DataTable";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/core/views/components/General/popover";
import { useMlDatasetOrchestrator } from "@/features/ml/orchestrators/mlDatasetOrchestrator";
import { distillTensorflowModel, trainTensorflowModel } from "@/features/ml/api/tensorflowApi";
import { getTrainingDefaults } from "@/features/ml/config/datasetTrainingDefaults";
import { useMlTrainingRunsStore } from "@/features/ml/state/zustand/mlTrainingRunsStore";
import {
  findOptimalParamsFromRuns,
  type HyperParams,
} from "@/app/ml/util/bayesianOptimizer.util";
import { Modal } from "@/core/views/components/General/Modal";
import {
  buildSweepCombinations,
  validateBatchSizes,
  validateDropouts,
  validateEpochValues,
  validateHiddenDims,
  validateLearningRates,
  validateNumHiddenLayers,
  validateTestSizes,
} from "@/app/ml/trainingSweep.validators";
import {
  calcTrainingTableHeight,
  formatCompletedAt,
  formatMetricNumber,
  TRAINING_RUN_COLUMNS,
  type TrainingMetrics,
  type TrainingRunRow,
} from "@/app/ml/util/trainingRuns.util";

function FieldHelp({ text }: { text: string }) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          title={text}
          aria-label="Field help"
          className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-300 text-[10px] font-bold text-zinc-500 hover:bg-zinc-100"
        >
          i
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="text-xs leading-relaxed text-zinc-700">
        {text}
      </PopoverContent>
    </Popover>
  );
}

type NumericInputSnapshot = {
  epochValuesInput: string;
  batchSizesInput: string;
  learningRatesInput: string;
  testSizesInput: string;
  hiddenDimsInput: string;
  numHiddenLayersInput: string;
  dropoutsInput: string;
};

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomFloat(min: number, max: number, decimals = 4): number {
  const value = Math.random() * (max - min) + min;
  return Number(value.toFixed(decimals));
}

function buildRandomSweepInputs(): NumericInputSnapshot {
  const randomEpochs = [randomInt(40, 180), randomInt(181, 500)]
    .sort((a, b) => a - b)
    .join(",");
  const randomBatchSizes = [randomInt(8, 96), randomInt(97, 200)]
    .sort((a, b) => a - b)
    .join(",");
  const randomLearningRates = [randomFloat(0.0002, 0.003, 4), randomFloat(0.0031, 0.03, 4)]
    .sort((a, b) => a - b)
    .join(",");
  const randomTestSizes = [randomFloat(0.1, 0.3, 2), randomFloat(0.31, 0.45, 2)]
    .sort((a, b) => a - b)
    .join(",");
  const randomHiddenDims = [randomInt(32, 220), randomInt(221, 500)]
    .sort((a, b) => a - b)
    .join(",");
  const randomHiddenLayers = [randomInt(1, 6), randomInt(7, 15)]
    .sort((a, b) => a - b)
    .join(",");
  const randomDropouts = [randomFloat(0.05, 0.2, 2), randomFloat(0.21, 0.45, 2)]
    .sort((a, b) => a - b)
    .join(",");

  return {
    epochValuesInput: randomEpochs,
    batchSizesInput: randomBatchSizes,
    learningRatesInput: randomLearningRates,
    testSizesInput: randomTestSizes,
    hiddenDimsInput: randomHiddenDims,
    numHiddenLayersInput: randomHiddenLayers,
    dropoutsInput: randomDropouts,
  };
}

function applyNumericInputs(
  snapshot: NumericInputSnapshot,
  setters: {
    setEpochValuesInput: (value: string) => void;
    setBatchSizesInput: (value: string) => void;
    setLearningRatesInput: (value: string) => void;
    setTestSizesInput: (value: string) => void;
    setHiddenDimsInput: (value: string) => void;
    setNumHiddenLayersInput: (value: string) => void;
    setDropoutsInput: (value: string) => void;
  }
) {
  setters.setEpochValuesInput(snapshot.epochValuesInput);
  setters.setBatchSizesInput(snapshot.batchSizesInput);
  setters.setLearningRatesInput(snapshot.learningRatesInput);
  setters.setTestSizesInput(snapshot.testSizesInput);
  setters.setHiddenDimsInput(snapshot.hiddenDimsInput);
  setters.setNumHiddenLayersInput(snapshot.numHiddenLayersInput);
  setters.setDropoutsInput(snapshot.dropoutsInput);
}

function parseNumericValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const normalized = trimmed.replace("x10^", "e");
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function metricHigherIsBetter(metricName: string): boolean {
  const normalized = metricName.toLowerCase();
  return (
    normalized.includes("accuracy") ||
    normalized.includes("f1") ||
    normalized.includes("auc") ||
    normalized.includes("precision") ||
    normalized.includes("recall") ||
    normalized.includes("r2")
  );
}

type TensorflowTrainingMode = "mlp" | "linear_glm_baseline";

const TENSORFLOW_MODE_EXPLAINERS: Record<TensorflowTrainingMode, string> = {
  mlp: "MLP (dense network): learns nonlinear feature interactions using hidden layers.",
  linear_glm_baseline:
    "Linear/GLM baseline: single linear head (logistic/linear regression). Fast, interpretable, and best as a baseline.",
};

export default function TensorFlowPage() {
  const {
    datasetOptions,
    selectedDatasetId,
    setSelectedDatasetId,
    isLoading,
    error,
    tableRows,
    tableColumns,
    rowCount,
    totalRowCount,
  } = useMlDatasetOrchestrator();
  const [targetColumn, setTargetColumn] = useState("");
  const [excludeColumnsInput, setExcludeColumnsInput] = useState<string | null>(null);
  const [dateColumnsInput, setDateColumnsInput] = useState<string | null>(null);
  const [task, setTask] = useState<"classification" | "regression" | "auto">("auto");
  const [trainingMode, setTrainingMode] = useState<TensorflowTrainingMode>("linear_glm_baseline");
  const [epochValuesInput, setEpochValuesInput] = useState("60");
  const [testSizesInput, setTestSizesInput] = useState("0.2");
  const [learningRatesInput, setLearningRatesInput] = useState("0.001");
  const [batchSizesInput, setBatchSizesInput] = useState("64");
  const [hiddenDimsInput, setHiddenDimsInput] = useState("128");
  const [numHiddenLayersInput, setNumHiddenLayersInput] = useState("2");
  const [dropoutsInput, setDropoutsInput] = useState("0.1");
  const [runSweepEnabled, setRunSweepEnabled] = useState(false);
  const [savedNumericInputs, setSavedNumericInputs] = useState<NumericInputSnapshot | null>(null);
  const [savedSweepInputs, setSavedSweepInputs] = useState<NumericInputSnapshot | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isDistilling, setIsDistilling] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<{ current: number; total: number }>({
    current: 0,
    total: 0,
  });
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const trainingRuns = useMlTrainingRunsStore((state) => state.trainingRuns);
  const prependTrainingRun = useMlTrainingRunsStore((state) => state.prependTrainingRun);
  const clearTrainingRuns = useMlTrainingRunsStore((state) => state.clearTrainingRuns);
  const [copyRunsStatus, setCopyRunsStatus] = useState<string | null>(null);
  const [optimizerStatus, setOptimizerStatus] = useState<string | null>(null);
  const [distillStatus, setDistillStatus] = useState<string | null>(null);
  const [saveDistilledModel, setSaveDistilledModel] = useState(false);
  const [isOptimalModalOpen, setIsOptimalModalOpen] = useState(false);
  const [pendingOptimalParams, setPendingOptimalParams] = useState<HyperParams | null>(null);
  const [pendingOptimalPrediction, setPendingOptimalPrediction] = useState<{
    metricName: string;
    metricValue: number;
  } | null>(null);
  const [isDistillMetricsModalOpen, setIsDistillMetricsModalOpen] = useState(false);
  const [distillMetrics, setDistillMetrics] = useState<TrainingMetrics | null>(null);
  const [distillModelId, setDistillModelId] = useState<string | null>(null);
  const [distillModelPath, setDistillModelPath] = useState<string | null>(null);
  const defaults = getTrainingDefaults(selectedDatasetId);
  const resolvedExcludeColumnsInput =
    excludeColumnsInput === null
      ? defaults.excludeColumns.join(",")
      : excludeColumnsInput;
  const resolvedDateColumnsInput =
    dateColumnsInput === null ? defaults.dateColumns.join(",") : dateColumnsInput;
  const epochsValidation = useMemo(
    () => validateEpochValues(epochValuesInput),
    [epochValuesInput]
  );
  const testSizesValidation = useMemo(
    () => validateTestSizes(testSizesInput),
    [testSizesInput]
  );
  const learningRatesValidation = useMemo(
    () => validateLearningRates(learningRatesInput),
    [learningRatesInput]
  );
  const batchSizesValidation = useMemo(
    () => validateBatchSizes(batchSizesInput),
    [batchSizesInput]
  );
  const hiddenDimsValidation = useMemo(
    () => validateHiddenDims(hiddenDimsInput),
    [hiddenDimsInput]
  );
  const numHiddenLayersValidation = useMemo(
    () => validateNumHiddenLayers(numHiddenLayersInput),
    [numHiddenLayersInput]
  );
  const dropoutsValidation = useMemo(
    () => validateDropouts(dropoutsInput),
    [dropoutsInput]
  );
  const isLinearBaselineMode = trainingMode === "linear_glm_baseline";

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
      (isLinearBaselineMode ? 1 : hiddenDimsValidation.values.length) *
      (isLinearBaselineMode ? 1 : numHiddenLayersValidation.values.length) *
      (isLinearBaselineMode ? 1 : dropoutsValidation.values.length)
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

  const trainingTableHeight = useMemo(() => {
    return calcTrainingTableHeight(trainingRuns.length);
  }, [trainingRuns.length]);

  const completedRuns = useMemo(() => {
    return trainingRuns.filter((run) => {
      if (String(run.result ?? "") !== "completed") return false;
      const metric = String(run.metric_name ?? "").toLowerCase();
      return metric !== "n/a" && parseNumericValue(run.metric_score) !== null;
    });
  }, [trainingRuns]);

  function onDatasetChange(nextDatasetId: string | null) {
    setSelectedDatasetId(nextDatasetId);
    const nextDefaults = getTrainingDefaults(nextDatasetId);
    setTargetColumn(nextDefaults.targetColumn);
    setTrainingMode("linear_glm_baseline");
    setExcludeColumnsInput(null);
    setTask(nextDefaults.task);
    setEpochValuesInput(String(nextDefaults.epochs));
    setTestSizesInput("0.2");
    setLearningRatesInput("0.001");
    setBatchSizesInput("64");
    setHiddenDimsInput("128");
    setNumHiddenLayersInput("2");
    setDropoutsInput("0.1");
    setRunSweepEnabled(false);
    setSavedNumericInputs(null);
    setSavedSweepInputs(null);
    setTrainingError(null);
    setDateColumnsInput(null);
  }

  function toggleRunSweep(checked: boolean) {
    if (checked) {
      setSavedNumericInputs({
        epochValuesInput,
        batchSizesInput,
        learningRatesInput,
        testSizesInput,
        hiddenDimsInput,
        numHiddenLayersInput,
        dropoutsInput,
      });
      const nextSweep = savedSweepInputs ?? buildRandomSweepInputs();
      setSavedSweepInputs(nextSweep);
      applyNumericInputs(nextSweep, {
        setEpochValuesInput,
        setBatchSizesInput,
        setLearningRatesInput,
        setTestSizesInput,
        setHiddenDimsInput,
        setNumHiddenLayersInput,
        setDropoutsInput,
      });
      setRunSweepEnabled(true);
      return;
    }

    setSavedSweepInputs({
      epochValuesInput,
      batchSizesInput,
      learningRatesInput,
      testSizesInput,
      hiddenDimsInput,
      numHiddenLayersInput,
      dropoutsInput,
    });

    if (savedNumericInputs) {
      applyNumericInputs(savedNumericInputs, {
        setEpochValuesInput,
        setBatchSizesInput,
        setLearningRatesInput,
        setTestSizesInput,
        setHiddenDimsInput,
        setNumHiddenLayersInput,
        setDropoutsInput,
      });
    } else {
      applyNumericInputs(
        {
          epochValuesInput: String(defaults.epochs),
          batchSizesInput: "64",
          learningRatesInput: "0.001",
          testSizesInput: "0.2",
          hiddenDimsInput: "128",
          numHiddenLayersInput: "2",
          dropoutsInput: "0.1",
        },
        {
          setEpochValuesInput,
          setBatchSizesInput,
          setLearningRatesInput,
          setTestSizesInput,
          setHiddenDimsInput,
          setNumHiddenLayersInput,
          setDropoutsInput,
        }
      );
    }
    setRunSweepEnabled(false);
  }

  function reloadSweepValues() {
    const nextSweep = buildRandomSweepInputs();
    setSavedSweepInputs(nextSweep);
    applyNumericInputs(nextSweep, {
      setEpochValuesInput,
      setBatchSizesInput,
      setLearningRatesInput,
      setTestSizesInput,
      setHiddenDimsInput,
      setNumHiddenLayersInput,
      setDropoutsInput,
    });
  }

  async function onTrainClick() {
    if (!selectedDatasetId) {
      setTrainingError("Please select a dataset first.");
      return;
    }
    const resolvedTargetColumn =
      targetColumn.trim() || defaults.targetColumn || tableColumns[0] || "";
    const excludeColumns = resolvedExcludeColumnsInput
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const dateColumns = resolvedDateColumnsInput
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    if (excludeColumns.includes(resolvedTargetColumn.trim())) {
      setTrainingError("Target column cannot also be in excluded columns.");
      return;
    }
    if (dateColumns.includes(resolvedTargetColumn.trim())) {
      setTrainingError("Target column cannot also be in date columns.");
      return;
    }
    const overlap = dateColumns.find((col) => excludeColumns.includes(col));
    if (overlap) {
      setTrainingError(`Column '${overlap}' cannot be in both excluded and date columns.`);
      return;
    }
    if (!resolvedTargetColumn.trim()) {
      setTrainingError("Please provide a target column.");
      return;
    }
    if (!epochsValidation.ok) {
      setTrainingError(epochsValidation.error);
      return;
    }
    if (!testSizesValidation.ok) {
      setTrainingError(testSizesValidation.error);
      return;
    }
    if (!learningRatesValidation.ok) {
      setTrainingError(learningRatesValidation.error);
      return;
    }
    if (!batchSizesValidation.ok) {
      setTrainingError(batchSizesValidation.error);
      return;
    }
    if (!isLinearBaselineMode && !hiddenDimsValidation.ok) {
      setTrainingError(hiddenDimsValidation.error);
      return;
    }
    if (!isLinearBaselineMode && !numHiddenLayersValidation.ok) {
      setTrainingError(numHiddenLayersValidation.error);
      return;
    }
    if (!isLinearBaselineMode && !dropoutsValidation.ok) {
      setTrainingError(dropoutsValidation.error);
      return;
    }

    setIsTraining(true);
    setTrainingError(null);
    const combinations = buildSweepCombinations({
      epochs: epochsValidation.values,
      testSizes: testSizesValidation.values,
      learningRates: learningRatesValidation.values,
      batchSizes: batchSizesValidation.values,
      hiddenDims: isLinearBaselineMode ? [0] : hiddenDimsValidation.values,
      numHiddenLayers: isLinearBaselineMode ? [0] : numHiddenLayersValidation.values,
      dropouts: isLinearBaselineMode ? [0] : dropoutsValidation.values,
    });
    setTrainingProgress({ current: 0, total: combinations.length });

    for (let i = 0; i < combinations.length; i += 1) {
      const combo = combinations[i];
      const result = await trainTensorflowModel({
        dataset_id: selectedDatasetId,
        target_column: resolvedTargetColumn.trim(),
        training_mode: trainingMode,
        save_model: false,
        exclude_columns: excludeColumns,
        date_columns: dateColumns,
        task,
        epochs: combo.epochs,
        batch_size: combo.batchSize,
        learning_rate: combo.learningRate,
        test_size: combo.testSize,
        hidden_dim: isLinearBaselineMode ? 128 : combo.hiddenDim,
        num_hidden_layers: isLinearBaselineMode ? 2 : combo.numHiddenLayers,
        dropout: isLinearBaselineMode ? 0.1 : combo.dropout,
      });
      setTrainingProgress({ current: i + 1, total: combinations.length });

      if (result.status === "error") {
        const failedRow: TrainingRunRow = {
          result: "failed",
          completed_at: formatCompletedAt(),
          epochs: combo.epochs,
          learning_rate: formatMetricNumber(combo.learningRate),
          test_size: formatMetricNumber(combo.testSize),
          batch_size: combo.batchSize,
          hidden_dim: isLinearBaselineMode ? "n/a" : combo.hiddenDim,
          num_hidden_layers: isLinearBaselineMode ? "n/a" : combo.numHiddenLayers,
          dropout: isLinearBaselineMode ? "n/a" : formatMetricNumber(combo.dropout),
          task,
          target_column: resolvedTargetColumn.trim(),
          dataset_id: selectedDatasetId,
          metric_name: "n/a",
          metric_score: "n/a",
          train_loss: "n/a",
          test_loss: "n/a",
          model_id: "n/a",
          model_path: "n/a",
          error: result.error,
        };
        prependTrainingRun(failedRow);
        continue;
      }

      const metrics = (result.metrics ?? {}) as TrainingMetrics;
      const runRow: TrainingRunRow = {
        result: "completed",
        completed_at: formatCompletedAt(),
        epochs: combo.epochs,
        learning_rate: formatMetricNumber(combo.learningRate),
        test_size: formatMetricNumber(combo.testSize),
        batch_size: combo.batchSize,
        hidden_dim: isLinearBaselineMode ? "n/a" : combo.hiddenDim,
        num_hidden_layers: isLinearBaselineMode ? "n/a" : combo.numHiddenLayers,
        dropout: isLinearBaselineMode ? "n/a" : formatMetricNumber(combo.dropout),
        task,
        target_column: resolvedTargetColumn.trim(),
        dataset_id: selectedDatasetId,
        metric_name: metrics.test_metric_name ?? "n/a",
        metric_score: formatMetricNumber(metrics.test_metric_value),
        train_loss: formatMetricNumber(metrics.train_loss),
        test_loss: formatMetricNumber(metrics.test_loss),
        model_id: result.model_id ?? "n/a",
        model_path: result.model_path ?? "n/a",
      };
      prependTrainingRun(runRow);
    }
    setTrainingError(null);
    setIsTraining(false);
    setTrainingProgress({ current: 0, total: 0 });
  }

  function onFindOptimalParamsClick() {
    const optimized = findOptimalParamsFromRuns(trainingRuns);
    if (!optimized) {
      setOptimizerStatus("Need at least 5 completed runs.");
      setTimeout(() => setOptimizerStatus(null), 2500);
      return;
    }
    setPendingOptimalParams(optimized.suggestion);
    setPendingOptimalPrediction({
      metricName: optimized.predictedMetricName,
      metricValue: optimized.predictedMetricValue,
    });
    setIsOptimalModalOpen(true);
    setOptimizerStatus(`Suggestion generated from ${optimized.basedOnRuns} runs.`);
    setTimeout(() => setOptimizerStatus(null), 2500);
  }

  function onApplyOptimalParams() {
    if (!pendingOptimalParams) return;
    setEpochValuesInput(String(pendingOptimalParams.epochs));
    setLearningRatesInput(String(Number(pendingOptimalParams.learning_rate.toPrecision(6))));
    setTestSizesInput(String(Number(pendingOptimalParams.test_size.toPrecision(4))));
    setBatchSizesInput(String(pendingOptimalParams.batch_size));
    setHiddenDimsInput(String(pendingOptimalParams.hidden_dim));
    setNumHiddenLayersInput(String(pendingOptimalParams.num_hidden_layers));
    setDropoutsInput(String(Number(pendingOptimalParams.dropout.toPrecision(4))));
    setRunSweepEnabled(false);
    setIsOptimalModalOpen(false);
    setPendingOptimalPrediction(null);
    setOptimizerStatus("Updated table with suggested values.");
    setTimeout(() => setOptimizerStatus(null), 2500);
  }

  async function onDistillClick() {
    if (!selectedDatasetId) {
      setTrainingError("Please select a dataset first.");
      return;
    }

    const resolvedTargetColumn =
      targetColumn.trim() || defaults.targetColumn || tableColumns[0] || "";
    if (!resolvedTargetColumn.trim()) {
      setTrainingError("Please provide a target column.");
      return;
    }

    const eligibleTeacherRuns = completedRuns.filter((run) => {
      const modelId = String(run.model_id ?? "");
      const modelPath = String(run.model_path ?? "");
      const datasetMatch = String(run.dataset_id ?? "") === selectedDatasetId;
      return datasetMatch && (modelId && modelId !== "n/a" || modelPath && modelPath !== "n/a");
    });

    if (eligibleTeacherRuns.length === 0) {
      setTrainingError("Need at least one completed run with a teacher model for this dataset.");
      return;
    }

    const bestTeacher = [...eligibleTeacherRuns].sort((a, b) => {
      const aMetric = parseNumericValue(a.metric_score) ?? Number.NEGATIVE_INFINITY;
      const bMetric = parseNumericValue(b.metric_score) ?? Number.NEGATIVE_INFINITY;
      const metricName = String(a.metric_name ?? b.metric_name ?? "accuracy");
      const higherIsBetter = metricHigherIsBetter(metricName);
      return higherIsBetter ? bMetric - aMetric : aMetric - bMetric;
    })[0];

    const teacherHidden = parseNumericValue(bestTeacher.hidden_dim) ?? 128;
    const teacherLayers = parseNumericValue(bestTeacher.num_hidden_layers) ?? 2;
    const teacherDropout = parseNumericValue(bestTeacher.dropout) ?? 0.1;
    const teacherEpochs = parseNumericValue(bestTeacher.epochs) ?? 60;
    const teacherBatch = parseNumericValue(bestTeacher.batch_size) ?? 64;
    const teacherLr = parseNumericValue(bestTeacher.learning_rate) ?? 1e-3;
    const teacherTestSize = parseNumericValue(bestTeacher.test_size) ?? 0.2;
    const teacherModelId = String(bestTeacher.model_id ?? "");
    const teacherModelPath = String(bestTeacher.model_path ?? "");

    const excludeColumns = resolvedExcludeColumnsInput
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const dateColumns = resolvedDateColumnsInput
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);

    setIsDistilling(true);
    setTrainingError(null);
    setDistillStatus("Running distillation...");

    const result = await distillTensorflowModel({
      dataset_id: selectedDatasetId,
      target_column: resolvedTargetColumn.trim(),
      training_mode: trainingMode,
      save_model: saveDistilledModel,
      teacher_model_id: teacherModelId && teacherModelId !== "n/a" ? teacherModelId : undefined,
      teacher_model_path: teacherModelPath && teacherModelPath !== "n/a" ? teacherModelPath : undefined,
      exclude_columns: excludeColumns,
      date_columns: dateColumns,
      task,
      epochs: Math.max(30, Math.round(teacherEpochs)),
      batch_size: Math.max(1, Math.round(teacherBatch)),
      learning_rate: teacherLr,
      test_size: teacherTestSize,
      temperature: 2.5,
      alpha: 0.5,
      student_hidden_dim: Math.max(16, Math.round(teacherHidden / 2)),
      student_num_hidden_layers: Math.max(1, Math.min(15, Math.round(teacherLayers - 1))),
      student_dropout: Math.min(0.5, teacherDropout + 0.05),
    });

    if (result.status === "error") {
      setTrainingError(result.error);
      setDistillStatus("Distillation failed.");
      setIsDistilling(false);
      return;
    }

    const metrics = (result.metrics ?? {}) as TrainingMetrics;
    setDistillMetrics(metrics);
    setDistillModelId(result.model_id ?? null);
    setDistillModelPath(result.model_path ?? null);
    setIsDistillMetricsModalOpen(true);
    const distilledRun: TrainingRunRow = {
      result: "distilled",
      completed_at: formatCompletedAt(),
      epochs: Math.max(30, Math.round(teacherEpochs)),
      learning_rate: formatMetricNumber(teacherLr),
      test_size: formatMetricNumber(teacherTestSize),
      batch_size: Math.max(1, Math.round(teacherBatch)),
      hidden_dim: Math.max(16, Math.round(teacherHidden / 2)),
      num_hidden_layers: Math.max(1, Math.min(15, Math.round(teacherLayers - 1))),
      dropout: formatMetricNumber(Math.min(0.5, teacherDropout + 0.05)),
      task,
      target_column: resolvedTargetColumn.trim(),
      dataset_id: selectedDatasetId,
      metric_name: metrics.test_metric_name ?? "n/a",
      metric_score: formatMetricNumber(metrics.test_metric_value),
      train_loss: formatMetricNumber(metrics.train_loss),
      test_loss: formatMetricNumber(metrics.test_loss),
      model_id: result.model_id ?? "n/a",
      model_path: result.model_path ?? "n/a",
      error: "",
    };
    prependTrainingRun(distilledRun);
    setDistillStatus("Distilled student model created.");
    setTimeout(() => setDistillStatus(null), 2500);
    setIsDistilling(false);
  }

  async function onCopyTrainingRuns() {
    if (trainingRuns.length === 0) return;

    const rowsAsTsv = trainingRuns.map((row) =>
      TRAINING_RUN_COLUMNS.map((column) => String(row[column] ?? "")).join("\t")
    );
    const tsv = [TRAINING_RUN_COLUMNS.join("\t"), ...rowsAsTsv].join("\n");

    try {
      await navigator.clipboard.writeText(tsv);
      setCopyRunsStatus("Copied");
      setTimeout(() => setCopyRunsStatus(null), 1500);
    } catch {
      setCopyRunsStatus("Copy failed");
      setTimeout(() => setCopyRunsStatus(null), 2000);
    }
  }

  return (
    <div className="flex min-h-screen flex-row bg-white text-zinc-900">
      <main className="min-w-0 flex-1 py-10">
        <div className="mx-auto flex max-w-5xl flex-col gap-4 px-6">
          <p className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
            Machine Learning with TensorFlow
          </p>
          

          <div className="mt-2 flex max-w-xl flex-col gap-2">
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
              Dataset (CSV/XLS/XLSX)
            </p>
            <CsvDatasetCombobox
              options={datasetOptions}
              selectedId={selectedDatasetId}
              onChange={onDatasetChange}
              emptyMessage={
                error ?? (isLoading ? "Loading datasets..." : "No dataset found.")
              }
            />
            {error ? (
              <p className="text-xs text-red-600">{error}</p>
            ) : null}
          </div>

          <section className="mt-6 rounded-xl border border-zinc-200 bg-zinc-50 p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
              Roadmap Notes
            </p>
            <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm text-zinc-700">
              <li>Linear / GLM baseline: one linear layer maps features directly to output; fast and interpretable baseline.</li>
              <li>MLP (dense): stacked dense layers learn nonlinear feature interactions through backpropagation.</li>
              <li>Wide &amp; Deep: linear branch memorizes sparse patterns while deep branch generalizes to new combinations.</li>
              <li>Entity embeddings: categorical values are learned as dense vectors instead of large one-hot encodings.</li>
              <li>Residual MLP / TabResNet: dense blocks with skip-connections improve deeper-network stability.</li>
              <li>Autoencoder + head: unsupervised encoder learns compressed representation, then supervised head predicts target.</li>
              <li>Quantile / distributional outputs: predicts uncertainty bands or full target distribution, not only point estimate.</li>
              <li>Multi-task learning: one shared trunk with multiple task-specific heads trained jointly.</li>
              <li>Time-aware tabular: transforms date/ordered rows into lag-window features or sequence inputs.</li>
              <li>Tree-teacher distillation: train tree model first, then transfer its behavior into a compact neural student.</li>
              <li>Calibration mode: post-train probability scaling to align confidence with actual outcome frequency.</li>
              <li>Imbalance-aware mode: class weighting/focal objectives to improve minority-class performance.</li>
            </ol>
          </section>

          <section className="rounded-xl border border-zinc-200 bg-white p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
              Training Algorithm
            </p>
            <div className="mt-3 grid max-w-3xl grid-cols-1 gap-3 md:grid-cols-3">
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Algorithm
                  <FieldHelp text="Select which TensorFlow training setup to run for this dataset." />
                </span>
                <select
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={trainingMode}
                  onChange={(event) => setTrainingMode(event.target.value as TensorflowTrainingMode)}
                >
                  <option value="linear_glm_baseline">linear/glm baseline</option>
                  <option value="mlp">mlp (dense)</option>
                </select>
              </label>
              <div className="md:col-span-2 rounded-md border border-blue-100 bg-blue-50 px-3 py-2 text-xs text-blue-900">
                {TENSORFLOW_MODE_EXPLAINERS[trainingMode]}
              </div>
            </div>
          </section>

          <section className="rounded-xl border border-zinc-200 bg-white p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
              Train TensorFlow Model
            </p>
            <div className="mt-3 grid max-w-3xl grid-cols-1 gap-3 md:grid-cols-3">
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Target Column
                  <FieldHelp text="Prediction target (label). This column is removed from model inputs and is what the model learns to predict." />
                </span>
                <select
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={targetColumn}
                  onChange={(event) => setTargetColumn(event.target.value)}
                >
                  <option value="">
                    {defaults.targetColumn || "Select target column"}
                  </option>
                  {tableColumns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Task
                  <FieldHelp text="auto infers classification vs regression from target values. Set explicitly when auto inference might be ambiguous." />
                </span>
                <select
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={task}
                  onChange={(event) =>
                    setTask(event.target.value as "classification" | "regression" | "auto")
                  }
                >
                  <option value="auto">auto</option>
                  <option value="classification">classification</option>
                  <option value="regression">regression</option>
                </select>
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Epoch Values
                  <FieldHelp text="Number of full passes over training data. Higher can improve fit but may overfit. Range: 1-500." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={epochValuesInput}
                  onChange={(event) => setEpochValuesInput(event.target.value)}
                  placeholder="e.g. 10,20,50,100,200"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Batch Sizes
                  <FieldHelp text="Rows processed per optimizer step. Larger batches are faster but can generalize differently. Range: 1-200." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={batchSizesInput}
                  onChange={(event) => setBatchSizesInput(event.target.value)}
                  placeholder="e.g. 32,64,128"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Learning Rates
                  <FieldHelp text="Optimizer step size. Too high can diverge, too low can train slowly. Valid range: >0 and <=1." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={learningRatesInput}
                  onChange={(event) => setLearningRatesInput(event.target.value)}
                  placeholder="e.g. 0.001,0.0005"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Test Sizes
                  <FieldHelp text="Fraction held out for evaluation. Example 0.2 means 20% test split. Valid range: >0 and <1." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={testSizesInput}
                  onChange={(event) => setTestSizesInput(event.target.value)}
                  placeholder="e.g. 0.2,0.3"
                />
              </label>
              <label className={`flex flex-col gap-1 text-xs ${isLinearBaselineMode ? "text-zinc-400" : "text-zinc-600"}`}>
                <span className="inline-flex items-center gap-1">
                  Hidden Dims
                  <FieldHelp text="Width of each hidden layer in the MLP. Larger values increase model capacity and cost. Range: 8-500." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900 disabled:bg-zinc-100"
                  value={hiddenDimsInput}
                  onChange={(event) => setHiddenDimsInput(event.target.value)}
                  placeholder="e.g. 128,256"
                  disabled={isLinearBaselineMode}
                />
              </label>
              <label className={`flex flex-col gap-1 text-xs ${isLinearBaselineMode ? "text-zinc-400" : "text-zinc-600"}`}>
                <span className="inline-flex items-center gap-1">
                  Hidden Layers
                  <FieldHelp text="Number of hidden layers in the MLP. More layers can model complex patterns but may overfit. Range: 1-15." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900 disabled:bg-zinc-100"
                  value={numHiddenLayersInput}
                  onChange={(event) => setNumHiddenLayersInput(event.target.value)}
                  placeholder="e.g. 2,3,4"
                  disabled={isLinearBaselineMode}
                />
              </label>
              <label className={`flex flex-col gap-1 text-xs ${isLinearBaselineMode ? "text-zinc-400" : "text-zinc-600"}`}>
                <span className="inline-flex items-center gap-1">
                  Dropouts
                  <FieldHelp text="Dropout probability per hidden layer (0 to 0.9). Helps regularization; too high can underfit." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900 disabled:bg-zinc-100"
                  value={dropoutsInput}
                  onChange={(event) => setDropoutsInput(event.target.value)}
                  placeholder="e.g. 0.1,0.2"
                  disabled={isLinearBaselineMode}
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Exclude Columns
                  <FieldHelp text="Columns to drop from training features (for example IDs) as 
                  they are simply noise.  Comma-separated list." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={resolvedExcludeColumnsInput}
                  onChange={(event) => setExcludeColumnsInput(event.target.value)}
                  placeholder="e.g. customerID,Order,PID"
                />
                <span className="text-[11px] text-zinc-500">
                  Preloaded: {defaults.excludeColumns.length > 0 ? defaults.excludeColumns.join(", ") : "(none)"}
                </span>
              </label>
              <label className="flex flex-col gap-1 text-xs text-zinc-600">
                <span className="inline-flex items-center gap-1">
                  Date Columns
                  <FieldHelp text="Columns parsed as dates and expanded into engineered numeric features (month/day/week/cyclical terms)." />
                </span>
                <input
                  className="rounded-md border border-zinc-300 px-2 py-1 text-sm text-zinc-900"
                  value={resolvedDateColumnsInput}
                  onChange={(event) => setDateColumnsInput(event.target.value)}
                  placeholder="e.g. Date"
                />
                <span className="text-[11px] text-zinc-500">
                  Preloaded: {defaults.dateColumns.length > 0 ? defaults.dateColumns.join(", ") : "(none)"}
                </span>
              </label>
            </div>
            <div className="mt-2 grid max-w-3xl grid-cols-1 gap-1 text-xs text-zinc-500 md:grid-cols-2">
              <p className={epochsValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Epochs: {epochsValidation.ok ? `${epochsValidation.values.join(", ")}` : epochsValidation.error}
              </p>
              <p className={batchSizesValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Batch sizes: {batchSizesValidation.ok ? `${batchSizesValidation.values.join(", ")}` : batchSizesValidation.error}
              </p>
              <p className={learningRatesValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Learning rates: {learningRatesValidation.ok ? `${learningRatesValidation.values.join(", ")}` : learningRatesValidation.error}
              </p>
              <p className={testSizesValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Test sizes: {testSizesValidation.ok ? `${testSizesValidation.values.join(", ")}` : testSizesValidation.error}
              </p>
              <p className={isLinearBaselineMode || hiddenDimsValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Hidden dims: {isLinearBaselineMode ? "n/a (linear baseline)" : hiddenDimsValidation.ok ? `${hiddenDimsValidation.values.join(", ")}` : hiddenDimsValidation.error}
              </p>
              <p className={isLinearBaselineMode || numHiddenLayersValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Hidden layers: {isLinearBaselineMode ? "n/a (linear baseline)" : numHiddenLayersValidation.ok ? `${numHiddenLayersValidation.values.join(", ")}` : numHiddenLayersValidation.error}
              </p>
              <p className={isLinearBaselineMode || dropoutsValidation.ok ? "text-zinc-500" : "text-red-600"}>
                Dropouts: {isLinearBaselineMode ? "n/a (linear baseline)" : dropoutsValidation.ok ? `${dropoutsValidation.values.join(", ")}` : dropoutsValidation.error}
              </p>
            </div>
            <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    className="rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-400"
                    onClick={onTrainClick}
                    disabled={isTraining || isDistilling || !selectedDatasetId || plannedRunCount === 0}
                  >
                    {isTraining
                      ? `Training ${trainingProgress.current}/${trainingProgress.total}...`
                      : "Train Model"}
                  </button>
                  <div className="flex flex-col gap-1">
                    <p className="text-xs text-zinc-500">
                      Dataset: <code>{selectedDatasetId ?? "none"}</code>
                    </p>
                    <p className="text-xs font-semibold text-red-600">
                      Planned runs: {plannedRunCount}
                    </p>
                  </div>
                </div>
                <div className="border-t border-zinc-200 pt-3">
                  <p className="mb-2 text-xs text-zinc-600">
                    <span className="font-semibold text-zinc-700">Bayesian Optimization</span>
                    : uses completed runs to suggest the next promising hyperparameter combination.
                  </p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    className="rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700 disabled:cursor-not-allowed disabled:text-zinc-400"
                    onClick={onFindOptimalParamsClick}
                    disabled={isTraining || isDistilling || completedRuns.length < 5}
                  >
                    Find Optimal Params
                  </button>
                  <FieldHelp text="Uses a Bayesian-style search over your previous runs to suggest the next hyperparameter set likely to improve model performance. Requires at least 5 completed runs." />
                  {optimizerStatus ? (
                    <span className="text-xs text-zinc-500">{optimizerStatus}</span>
                  ) : null}
                </div>
                </div>
                {/* Distill section intentionally hidden for now.
                <div className="mt-2 border-t border-zinc-200 pt-3">
                  <div className="mt-2 flex items-start gap-2">
                    <button
                      type="button"
                      className="min-w-[132px] whitespace-nowrap rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-400"
                      onClick={onDistillClick}
                      disabled={isTraining || isDistilling || completedRuns.length === 0}
                    >
                      {isDistilling ? "Distilling..." : "Distill Model"}
                    </button>
                    <p className="text-xs text-zinc-600">
                      <span className="font-semibold text-zinc-700">Knowledge Distillation:</span>{" "}
                      train a smaller student model to mimic a stronger teacher model while preserving
                      similar performance.
                    </p>
                  </div>
                  {distillStatus ? (
                    <p className="mt-1 text-xs text-zinc-500">{distillStatus}</p>
                  ) : null}
                  <label className="mt-2 inline-flex items-center gap-2 text-xs text-zinc-600">
                    <input
                      type="checkbox"
                      className="h-3.5 w-3.5 accent-zinc-900"
                      checked={saveDistilledModel}
                      onChange={(event) => setSaveDistilledModel(event.target.checked)}
                    />
                    Save distilled model to <code>ai/ml/tensorflow_artifacts</code>
                  </label>
                </div>
                */}
              </div>
              <div className="border-zinc-200 md:border-l md:pl-4">
                <label className="inline-flex items-center gap-2 text-sm font-medium text-zinc-700">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-zinc-900"
                    checked={runSweepEnabled}
                    onChange={(event) => toggleRunSweep(event.target.checked)}
                  />
                  Run Sweep
                  <FieldHelp text="A sweep runs multiple training experiments with different parameter combinations so you can compare results and find better-performing settings." />
                </label>
                <div className="mt-2">
                  <button
                    type="button"
                    className="rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs font-medium text-zinc-700 disabled:cursor-not-allowed disabled:text-zinc-400"
                    onClick={reloadSweepValues}
                    disabled={!runSweepEnabled || isTraining}
                  >
                    Reload
                  </button>
                </div>
                <ul className="mt-1 list-disc space-y-1 pl-4 text-xs text-zinc-500">
                  <li>Toggle ON to use sweep values.</li>
                  <li>Toggle OFF restores your previous non-sweep values.</li>
                  <li>Use Reload to generate a fresh random sweep set.</li>
                </ul>
              </div>
            </div>
            {trainingError ? (
              <p className="mt-3 text-xs text-red-600">{trainingError}</p>
            ) : null}
            <div className="mt-4 border-t border-zinc-200 pt-4">
              <div className="mb-2 flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
                  Training Runs
                </p>
                <div className="flex items-center gap-2">
                  {copyRunsStatus ? (
                    <span className="text-xs text-zinc-500">{copyRunsStatus}</span>
                  ) : null}
                  <button
                    type="button"
                    className="rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs font-medium text-zinc-700 disabled:cursor-not-allowed disabled:text-zinc-400"
                    onClick={onCopyTrainingRuns}
                    disabled={trainingRuns.length === 0}
                  >
                    Copy Results
                  </button>
                  <button
                    type="button"
                    className="rounded-md bg-zinc-900 px-2 py-1 text-xs font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-400"
                    onClick={clearTrainingRuns}
                    disabled={trainingRuns.length === 0}
                  >
                    Clear Runs
                  </button>
                </div>
              </div>
              {trainingRuns.length === 0 ? (
                <p className="text-xs text-zinc-500">
                  No runs yet. Train once to populate the results table.
                </p>
              ) : (
                <DataTable
                  rows={trainingRuns}
                  columns={[...TRAINING_RUN_COLUMNS]}
                  height={trainingTableHeight}
                  maxWidth={980}
                />
              )}
            </div>
          </section>

          <details
            className="rounded-xl border border-zinc-200 bg-white p-4"
            open
          >
            <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-zinc-500">
              Dataset Table Preview
            </summary>
            <p className="mt-3 text-xs text-zinc-500">
              Showing {rowCount} rows
              {totalRowCount > rowCount ? ` of ${totalRowCount}` : ""} for{" "}
              <code>{selectedDatasetId ?? "no selection"}</code>.
            </p>
            <div className="mt-3">
              <DataTable rows={tableRows} columns={tableColumns} height={360} maxWidth={980} />
            </div>
          </details>
        </div>
      </main>
      {/* <div className="sticky top-0 h-screen w-[420px] shrink-0 overflow-hidden">
        <CopilotSidebar mode="ag-ui" />
      </div> */}
      <Modal
        isOpen={isOptimalModalOpen}
        onClose={() => setIsOptimalModalOpen(false)}
        title="Bayesian Optimization Suggestion"
      >
        <div className="space-y-4 p-1">
          <p className="text-sm text-zinc-600">
            Suggested next hyperparameters for better accuracy based on completed runs.

          </p>
          <div className="grid grid-cols-1 gap-2 text-sm text-zinc-800 md:grid-cols-2">
            <p>epochs: <span className="font-semibold">{pendingOptimalParams?.epochs ?? "n/a"}</span></p>
            <p>learning_rate: <span className="font-semibold">{pendingOptimalParams ? Number(pendingOptimalParams.learning_rate.toPrecision(6)) : "n/a"}</span></p>
            <p>test_size: <span className="font-semibold">{pendingOptimalParams ? Number(pendingOptimalParams.test_size.toPrecision(4)) : "n/a"}</span></p>
            <p>batch_size: <span className="font-semibold">{pendingOptimalParams?.batch_size ?? "n/a"}</span></p>
            <p>hidden_dim: <span className="font-semibold">{pendingOptimalParams?.hidden_dim ?? "n/a"}</span></p>
            <p>num_hidden_layers: <span className="font-semibold">{pendingOptimalParams?.num_hidden_layers ?? "n/a"}</span></p>
            <p>dropout: <span className="font-semibold">{pendingOptimalParams ? Number(pendingOptimalParams.dropout.toPrecision(4)) : "n/a"}</span></p>
          </div>
          {pendingOptimalPrediction ? (
            <p className="text-sm font-semibold text-red-600">
              Predicted: {pendingOptimalPrediction.metricName} ≈{" "}
              {formatMetricNumber(pendingOptimalPrediction.metricValue)}
            </p>
          ) : null}
          <div className="flex items-center justify-end gap-2 pt-2">
            <button
              type="button"
              className="rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700"
              onClick={() => setIsOptimalModalOpen(false)}
            >
              Cancel
            </button>
            <button
              type="button"
              className="rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white"
              onClick={onApplyOptimalParams}
              disabled={!pendingOptimalParams}
            >
              Update Table With Values
            </button>
          </div>
        </div>
      </Modal>
      <Modal
        isOpen={isDistillMetricsModalOpen}
        onClose={() => setIsDistillMetricsModalOpen(false)}
        title="Distillation Metrics"
      >
        <div className="space-y-3 p-1 text-sm text-zinc-700">
          <p>
            metric_name:{" "}
            <span className="font-semibold text-zinc-900">
              {distillMetrics?.test_metric_name ?? "n/a"}
            </span>
          </p>
          <p>
            metric_score:{" "}
            <span className="font-semibold text-zinc-900">
              {formatMetricNumber(distillMetrics?.test_metric_value)}
            </span>
          </p>
          <p>
            train_loss:{" "}
            <span className="font-semibold text-zinc-900">
              {formatMetricNumber(distillMetrics?.train_loss)}
            </span>
          </p>
          <p>
            test_loss:{" "}
            <span className="font-semibold text-zinc-900">
              {formatMetricNumber(distillMetrics?.test_loss)}
            </span>
          </p>
          {distillModelId || distillModelPath ? (
            <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-600">
              <p>
                model_id: <span className="font-medium text-zinc-800">{distillModelId ?? "n/a"}</span>
              </p>
              <p className="mt-1 break-all">
                model_path: <span className="font-medium text-zinc-800">{distillModelPath ?? "n/a"}</span>
              </p>
            </div>
          ) : (
            <p className="text-xs text-zinc-500">
              Model files were not saved for this run.
            </p>
          )}
          <div className="flex justify-end pt-1">
            <button
              type="button"
              className="rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white"
              onClick={() => setIsDistillMetricsModalOpen(false)}
            >
              Close
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
