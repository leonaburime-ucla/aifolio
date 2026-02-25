import { findOptimalParamsFromRuns, type HyperParams } from "@/features/ml/utils/bayesianOptimizer.util";
import { TRAINING_RUN_COLUMNS, type TrainingRunRow } from "@/features/ml/utils/trainingRuns.util";
import {
  applyNumericInputs,
  buildRandomSweepInputs,
  type NumericInputSnapshot,
} from "@/features/ml/utils/trainingUiShared";

type NumericInputSetters = {
  setEpochValuesInput: (value: string) => void;
  setBatchSizesInput: (value: string) => void;
  setLearningRatesInput: (value: string) => void;
  setTestSizesInput: (value: string) => void;
  setHiddenDimsInput: (value: string) => void;
  setNumHiddenLayersInput: (value: string) => void;
  setDropoutsInput: (value: string) => void;
};

type NumericInputState = NumericInputSnapshot &
  NumericInputSetters & {
    savedNumericInputs: NumericInputSnapshot | null;
    setSavedNumericInputs: (value: NumericInputSnapshot | null) => void;
    savedSweepInputs: NumericInputSnapshot | null;
    setSavedSweepInputs: (value: NumericInputSnapshot | null) => void;
    setRunSweepEnabled: (value: boolean) => void;
  };

type OptimizerUiState = {
  pendingOptimalParams: HyperParams | null;
  setPendingOptimalParams: (value: HyperParams | null) => void;
  setPendingOptimalPrediction: (
    value: {
      metricName: string;
      metricValue: number;
    } | null
  ) => void;
  setIsOptimalModalOpen: (value: boolean) => void;
  setOptimizerStatus: (value: string | null) => void;
};

/**
 * Creates a run-sweep toggle handler shared by ML training hooks.
 */
export function createToggleRunSweepHandler({
  ui,
  defaultEpochs,
}: {
  ui: NumericInputState;
  defaultEpochs: number;
}) {
  return function toggleRunSweep(checked: boolean) {
    if (checked) {
      ui.setSavedNumericInputs({
        epochValuesInput: ui.epochValuesInput,
        batchSizesInput: ui.batchSizesInput,
        learningRatesInput: ui.learningRatesInput,
        testSizesInput: ui.testSizesInput,
        hiddenDimsInput: ui.hiddenDimsInput,
        numHiddenLayersInput: ui.numHiddenLayersInput,
        dropoutsInput: ui.dropoutsInput,
      });
      const nextSweep = ui.savedSweepInputs ?? buildRandomSweepInputs();
      ui.setSavedSweepInputs(nextSweep);
      applyNumericInputs(nextSweep, {
        setEpochValuesInput: ui.setEpochValuesInput,
        setBatchSizesInput: ui.setBatchSizesInput,
        setLearningRatesInput: ui.setLearningRatesInput,
        setTestSizesInput: ui.setTestSizesInput,
        setHiddenDimsInput: ui.setHiddenDimsInput,
        setNumHiddenLayersInput: ui.setNumHiddenLayersInput,
        setDropoutsInput: ui.setDropoutsInput,
      });
      ui.setRunSweepEnabled(true);
      return;
    }

    ui.setSavedSweepInputs({
      epochValuesInput: ui.epochValuesInput,
      batchSizesInput: ui.batchSizesInput,
      learningRatesInput: ui.learningRatesInput,
      testSizesInput: ui.testSizesInput,
      hiddenDimsInput: ui.hiddenDimsInput,
      numHiddenLayersInput: ui.numHiddenLayersInput,
      dropoutsInput: ui.dropoutsInput,
    });

    if (ui.savedNumericInputs) {
      applyNumericInputs(ui.savedNumericInputs, {
        setEpochValuesInput: ui.setEpochValuesInput,
        setBatchSizesInput: ui.setBatchSizesInput,
        setLearningRatesInput: ui.setLearningRatesInput,
        setTestSizesInput: ui.setTestSizesInput,
        setHiddenDimsInput: ui.setHiddenDimsInput,
        setNumHiddenLayersInput: ui.setNumHiddenLayersInput,
        setDropoutsInput: ui.setDropoutsInput,
      });
    } else {
      applyNumericInputs(
        {
          epochValuesInput: String(defaultEpochs),
          batchSizesInput: "64",
          learningRatesInput: "0.001",
          testSizesInput: "0.2",
          hiddenDimsInput: "128",
          numHiddenLayersInput: "2",
          dropoutsInput: "0.1",
        },
        {
          setEpochValuesInput: ui.setEpochValuesInput,
          setBatchSizesInput: ui.setBatchSizesInput,
          setLearningRatesInput: ui.setLearningRatesInput,
          setTestSizesInput: ui.setTestSizesInput,
          setHiddenDimsInput: ui.setHiddenDimsInput,
          setNumHiddenLayersInput: ui.setNumHiddenLayersInput,
          setDropoutsInput: ui.setDropoutsInput,
        }
      );
    }
    ui.setRunSweepEnabled(false);
  };
}

/**
 * Creates a sweep-value reload handler shared by ML training hooks.
 */
export function createReloadSweepValuesHandler({
  ui,
}: {
  ui: Pick<
    NumericInputState,
    | "setSavedSweepInputs"
    | "setEpochValuesInput"
    | "setBatchSizesInput"
    | "setLearningRatesInput"
    | "setTestSizesInput"
    | "setHiddenDimsInput"
    | "setNumHiddenLayersInput"
    | "setDropoutsInput"
  >;
}) {
  return function reloadSweepValues() {
    const nextSweep = buildRandomSweepInputs();
    ui.setSavedSweepInputs(nextSweep);
    applyNumericInputs(nextSweep, {
      setEpochValuesInput: ui.setEpochValuesInput,
      setBatchSizesInput: ui.setBatchSizesInput,
      setLearningRatesInput: ui.setLearningRatesInput,
      setTestSizesInput: ui.setTestSizesInput,
      setHiddenDimsInput: ui.setHiddenDimsInput,
      setNumHiddenLayersInput: ui.setNumHiddenLayersInput,
      setDropoutsInput: ui.setDropoutsInput,
    });
  };
}

/**
 * Generates and stores an optimizer suggestion from completed runs.
 */
export function handleFindOptimalParams({
  trainingRuns,
  ui,
}: {
  trainingRuns: TrainingRunRow[];
  ui: OptimizerUiState;
}) {
  const optimized = findOptimalParamsFromRuns(trainingRuns);
  if (!optimized) {
    ui.setOptimizerStatus("Need at least 5 completed runs for the specific algorithm.");
    setTimeout(() => ui.setOptimizerStatus(null), 2500);
    return;
  }
  ui.setPendingOptimalParams(optimized.suggestion);
  ui.setPendingOptimalPrediction({
    metricName: optimized.predictedMetricName,
    metricValue: optimized.predictedMetricValue,
  });
  ui.setIsOptimalModalOpen(true);
  ui.setOptimizerStatus(`Suggestion generated from ${optimized.basedOnRuns} runs.`);
  setTimeout(() => ui.setOptimizerStatus(null), 2500);
}

/**
 * Applies the pending optimizer suggestion to numeric input fields.
 */
export function handleApplyOptimalParams({
  ui,
}: {
  ui: OptimizerUiState &
  Pick<
    NumericInputSetters,
    | "setEpochValuesInput"
    | "setLearningRatesInput"
    | "setTestSizesInput"
    | "setBatchSizesInput"
    | "setHiddenDimsInput"
    | "setNumHiddenLayersInput"
    | "setDropoutsInput"
  > & {
    setRunSweepEnabled: (value: boolean) => void;
  };
}) {
  if (!ui.pendingOptimalParams) return;
  ui.setEpochValuesInput(String(ui.pendingOptimalParams.epochs));
  ui.setLearningRatesInput(String(Number(ui.pendingOptimalParams.learning_rate.toPrecision(6))));
  ui.setTestSizesInput(String(Number(ui.pendingOptimalParams.test_size.toPrecision(4))));
  ui.setBatchSizesInput(String(ui.pendingOptimalParams.batch_size));
  ui.setHiddenDimsInput(String(ui.pendingOptimalParams.hidden_dim));
  ui.setNumHiddenLayersInput(String(ui.pendingOptimalParams.num_hidden_layers));
  ui.setDropoutsInput(String(Number(ui.pendingOptimalParams.dropout.toPrecision(4))));
  ui.setRunSweepEnabled(false);
  ui.setIsOptimalModalOpen(false);
  ui.setPendingOptimalPrediction(null);
  ui.setOptimizerStatus("Updated table with suggested values.");
  setTimeout(() => ui.setOptimizerStatus(null), 2500);
}

/**
 * Copies the training-runs table to clipboard in TSV format.
 */
export async function handleCopyTrainingRuns({
  trainingRuns,
  setCopyRunsStatus,
}: {
  trainingRuns: TrainingRunRow[];
  setCopyRunsStatus: (value: string | null) => void;
}) {
  if (trainingRuns.length === 0) return;

  const rowsAsTsv = trainingRuns.map((row) =>
    TRAINING_RUN_COLUMNS.map((column) => {
      if (column === "distill_action") {
        const value = String(row[column] ?? "").trim();
        return value.length > 0 ? value : "Not Available";
      }
      return String(row[column] ?? "");
    }).join("\t")
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
