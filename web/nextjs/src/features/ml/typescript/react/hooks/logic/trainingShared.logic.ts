import { findOptimalParamsFromRuns } from "@/features/ml/typescript/utils/bayesianOptimizer.util";
import { TRAINING_RUN_COLUMNS } from "@/features/ml/typescript/utils/trainingRuns.util";
import {
  applyNumericInputs,
  buildRandomSweepInputs,
} from "@/features/ml/typescript/utils/trainingUiShared";
import type {
  HandleApplyOptimalParamsUi,
  HandleCopyTrainingRunsArgs,
  HandleFindOptimalParamsArgs,
  NumericInputState,
  TrainingSharedRuntime,
} from "@/features/ml/__types__/typescript/react/hooks/trainingShared.types";
type ClipboardWriteError = {
  code: "CLIPBOARD_WRITE_FAILED";
  message: string;
};

function getDefaultTrainingSharedRuntime(): TrainingSharedRuntime {
  return {
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
}

/**
 * Maps unknown clipboard failures into a typed, assertion-friendly error shape.
 * @param params - Required parameters.
 * @returns Typed clipboard write error.
 */
function toClipboardWriteError(
  { error }: { error: unknown },
  {}: Record<string, never> = {}
): ClipboardWriteError {
  if (error instanceof Error && error.message.trim().length > 0) {
    return {
      code: "CLIPBOARD_WRITE_FAILED",
      message: error.message,
    };
  }
  return {
    code: "CLIPBOARD_WRITE_FAILED",
    message: "Clipboard write failed.",
  };
}

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
      const nextSweep = ui.savedSweepInputs ?? buildRandomSweepInputs({});
      ui.setSavedSweepInputs(nextSweep);
      applyNumericInputs({ snapshot: nextSweep, setters: {
        setEpochValuesInput: ui.setEpochValuesInput,
        setBatchSizesInput: ui.setBatchSizesInput,
        setLearningRatesInput: ui.setLearningRatesInput,
        setTestSizesInput: ui.setTestSizesInput,
        setHiddenDimsInput: ui.setHiddenDimsInput,
        setNumHiddenLayersInput: ui.setNumHiddenLayersInput,
        setDropoutsInput: ui.setDropoutsInput,
      } });
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
      applyNumericInputs({ snapshot: ui.savedNumericInputs, setters: {
        setEpochValuesInput: ui.setEpochValuesInput,
        setBatchSizesInput: ui.setBatchSizesInput,
        setLearningRatesInput: ui.setLearningRatesInput,
        setTestSizesInput: ui.setTestSizesInput,
        setHiddenDimsInput: ui.setHiddenDimsInput,
        setNumHiddenLayersInput: ui.setNumHiddenLayersInput,
        setDropoutsInput: ui.setDropoutsInput,
      } });
    } else {
      applyNumericInputs({
        snapshot: {
          epochValuesInput: String(defaultEpochs),
          batchSizesInput: "64",
          learningRatesInput: "0.001",
          testSizesInput: "0.2",
          hiddenDimsInput: "128",
          numHiddenLayersInput: "2",
          dropoutsInput: "0.1",
        },
        setters: {
          setEpochValuesInput: ui.setEpochValuesInput,
          setBatchSizesInput: ui.setBatchSizesInput,
          setLearningRatesInput: ui.setLearningRatesInput,
          setTestSizesInput: ui.setTestSizesInput,
          setHiddenDimsInput: ui.setHiddenDimsInput,
          setNumHiddenLayersInput: ui.setNumHiddenLayersInput,
          setDropoutsInput: ui.setDropoutsInput,
        },
      });
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
    const nextSweep = buildRandomSweepInputs({});
    ui.setSavedSweepInputs(nextSweep);
    applyNumericInputs({ snapshot: nextSweep, setters: {
      setEpochValuesInput: ui.setEpochValuesInput,
      setBatchSizesInput: ui.setBatchSizesInput,
      setLearningRatesInput: ui.setLearningRatesInput,
      setTestSizesInput: ui.setTestSizesInput,
      setHiddenDimsInput: ui.setHiddenDimsInput,
      setNumHiddenLayersInput: ui.setNumHiddenLayersInput,
      setDropoutsInput: ui.setDropoutsInput,
    } });
  };
}

/**
 * Generates and stores an optimizer suggestion from completed runs.
 */
export function handleFindOptimalParams({
  trainingRuns,
  ui,
}: HandleFindOptimalParamsArgs, {
  runtime = getDefaultTrainingSharedRuntime(),
}: { runtime?: TrainingSharedRuntime } = {}) {
  const optimized = findOptimalParamsFromRuns({ rows: trainingRuns });
  if (!optimized) {
    ui.setOptimizerStatus("Need at least 5 completed runs for the specific algorithm.");
    runtime.schedule(() => ui.setOptimizerStatus(null), 2500);
    return;
  }
  ui.setPendingOptimalParams(optimized.suggestion);
  ui.setPendingOptimalPrediction({
    metricName: optimized.predictedMetricName,
    metricValue: optimized.predictedMetricValue,
  });
  ui.setIsOptimalModalOpen(true);
  ui.setOptimizerStatus(`Suggestion generated from ${optimized.basedOnRuns} runs.`);
  runtime.schedule(() => ui.setOptimizerStatus(null), 2500);
}

/**
 * Applies the pending optimizer suggestion to numeric input fields.
 */
export function handleApplyOptimalParams({
  ui,
}: {
  ui: HandleApplyOptimalParamsUi;
}, {
  runtime = getDefaultTrainingSharedRuntime(),
}: { runtime?: TrainingSharedRuntime } = {}) {
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
  runtime.schedule(() => ui.setOptimizerStatus(null), 2500);
}

/**
 * Copies the training-runs table to clipboard in TSV format.
 */
export async function handleCopyTrainingRuns({
  trainingRuns,
  setCopyRunsStatus,
}: HandleCopyTrainingRunsArgs, {
  runtime = getDefaultTrainingSharedRuntime(),
}: { runtime?: TrainingSharedRuntime } = {}) {
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
    await runtime.writeClipboardText(tsv);
    setCopyRunsStatus("Copied");
    runtime.schedule(() => setCopyRunsStatus(null), 1500);
  } catch (error) {
    void toClipboardWriteError({ error });
    setCopyRunsStatus("Copy failed");
    runtime.schedule(() => setCopyRunsStatus(null), 2000);
  }
}
