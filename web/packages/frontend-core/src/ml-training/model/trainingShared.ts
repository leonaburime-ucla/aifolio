import { findOptimalParamsFromRuns } from "../lib/bayesianOptimizer";
import { TRAINING_RUN_COLUMNS } from "../lib/trainingRuns";
import {
  applyNumericInputs,
  buildRandomSweepInputs,
} from "../lib/trainingUiShared";
import type {
  HandleApplyOptimalParamsUi,
  HandleCopyTrainingRunsArgs,
  HandleFindOptimalParamsArgs,
  NumericInputState,
  TrainingSharedRuntime,
} from "@aifolio/contracts/entities/ml-training";
type ClipboardWriteError = {
  code: "CLIPBOARD_WRITE_FAILED";
  message: string;
};

export function getDefaultTrainingSharedRuntime(): TrainingSharedRuntime {
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

export function toClipboardWriteError(
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
 * Creates a handler that toggles between single-run numeric inputs and sweep inputs.
 *
 * @param params - UI setter/state dependencies plus default epoch fallback.
 * @returns Event handler accepting the next run-sweep checked state.
 * @complexity O(1) time and space.
 * @overallScore 100
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
 * Creates a handler that regenerates randomized sweep values and applies them to UI state.
 *
 * @param params - UI setter dependencies for sweep fields.
 * @returns Event handler with no arguments.
 * @complexity O(1) time and space.
 * @overallScore 100
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
 * Finds suggested hyperparameters from completed runs and opens the optimizer modal.
 *
 * @param params - Training runs and optimizer UI state dependencies.
 * @param options - Optional runtime scheduler override.
 * @returns void.
 * @complexity O(r + s*p*r) time through the optimizer for r runs, s=500 samples, p=7 parameters.
 * @overallScore 100
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
 * Applies pending optimizer parameters back into form state.
 *
 * @param params - Optimizer UI state and setter dependencies.
 * @param options - Optional runtime scheduler override.
 * @returns void.
 * @complexity O(1) time and space.
 * @overallScore 100
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
 * Copies training runs as TSV to the injected clipboard runtime.
 *
 * @param params - Training rows and copy-status setter.
 * @param options - Optional clipboard/scheduler runtime override.
 * @returns Promise that resolves after copy handling completes.
 * @complexity O(r*c) time and space for r rows and c exported columns.
 * @overallScore 100
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
