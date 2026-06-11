import type {
  MlFormPatch,
  MlFormPatchBindings,
  PytorchBridgeApplyResult,
  PytorchBridgePatch,
  PytorchBridgePatchBindings,
  TensorflowBridgeApplyResult,
  TensorflowBridgePatch,
  TensorflowBridgePatchBindings,
} from "@aifolio/contracts/entities/ml-training";

/**
 * Normalizes scalar and array patch values into the comma-separated format used
 * by ML form text controls.
 *
 * @param value - Raw tool-call patch value.
 * @returns Trimmed comma-separated value for a controlled input.
 * @complexity O(n) time for array values, O(1) space besides the output string.
 * @overallScore 100
 */
export function toBridgeCsv(value: unknown): string {
  if (Array.isArray(value)) {
    return value
      .map((item) => String(item).trim())
      .filter(Boolean)
      .join(",");
  }
  return String(value ?? "").trim();
}

/**
 * Applies a framework-agnostic ML form patch to injected setter bindings.
 *
 * @param patch - Canonical ML form patch payload.
 * @param bindings - Setter bindings and current toggle state for the active form.
 * @returns Applied and skipped patch keys.
 * @complexity O(k) time and O(k) space where k is the number of keys in patch.
 * @overallScore 100
 */
export function applyMlFormBridgePatch<TMode extends string>(
  patch: MlFormPatch<TMode>,
  bindings: MlFormPatchBindings<TMode>
) {
  const applied: string[] = [];
  const handled = new Set<string>();

  if (patch.dataset_id !== undefined) {
    bindings.setDatasetId(String(patch.dataset_id));
    applied.push("dataset_id");
    handled.add("dataset_id");
  }
  if (patch.training_mode !== undefined) {
    bindings.setTrainingMode(patch.training_mode);
    applied.push("training_mode");
    handled.add("training_mode");
  }
  if (patch.target_column !== undefined) {
    bindings.setTargetColumn(String(patch.target_column));
    applied.push("target_column");
    handled.add("target_column");
  }
  if (patch.task !== undefined) {
    bindings.setTask(patch.task);
    applied.push("task");
    handled.add("task");
  }
  const runSweepValue = patch.run_sweep ?? patch.set_sweep_values;
  if (runSweepValue !== undefined) {
    const desired = Boolean(runSweepValue);
    if (desired !== bindings.runSweepEnabled) {
      bindings.toggleRunSweep(desired);
    }
    applied.push("run_sweep");
    handled.add("run_sweep");
  }
  if (patch.epoch_values !== undefined) {
    bindings.setEpochValuesInput(toBridgeCsv(patch.epoch_values));
    applied.push("epoch_values");
    handled.add("epoch_values");
  }
  if (patch.batch_sizes !== undefined) {
    bindings.setBatchSizesInput(toBridgeCsv(patch.batch_sizes));
    applied.push("batch_sizes");
    handled.add("batch_sizes");
  }
  if (patch.learning_rates !== undefined) {
    bindings.setLearningRatesInput(toBridgeCsv(patch.learning_rates));
    applied.push("learning_rates");
    handled.add("learning_rates");
  }
  if (patch.test_sizes !== undefined) {
    bindings.setTestSizesInput(toBridgeCsv(patch.test_sizes));
    applied.push("test_sizes");
    handled.add("test_sizes");
  }
  if (patch.hidden_dims !== undefined) {
    bindings.setHiddenDimsInput(toBridgeCsv(patch.hidden_dims));
    applied.push("hidden_dims");
    handled.add("hidden_dims");
  }
  if (patch.num_hidden_layers !== undefined) {
    bindings.setNumHiddenLayersInput(toBridgeCsv(patch.num_hidden_layers));
    applied.push("num_hidden_layers");
    handled.add("num_hidden_layers");
  }
  if (patch.dropouts !== undefined) {
    bindings.setDropoutsInput(toBridgeCsv(patch.dropouts));
    applied.push("dropouts");
    handled.add("dropouts");
  }
  if (patch.exclude_columns !== undefined) {
    bindings.setExcludeColumnsInput(toBridgeCsv(patch.exclude_columns));
    applied.push("exclude_columns");
    handled.add("exclude_columns");
  }
  if (patch.date_columns !== undefined) {
    bindings.setDateColumnsInput(toBridgeCsv(patch.date_columns));
    applied.push("date_columns");
    handled.add("date_columns");
  }
  if (patch.auto_distill !== undefined) {
    const desired = Boolean(patch.auto_distill);
    if (desired !== bindings.autoDistillEnabled) {
      bindings.setAutoDistillEnabled(desired);
    }
    applied.push("auto_distill");
    handled.add("auto_distill");
  }

  const skipped = Object.keys(patch).filter((key) => !handled.has(key));
  return { applied, skipped };
}

/**
 * Applies a PyTorch form patch to injected setter bindings.
 *
 * @param patch - PyTorch form patch payload.
 * @param bindings - PyTorch setter bindings and toggle state.
 * @returns Applied and skipped patch keys.
 * @complexity O(k) time and O(k) space where k is the number of keys in patch.
 * @overallScore 100
 */
export function applyPytorchBridgePatch(
  patch: PytorchBridgePatch,
  bindings: PytorchBridgePatchBindings
): PytorchBridgeApplyResult {
  return applyMlFormBridgePatch(patch, bindings);
}

/**
 * Applies a TensorFlow form patch to injected setter bindings.
 *
 * @param patch - TensorFlow form patch payload.
 * @param bindings - TensorFlow setter bindings and toggle state.
 * @returns Applied and skipped patch keys.
 * @complexity O(k) time and O(k) space where k is the number of keys in patch.
 * @overallScore 100
 */
export function applyTensorflowBridgePatch(
  patch: TensorflowBridgePatch,
  bindings: TensorflowBridgePatchBindings
): TensorflowBridgeApplyResult {
  return applyMlFormBridgePatch(patch, bindings);
}
