import type {
  PytorchBridgeApplyResult,
  PytorchBridgePatch,
  PytorchBridgePatchBindings,
} from "@/features/ml/__types__/typescript/react/ai/tools/pytorchFormBridge.tools.types";

/**
 * Normalizes scalar/array patch values into comma-separated UI input format.
 *
 * @param value Raw value from tool-call payload.
 * @returns Comma-separated string suitable for controlled text inputs.
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
 * Applies a tool-call patch to bound PyTorch form setters.
 *
 * This function is intentionally pure with respect to global/browser state:
 * - It only uses passed bindings.
 * - It returns deterministic `applied/skipped` metadata for assertions.
 *
 * @param patch Tool-call patch payload.
 * @param bindings Patch application dependencies and current boolean toggles.
 * @returns Structured patch result with applied and skipped keys.
 */
export function applyPytorchBridgePatch(
  patch: PytorchBridgePatch,
  bindings: PytorchBridgePatchBindings
): PytorchBridgeApplyResult {
  const applied: string[] = [];
  const handled = new Set<string>();

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
