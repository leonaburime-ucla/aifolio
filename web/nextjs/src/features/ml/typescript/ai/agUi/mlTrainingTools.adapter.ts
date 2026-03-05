import type {
  PytorchFormBridge,
  PytorchFormPatch,
  PytorchRandomizeArgs,
  MlTask,
  PytorchTrainingMode,
  TensorflowFormBridge,
  TensorflowFormPatch,
  TensorflowRandomizeArgs,
  TensorflowTrainingMode,
} from "@/features/ml/__types__/typescript/ai/agUi/mlTrainingTooling.types";
import { trainPytorchModel, type PytorchTrainRequest } from "@/features/ml/typescript/api/pytorchApi";
import { trainTensorflowModel, type TensorflowTrainRequest } from "@/features/ml/typescript/api/tensorflowApi";

/**
 * ML AG-UI training tool adapter layer.
 *
 * Purpose:
 * - Encapsulate browser/runtime side effects for framework training tool calls.
 * - Bridge pure training logic with form state and backend APIs.
 *
 * Layering:
 * - This file is intentionally adapter-bound (window/document/network).
 * - Pure transformations/validation stay in `mlTrainingTools.logic.ts`.
 */

const PYTORCH_FIELD_SELECTORS = {
  training_mode: '[data-ai-field="pytorch_training_mode"]',
  target_column: '[data-ai-field="pytorch_target_column"]',
  task: '[data-ai-field="pytorch_task"]',
  epoch_values: '[data-ai-field="pytorch_epoch_values"]',
  batch_sizes: '[data-ai-field="pytorch_batch_sizes"]',
  learning_rates: '[data-ai-field="pytorch_learning_rates"]',
  test_sizes: '[data-ai-field="pytorch_test_sizes"]',
  hidden_dims: '[data-ai-field="pytorch_hidden_dims"]',
  num_hidden_layers: '[data-ai-field="pytorch_num_hidden_layers"]',
  dropouts: '[data-ai-field="pytorch_dropouts"]',
  exclude_columns: '[data-ai-field="pytorch_exclude_columns"]',
  date_columns: '[data-ai-field="pytorch_date_columns"]',
  run_sweep: '[data-ai-field="pytorch_run_sweep"]',
  auto_distill: '[data-ai-field="pytorch_auto_distill"]',
} as const;

const PYTORCH_MODES: PytorchTrainingMode[] = [
  "mlp_dense",
  "linear_glm_baseline",
  "tabresnet",
  "imbalance_aware",
  "calibrated_classifier",
  "tree_teacher_distillation",
];

const ML_TASKS: MlTask[] = ["auto", "classification", "regression"];

const TENSORFLOW_MODES: TensorflowTrainingMode[] = [
  "mlp_dense",
  "linear_glm_baseline",
  "wide_and_deep",
  "imbalance_aware",
  "quantile_regression",
  "calibrated_classifier",
  "entity_embeddings",
  "autoencoder_head",
  "multi_task_learning",
  "time_aware_tabular",
];

/**
 * Resolves the optional global PyTorch form bridge from `window`.
 *
 * @returns Bridge when available in browser runtime, otherwise `null`.
 */
function getPytorchBridge(): PytorchFormBridge | null {
  if (typeof window === "undefined") return null;
  const bridge = (window as Window & { __AIFOLIO_PYTORCH_FORM_BRIDGE__?: PytorchFormBridge })
    .__AIFOLIO_PYTORCH_FORM_BRIDGE__;
  return bridge ?? null;
}

/**
 * Resolves the optional global TensorFlow form bridge from `window`.
 *
 * @returns Bridge when available in browser runtime, otherwise `null`.
 */
function getTensorflowBridge(): TensorflowFormBridge | null {
  if (typeof window === "undefined") return null;
  const bridge = (window as Window & { __AIFOLIO_TENSORFLOW_FORM_BRIDGE__?: TensorflowFormBridge })
    .__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;
  return bridge ?? null;
}

/**
 * Normalizes scalar/list values into a comma-delimited string format suitable
 * for text/select control assignment.
 *
 * @param value Raw patch value.
 * @returns Normalized display/input string.
 */
function normalizeListValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean).join(",");
  }
  return String(value ?? "").trim();
}

/**
 * Picks a random item from a non-empty array.
 *
 * @param values Candidate values.
 * @returns Selected item.
 */
function randomItem<T>(values: T[]): T {
  return values[Math.floor(Math.random() * values.length)];
}

/**
 * Finds a PyTorch form control by selector in browser context.
 *
 * @param selector CSS selector for the target control.
 * @returns Matching DOM element or `null`.
 */
function findPytorchControl(selector: string): Element | null {
  if (typeof document === "undefined") return null;
  return document.querySelector(selector);
}

/**
 * Applies a value to a control and dispatches synthetic events so React state
 * observes the mutation.
 *
 * @param selector Control selector.
 * @param value Value to apply.
 * @returns `true` when the value was applied; otherwise `false`.
 */
function setControlValue(selector: string, value: unknown): boolean {
  if (typeof document === "undefined") return false;
  const element = findPytorchControl(selector);
  if (!element) return false;

  if (element instanceof HTMLInputElement) {
    if (element.type === "checkbox") {
      element.checked = Boolean(value);
      element.dispatchEvent(new Event("change", { bubbles: true }));
      return true;
    }
    element.value = normalizeListValue(value);
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  if (element instanceof HTMLSelectElement) {
    const nextValue = normalizeListValue(value);
    const hasOption = Array.from(element.options).some((option) => option.value === nextValue);
    if (!hasOption) return false;
    element.value = nextValue;
    element.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  return false;
}

/**
 * Selects a random non-empty target-column option from the PyTorch form.
 *
 * @returns Random target column id or `null` when unavailable.
 */
function pickRandomTargetColumn(): string | null {
  const select = findPytorchControl(PYTORCH_FIELD_SELECTORS.target_column);
  if (!(select instanceof HTMLSelectElement)) return null;
  const options = Array.from(select.options)
    .map((option) => option.value)
    .filter((value) => value && value.trim().length > 0);
  if (!options.length) return null;
  return randomItem(options);
}

/**
 * Reads a trimmed selected value from a `<select>` control.
 *
 * @param selector Control selector.
 * @returns Selected value or `null` when unavailable/empty.
 */
function getSelectValue(selector: string): string | null {
  const element = findPytorchControl(selector);
  if (!(element instanceof HTMLSelectElement)) return null;
  const value = element.value?.trim();
  return value ? value : null;
}

/**
 * Executes a direct backend PyTorch training request.
 *
 * @param payload Backend request body for a single PyTorch training run.
 * @returns API result from `trainPytorchModel`.
 */
export async function handleTrainPytorchModel(payload: PytorchTrainRequest) {
  return trainPytorchModel(payload);
}

/**
 * Executes a direct backend TensorFlow training request.
 *
 * @param payload Backend request body for a single TensorFlow training run.
 * @returns API result from `trainTensorflowModel`.
 */
export async function handleTrainTensorflowModel(payload: TensorflowTrainRequest) {
  return trainTensorflowModel(payload);
}

/**
 * Applies a partial PyTorch form patch through bridge first, then DOM fallback.
 *
 * @param patch Partial field patch for PyTorch form controls.
 * @returns Applied/skipped field report or an availability/validation error.
 */
export function handleSetPytorchFormFields(patch: PytorchFormPatch) {
  if (typeof document === "undefined") {
    return { status: "error" as const, code: "PYTORCH_FORM_UNAVAILABLE" as const };
  }

  const bridge = getPytorchBridge();
  if (bridge) {
    const result = bridge.applyPatch(patch);
    if (!result.applied.length) {
      return {
        status: "error" as const,
        code: "NO_FIELDS_APPLIED" as const,
        skipped: result.skipped,
      };
    }
    return {
      status: "ok" as const,
      applied: result.applied,
      skipped: result.skipped,
      via: "bridge" as const,
    };
  }

  const applied: string[] = [];
  const skipped: string[] = [];
  const orderedKeys: Array<keyof PytorchFormPatch> = [
    "run_sweep",
    "training_mode",
    "target_column",
    "task",
    "epoch_values",
    "batch_sizes",
    "learning_rates",
    "test_sizes",
    "hidden_dims",
    "num_hidden_layers",
    "dropouts",
    "exclude_columns",
    "date_columns",
    "auto_distill",
  ];

  orderedKeys.forEach((key) => {
    const value = patch[key];
    if (value === undefined) return;
    const selector = PYTORCH_FIELD_SELECTORS[key];
    if (!selector) return;
    const ok = setControlValue(selector, value);
    if (ok) applied.push(String(key));
    else skipped.push(String(key));
  });

  if (!applied.length) {
    return {
      status: "error" as const,
      code: "NO_FIELDS_APPLIED" as const,
      skipped,
    };
  }

  return {
    status: "ok" as const,
    applied,
    skipped,
  };
}

/**
 * Applies a partial TensorFlow form patch through bridge.
 *
 * @param patch Partial field patch for TensorFlow form controls.
 * @returns Applied/skipped field report or an availability/validation error.
 */
export function handleSetTensorflowFormFields(patch: TensorflowFormPatch) {
  if (typeof document === "undefined") {
    return { status: "error" as const, code: "TENSORFLOW_FORM_UNAVAILABLE" as const };
  }

  const bridge = getTensorflowBridge();
  if (!bridge) {
    return { status: "error" as const, code: "TENSORFLOW_FORM_UNAVAILABLE" as const };
  }

  const result = bridge.applyPatch(patch);
  if (!result.applied.length) {
    return {
      status: "error" as const,
      code: "NO_FIELDS_APPLIED" as const,
      skipped: result.skipped,
    };
  }
  return {
    status: "ok" as const,
    applied: result.applied,
    skipped: result.skipped,
    via: "bridge" as const,
  };
}

/**
 * Builds a validator-safe randomized PyTorch form patch from current DOM state.
 *
 * @param args Optional randomization options controlling aggressiveness and field locking.
 * @returns Patch suitable for `handleSetPytorchFormFields`.
 */
export function buildRandomPytorchFormPatch(args: PytorchRandomizeArgs = {}): PytorchFormPatch {
  const style = args.style ?? "balanced";
  const isAggressive = style === "aggressive";
  const isSafe = style === "safe";

  const epochs = isAggressive ? [20, 40, 60] : isSafe ? [40, 80] : [30, 60];
  const batchSizes = isAggressive ? [32, 64, 128] : isSafe ? [32, 64] : [32, 64];
  const learningRates = isAggressive ? [0.0005, 0.001, 0.002] : isSafe ? [0.0008, 0.0012] : [0.0008, 0.0015];
  const testSizes = isAggressive ? [0.15, 0.2, 0.25] : isSafe ? [0.2, 0.25] : [0.2, 0.25];
  const hiddenDims = isAggressive ? [64, 128, 256] : isSafe ? [96, 128] : [96, 192];
  const hiddenLayers = isAggressive ? [2, 3, 4] : isSafe ? [2, 3] : [2, 3];
  const dropouts = isAggressive ? [0.1, 0.2, 0.3] : isSafe ? [0.1, 0.2] : [0.1, 0.2];

  const randomizeModelFields = args.randomize_model_fields ?? false;
  const currentMode = getSelectValue(PYTORCH_FIELD_SELECTORS.training_mode) as PytorchTrainingMode | null;
  const trainingMode = randomizeModelFields ? randomItem(PYTORCH_MODES) : (currentMode ?? "mlp_dense");
  const isLinearMode = trainingMode === "linear_glm_baseline";
  const currentTask = getSelectValue(PYTORCH_FIELD_SELECTORS.task) as PytorchTask | null;
  const targetColumn = args.lock_target_column ? undefined : randomizeModelFields ? (pickRandomTargetColumn() ?? undefined) : undefined;
  const task = randomizeModelFields ? randomItem(ML_TASKS) : (currentTask ?? "auto");

  return {
    training_mode: randomizeModelFields ? trainingMode : undefined,
    target_column: targetColumn,
    task,
    epoch_values: epochs,
    batch_sizes: batchSizes,
    learning_rates: learningRates,
    test_sizes: testSizes,
    hidden_dims: isLinearMode ? "128" : hiddenDims,
    num_hidden_layers: isLinearMode ? "2" : hiddenLayers,
    dropouts: isLinearMode ? "0.1" : dropouts,
    run_sweep: args.run_sweep ?? args.set_sweep_values,
    auto_distill: args.auto_distill,
  };
}

/**
 * Randomizes PyTorch form fields and immediately applies the patch.
 *
 * @param args Optional randomization behavior controls.
 * @returns Randomization/apply status plus applied patch details.
 */
export function handleRandomizePytorchFormFields(args: PytorchRandomizeArgs = {}) {
  const patch = buildRandomPytorchFormPatch(args);
  const applied = handleSetPytorchFormFields(patch);
  if (applied.status !== "ok") {
    return applied;
  }
  return {
    status: "ok" as const,
    randomized: true,
    patch,
    applied: applied.applied,
    skipped: applied.skipped,
  };
}

/**
 * Builds a validator-safe randomized TensorFlow form patch from current DOM state.
 *
 * @param args Optional randomization options controlling aggressiveness and field locking.
 * @returns Patch suitable for `handleSetTensorflowFormFields`.
 */
export function buildRandomTensorflowFormPatch(args: TensorflowRandomizeArgs = {}): TensorflowFormPatch {
  const style = args.style ?? "balanced";
  const isAggressive = style === "aggressive";
  const isSafe = style === "safe";

  const epochs = isAggressive ? [20, 40, 60] : isSafe ? [40, 80] : [30, 60];
  const batchSizes = isAggressive ? [32, 64, 128] : isSafe ? [32, 64] : [32, 64];
  const learningRates = isAggressive ? [0.0005, 0.001, 0.002] : isSafe ? [0.0008, 0.0012] : [0.0008, 0.0015];
  const testSizes = isAggressive ? [0.15, 0.2, 0.25] : isSafe ? [0.2, 0.25] : [0.2, 0.25];
  const hiddenDims = isAggressive ? [64, 128, 256] : isSafe ? [96, 128] : [96, 192];
  const hiddenLayers = isAggressive ? [2, 3, 4] : isSafe ? [2, 3] : [2, 3];
  const dropouts = isAggressive ? [0.1, 0.2, 0.3] : isSafe ? [0.1, 0.2] : [0.1, 0.2];

  const randomizeModelFields = args.randomize_model_fields ?? false;
  const trainingMode = randomizeModelFields ? randomItem(TENSORFLOW_MODES) : undefined;
  const task = randomizeModelFields ? randomItem(ML_TASKS) : "auto";

  return {
    training_mode: trainingMode,
    target_column: args.lock_target_column ? undefined : undefined,
    task,
    epoch_values: epochs,
    batch_sizes: batchSizes,
    learning_rates: learningRates,
    test_sizes: testSizes,
    hidden_dims: hiddenDims,
    num_hidden_layers: hiddenLayers,
    dropouts,
    run_sweep: args.run_sweep ?? args.set_sweep_values,
    auto_distill: args.auto_distill,
  };
}

/**
 * Randomizes TensorFlow form fields and immediately applies the patch.
 *
 * @param args Optional randomization behavior controls.
 * @returns Randomization/apply status plus applied patch details.
 */
export function handleRandomizeTensorflowFormFields(args: TensorflowRandomizeArgs = {}) {
  const patch = buildRandomTensorflowFormPatch(args);
  const applied = handleSetTensorflowFormFields(patch);
  if (applied.status !== "ok") {
    return applied;
  }
  return {
    status: "ok" as const,
    randomized: true,
    patch,
    applied: applied.applied,
    skipped: applied.skipped,
  };
}

/**
 * Starts PyTorch training runs via the injected window bridge on the PyTorch page.
 *
 * @returns Success when training was started; otherwise a typed error code.
 */
export async function handleStartPytorchTrainingRuns() {
  const bridge = getPytorchBridge();
  if (!bridge?.startTrainingRuns) {
    return { status: "error" as const, code: "PYTORCH_FORM_UNAVAILABLE" as const };
  }
  const result = await bridge.startTrainingRuns();
  if (result.status === "ok") {
    return { status: "ok" as const, started: true };
  }
  return { status: "error" as const, code: result.reason };
}

/**
 * Starts TensorFlow training runs via the injected window bridge on the TensorFlow page.
 *
 * @returns Success when training was started; otherwise a typed error code.
 */
export async function handleStartTensorflowTrainingRuns() {
  const bridge = getTensorflowBridge();
  if (!bridge?.startTrainingRuns) {
    return { status: "error" as const, code: "TENSORFLOW_FORM_UNAVAILABLE" as const };
  }
  const result = await bridge.startTrainingRuns();
  if (result.status === "ok") {
    return { status: "ok" as const, started: true };
  }
  return { status: "error" as const, code: result.reason };
}
