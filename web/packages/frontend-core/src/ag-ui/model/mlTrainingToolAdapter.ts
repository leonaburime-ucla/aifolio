import type {
  MlTask,
  PytorchFormPatch,
  PytorchRandomizeArgs,
  PytorchTrainingMode,
  TensorflowFormPatch,
  TensorflowRandomizeArgs,
  TensorflowTrainingMode,
} from "@aifolio/contracts/entities/ml-training";

export const PYTORCH_FIELD_SELECTORS = {
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

export const TENSORFLOW_FIELD_SELECTORS = {
  training_mode: '[data-ai-field="tensorflow_training_mode"]',
  target_column: '[data-ai-field="tensorflow_target_column"]',
  task: '[data-ai-field="tensorflow_task"]',
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

export type RandomizeMlFormRuntime = {
  random?: () => number;
  getSelectValue?: (field: "pytorch_training_mode" | "pytorch_task" | "tensorflow_task") => string | null;
};

export type MlTargetColumnOption = {
  value: string;
  isCurrent?: boolean;
};

function randomItem<T>(
  values: T[],
  runtime: Pick<RandomizeMlFormRuntime, "random"> = {}
): T {
  const random = runtime.random ?? Math.random;
  return values[Math.floor(random() * values.length)];
}

/**
 * Selects one or more numeric sweep values from a candidate list.
 *
 * @param values - Candidate values for one sweep field.
 * @param valueCount - Optional requested number of distinct values.
 * @param runtime - Optional deterministic random source for tests.
 * @returns Selected values.
 * @complexity O(n) time and O(n) space when multiple values are requested.
 * @overallScore 100
 */
export function selectSweepValues(
  values: number[],
  valueCount?: number,
  runtime: Pick<RandomizeMlFormRuntime, "random"> = {}
): number[] {
  if (!values.length) return [];
  if (typeof valueCount !== "number" || !Number.isFinite(valueCount)) {
    return [randomItem(values, runtime)];
  }

  const nextCount = Math.max(1, Math.min(values.length, Math.floor(valueCount)));
  if (nextCount === values.length) return [...values];
  if (nextCount === 1) return [randomItem(values, runtime)];

  const pool = [...values];
  const selected: number[] = [];
  while (selected.length < nextCount && pool.length > 0) {
    const random = runtime.random ?? Math.random;
    const index = Math.floor(random() * pool.length);
    const [picked] = pool.splice(index, 1);
    selected.push(picked);
  }
  return selected;
}

/**
 * Resolves a target-column change request against available select options.
 *
 * @param options - Select options with the current value marked when known.
 * @param args - Optional explicit target or selection mode.
 * @param runtime - Optional deterministic random source for tests.
 * @returns Concrete target column or null when no valid target exists.
 * @complexity O(n) time and O(n) space for n target options.
 * @overallScore 100
 */
export function resolveTargetColumnChangeFromOptions(
  options: MlTargetColumnOption[],
  args: { target_column?: string; mode?: "different" | "random" | "next" },
  runtime: Pick<RandomizeMlFormRuntime, "random"> = {}
): string | null {
  const values = options
    .map((option) => option.value)
    .filter((value) => value && value.trim().length > 0);
  if (!values.length) return null;

  if (typeof args.target_column === "string" && args.target_column.trim().length > 0) {
    const explicit = args.target_column.trim();
    return values.includes(explicit) ? explicit : null;
  }

  const currentIndex = options.findIndex((option) => option.isCurrent);
  if (args.mode === "next") {
    if (currentIndex === -1) return values[0] ?? null;
    return values[(currentIndex + 1) % values.length] ?? null;
  }

  const currentValue =
    currentIndex >= 0 ? options[currentIndex]?.value?.trim() : "";
  const alternatives = currentValue
    ? values.filter((value) => value !== currentValue)
    : values;
  const candidatePool = alternatives.length > 0 ? alternatives : values;
  return randomItem(candidatePool, runtime);
}

/**
 * Builds a validator-safe randomized PyTorch form patch.
 *
 * @param args - Randomization options controlling style and locked fields.
 * @param runtime - Optional current-select reader and deterministic random source.
 * @returns PyTorch form patch suitable for a bridge or DOM adapter.
 * @complexity O(v) time and space for selected sweep values.
 * @overallScore 100
 */
export function buildRandomPytorchFormPatch(
  args: PytorchRandomizeArgs = {},
  runtime: RandomizeMlFormRuntime = {}
): PytorchFormPatch {
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
  const currentMode = runtime.getSelectValue?.("pytorch_training_mode") as PytorchTrainingMode | null;
  const trainingMode = randomizeModelFields ? randomItem(PYTORCH_MODES, runtime) : (currentMode ?? "mlp_dense");
  const isLinearMode = trainingMode === "linear_glm_baseline";
  const currentTask = runtime.getSelectValue?.("pytorch_task") as MlTask | null;
  const task = randomizeModelFields ? randomItem(ML_TASKS, runtime) : (currentTask ?? "auto");
  const valueCount = args.value_count;

  return {
    training_mode: randomizeModelFields ? trainingMode : undefined,
    task,
    epoch_values: selectSweepValues(epochs, valueCount, runtime),
    batch_sizes: selectSweepValues(batchSizes, valueCount, runtime),
    learning_rates: selectSweepValues(learningRates, valueCount, runtime),
    test_sizes: selectSweepValues(testSizes, valueCount, runtime),
    hidden_dims: isLinearMode ? "128" : selectSweepValues(hiddenDims, valueCount, runtime),
    num_hidden_layers: isLinearMode ? "2" : selectSweepValues(hiddenLayers, valueCount, runtime),
    dropouts: isLinearMode ? "0.1" : selectSweepValues(dropouts, valueCount, runtime),
    run_sweep: args.run_sweep ?? args.set_sweep_values,
    auto_distill: args.auto_distill,
  };
}

/**
 * Builds a validator-safe randomized TensorFlow form patch.
 *
 * @param args - Randomization options controlling style and locked fields.
 * @param runtime - Optional current-select reader and deterministic random source.
 * @returns TensorFlow form patch suitable for a bridge or DOM adapter.
 * @complexity O(v) time and space for selected sweep values.
 * @overallScore 100
 */
export function buildRandomTensorflowFormPatch(
  args: TensorflowRandomizeArgs = {},
  runtime: RandomizeMlFormRuntime = {}
): TensorflowFormPatch {
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
  const trainingMode = randomizeModelFields
    ? randomItem(TENSORFLOW_MODES, runtime)
    : undefined;
  const currentTask = runtime.getSelectValue?.("tensorflow_task") as MlTask | null;
  const task = randomizeModelFields ? randomItem(ML_TASKS, runtime) : (currentTask ?? "auto");
  const valueCount = args.value_count;

  return {
    training_mode: trainingMode,
    task,
    epoch_values: selectSweepValues(epochs, valueCount, runtime),
    batch_sizes: selectSweepValues(batchSizes, valueCount, runtime),
    learning_rates: selectSweepValues(learningRates, valueCount, runtime),
    test_sizes: selectSweepValues(testSizes, valueCount, runtime),
    hidden_dims: selectSweepValues(hiddenDims, valueCount, runtime),
    num_hidden_layers: selectSweepValues(hiddenLayers, valueCount, runtime),
    dropouts: selectSweepValues(dropouts, valueCount, runtime),
    run_sweep: args.run_sweep ?? args.set_sweep_values,
    auto_distill: args.auto_distill,
  };
}
