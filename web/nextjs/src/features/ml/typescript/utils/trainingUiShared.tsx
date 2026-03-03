import type {
  NumericInputSetters,
  NumericInputSnapshot,
  RandomValueProvider,
} from "@/features/ml/__types__/typescript/utils/trainingUiShared.types";
export type {
  NumericInputSetters,
  NumericInputSnapshot,
  RandomValueProvider,
} from "@/features/ml/__types__/typescript/utils/trainingUiShared.types";
type EmptyOptions = Record<string, never>;
type RandomOptions = {
  random?: RandomValueProvider;
};

function randomInt(
  { min, max }: { min: number; max: number },
  { random = Math.random }: RandomOptions = {}
): number {
  return Math.floor(random() * (max - min + 1)) + min;
}

function randomFloat(
  { min, max, decimals = 4 }: { min: number; max: number; decimals?: number },
  { random = Math.random }: RandomOptions = {}
): number {
  const value = random() * (max - min) + min;
  return Number(value.toFixed(decimals));
}

/**
 * Generates randomized sweep input strings within safe training ranges.
 *
 * @param _params - Required parameter object.
 * @param _options - Optional reserved options object.
 * @returns A full snapshot compatible with ML sweep form inputs.
 */
export function buildRandomSweepInputs(
  {}: Record<string, never>,
  { random = Math.random }: RandomOptions = {}
): NumericInputSnapshot {
  const randomEpochs = [
    randomInt({ min: 40, max: 180 }, { random }),
    randomInt({ min: 181, max: 500 }, { random }),
  ]
    .sort((a, b) => a - b)
    .join(",");
  const randomBatchSizes = [
    randomInt({ min: 8, max: 96 }, { random }),
    randomInt({ min: 97, max: 200 }, { random }),
  ]
    .sort((a, b) => a - b)
    .join(",");
  const randomLearningRates = [
    randomFloat({ min: 0.0002, max: 0.003, decimals: 4 }, { random }),
    randomFloat({ min: 0.0031, max: 0.03, decimals: 4 }, { random }),
  ]
    .sort((a, b) => a - b)
    .join(",");
  const randomTestSizes = [
    randomFloat({ min: 0.1, max: 0.3, decimals: 2 }, { random }),
    randomFloat({ min: 0.31, max: 0.45, decimals: 2 }, { random }),
  ]
    .sort((a, b) => a - b)
    .join(",");
  const randomHiddenDims = [
    randomInt({ min: 32, max: 220 }, { random }),
    randomInt({ min: 221, max: 500 }, { random }),
  ]
    .sort((a, b) => a - b)
    .join(",");
  const randomHiddenLayers = [
    randomInt({ min: 1, max: 6 }, { random }),
    randomInt({ min: 7, max: 15 }, { random }),
  ]
    .sort((a, b) => a - b)
    .join(",");
  const randomDropouts = [
    randomFloat({ min: 0.05, max: 0.2, decimals: 2 }, { random }),
    randomFloat({ min: 0.21, max: 0.45, decimals: 2 }, { random }),
  ]
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

/**
 * Applies a numeric-input snapshot to UI setter callbacks.
 *
 * @param params - Required parameter object.
 * @param params.snapshot - Source values to apply.
 * @param params.setters - Setter callbacks for the numeric controls.
 * @param _options - Optional reserved options object.
 * @returns `void`.
 */
export function applyNumericInputs(
  { snapshot, setters }: { snapshot: NumericInputSnapshot; setters: NumericInputSetters },
  {}: EmptyOptions = {}
): void {
  setters.setEpochValuesInput(snapshot.epochValuesInput);
  setters.setBatchSizesInput(snapshot.batchSizesInput);
  setters.setLearningRatesInput(snapshot.learningRatesInput);
  setters.setTestSizesInput(snapshot.testSizesInput);
  setters.setHiddenDimsInput(snapshot.hiddenDimsInput);
  setters.setNumHiddenLayersInput(snapshot.numHiddenLayersInput);
  setters.setDropoutsInput(snapshot.dropoutsInput);
}

/**
 * Parses numeric UI values including scientific notation aliases.
 *
 * @param params - Required parameter object.
 * @param params.value - Candidate value from form inputs or table cells.
 * @param _options - Optional reserved options object.
 * @returns Parsed finite number or `null`.
 */
export function parseNumericValue(
  { value }: { value: unknown },
  {}: EmptyOptions = {}
): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const normalized = trimmed.replace("x10^", "e");
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * Determines whether higher metric values represent better model quality.
 *
 * @param params - Required parameter object.
 * @param params.metricName - Metric name as reported by backend results.
 * @param _options - Optional reserved options object.
 * @returns `true` when higher values are better for the metric.
 */
export function metricHigherIsBetter(
  { metricName }: { metricName: string },
  {}: EmptyOptions = {}
): boolean {
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
