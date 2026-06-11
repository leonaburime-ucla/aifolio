/**
 * Result shape for list-style hyperparameter validation.
 * Returns either normalized values or a user-safe error string.
 */
import type {
  SweepCombination,
  SweepConfig,
  ValidationResult,
} from "@aifolio/contracts/entities/ml-training";
export type {
  SweepCombination,
  SweepConfig,
  ValidationResult,
} from "@aifolio/contracts/entities/ml-training";

const LIST_SPLIT_REGEX = /[,\s]+/;
type EmptyOptions = Record<string, never>;

function parseTokens(
  { raw }: { raw: string },
  {}: EmptyOptions = {}
): string[] {
  return raw
    .split(LIST_SPLIT_REGEX)
    .map((token) => token.trim())
    .filter(Boolean);
}

function parseNumberList(
  { raw }: { raw: string },
  {}: EmptyOptions = {}
): ValidationResult<number> {
  const tokens = parseTokens({ raw });
  if (tokens.length === 0) {
    return { ok: false, error: "Provide at least one value." };
  }

  const numbers: number[] = [];
  for (const token of tokens) {
    const value = Number(token);
    if (!Number.isFinite(value)) {
      return { ok: false, error: `Invalid number: ${token}` };
    }
    numbers.push(value);
  }
  return { ok: true, values: numbers };
}

function uniqueSorted(
  { values }: { values: number[] },
  {}: EmptyOptions = {}
): number[] {
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

/**
 * Validate integer epoch values in the inclusive range `[1, 500]`.
 *
 * @param params - Raw epoch list input.
 * @returns Validated sorted unique epochs or a user-safe error.
 * @complexity O(n log n) time and O(n) space for n tokens.
 * @overallScore 100
 */
export function validateEpochValues(
  { raw }: { raw: string },
  {}: EmptyOptions = {}
): ValidationResult<number> {
  const parsed = parseNumberList({ raw });
  if (!parsed.ok) return parsed;

  const out: number[] = [];
  for (const value of parsed.values) {
    if (!Number.isInteger(value)) {
      return { ok: false, error: `Epoch must be an integer: ${value}` };
    }
    if (value < 1 || value > 500) {
      return { ok: false, error: `Epoch out of range (1-500): ${value}` };
    }
    out.push(value);
  }
  return { ok: true, values: uniqueSorted({ values: out }) };
}

/**
 * Validate test split values where each value is strictly between `0` and `1`.
 *
 * @param params - Raw test-size list input.
 * @returns Validated sorted unique test sizes or a user-safe error.
 * @complexity O(n log n) time and O(n) space for n tokens.
 * @overallScore 100
 */
export function validateTestSizes(
  { raw }: { raw: string },
  {}: EmptyOptions = {}
): ValidationResult<number> {
  const parsed = parseNumberList({ raw });
  if (!parsed.ok) return parsed;

  const out: number[] = [];
  for (const value of parsed.values) {
    if (value <= 0 || value >= 1) {
      return { ok: false, error: `Test size must be > 0 and < 1: ${value}` };
    }
    out.push(value);
  }
  return { ok: true, values: uniqueSorted({ values: out }) };
}

/**
 * Validate learning-rate values in the range `(0, 1]`.
 *
 * @param params - Raw learning-rate list input.
 * @returns Validated sorted unique learning rates or a user-safe error.
 * @complexity O(n log n) time and O(n) space for n tokens.
 * @overallScore 100
 */
export function validateLearningRates(
  { raw }: { raw: string },
  {}: EmptyOptions = {}
): ValidationResult<number> {
  const parsed = parseNumberList({ raw });
  if (!parsed.ok) return parsed;

  const out: number[] = [];
  for (const value of parsed.values) {
    if (value <= 0 || value > 1) {
      return { ok: false, error: `Learning rate must be > 0 and <= 1: ${value}` };
    }
    out.push(value);
  }
  return { ok: true, values: uniqueSorted({ values: out }) };
}

/**
 * Validate integer batch sizes in the inclusive range `[1, 200]`.
 *
 * @param params - Raw batch-size list input.
 * @returns Validated sorted unique batch sizes or a user-safe error.
 * @complexity O(n log n) time and O(n) space for n tokens.
 * @overallScore 100
 */
export function validateBatchSizes(
  { raw }: { raw: string },
  {}: EmptyOptions = {}
): ValidationResult<number> {
  const parsed = parseNumberList({ raw });
  if (!parsed.ok) return parsed;

  const out: number[] = [];
  for (const value of parsed.values) {
    if (!Number.isInteger(value)) {
      return { ok: false, error: `Batch size must be an integer: ${value}` };
    }
    if (value < 1 || value > 200) {
      return { ok: false, error: `Batch size out of range (1-200): ${value}` };
    }
    out.push(value);
  }
  return { ok: true, values: uniqueSorted({ values: out }) };
}

/**
 * Validate integer hidden dimensions in the inclusive range `[8, 500]`.
 *
 * @param params - Raw hidden-dimension list input.
 * @returns Validated sorted unique hidden dimensions or a user-safe error.
 * @complexity O(n log n) time and O(n) space for n tokens.
 * @overallScore 100
 */
export function validateHiddenDims(
  { raw }: { raw: string },
  {}: EmptyOptions = {}
): ValidationResult<number> {
  const parsed = parseNumberList({ raw });
  if (!parsed.ok) return parsed;

  const out: number[] = [];
  for (const value of parsed.values) {
    if (!Number.isInteger(value)) {
      return { ok: false, error: `Hidden dim must be an integer: ${value}` };
    }
    if (value < 8 || value > 500) {
      return { ok: false, error: `Hidden dim out of range (8-500): ${value}` };
    }
    out.push(value);
  }
  return { ok: true, values: uniqueSorted({ values: out }) };
}

/**
 * Validate integer hidden-layer counts in the inclusive range `[1, 15]`.
 *
 * @param params - Raw hidden-layer count list input.
 * @returns Validated sorted unique hidden-layer counts or a user-safe error.
 * @complexity O(n log n) time and O(n) space for n tokens.
 * @overallScore 100
 */
export function validateNumHiddenLayers(
  { raw }: { raw: string },
  {}: EmptyOptions = {}
): ValidationResult<number> {
  const parsed = parseNumberList({ raw });
  if (!parsed.ok) return parsed;

  const out: number[] = [];
  for (const value of parsed.values) {
    if (!Number.isInteger(value)) {
      return { ok: false, error: `Hidden layers must be an integer: ${value}` };
    }
    if (value < 1 || value > 15) {
      return { ok: false, error: `Hidden layers out of range (1-15): ${value}` };
    }
    out.push(value);
  }
  return { ok: true, values: uniqueSorted({ values: out }) };
}

/**
 * Validate dropout values in the range `[0, 0.9]`.
 *
 * @param params - Raw dropout list input.
 * @returns Validated sorted unique dropout values or a user-safe error.
 * @complexity O(n log n) time and O(n) space for n tokens.
 * @overallScore 100
 */
export function validateDropouts(
  { raw }: { raw: string },
  {}: EmptyOptions = {}
): ValidationResult<number> {
  const parsed = parseNumberList({ raw });
  if (!parsed.ok) return parsed;

  const out: number[] = [];
  for (const value of parsed.values) {
    if (value < 0 || value > 0.9) {
      return { ok: false, error: `Dropout out of range (0-0.9): ${value}` };
    }
    out.push(value);
  }
  return { ok: true, values: uniqueSorted({ values: out }) };
}

/**
 * Build the Cartesian product of validated sweep dimensions.
 * Each returned entry corresponds to one planned training run.
 *
 * @param params - Validated sweep configuration.
 * @returns Planned sweep combinations.
 * @complexity O(a*b*c*d*e*f*g) time and space over sweep dimension counts.
 * @overallScore 100
 */
export function buildSweepCombinations(
  { config }: { config: SweepConfig },
  {}: EmptyOptions = {}
): SweepCombination[] {
  const out: SweepCombination[] = [];
  for (const epochs of config.epochs) {
    for (const testSize of config.testSizes) {
      for (const learningRate of config.learningRates) {
        for (const batchSize of config.batchSizes) {
          for (const hiddenDim of config.hiddenDims) {
            for (const numHiddenLayers of config.numHiddenLayers) {
              for (const dropout of config.dropouts) {
                out.push({
                  epochs,
                  testSize,
                  learningRate,
                  batchSize,
                  hiddenDim,
                  numHiddenLayers,
                  dropout,
                });
              }
            }
          }
        }
      }
    }
  }
  return out;
}
