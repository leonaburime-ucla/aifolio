/**
 * Result shape for list-style hyperparameter validation.
 * Returns either normalized values or a user-safe error string.
 */
export type ValidationResult<T> =
  | { ok: true; values: T[] }
  | { ok: false; error: string };

const LIST_SPLIT_REGEX = /[,\s]+/;

function parseTokens(raw: string): string[] {
  return raw
    .split(LIST_SPLIT_REGEX)
    .map((token) => token.trim())
    .filter(Boolean);
}

function parseNumberList(raw: string): ValidationResult<number> {
  const tokens = parseTokens(raw);
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

function uniqueSorted(values: number[]): number[] {
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

/**
 * Validate integer epoch values in the inclusive range `[1, 500]`.
 */
export function validateEpochValues(raw: string): ValidationResult<number> {
  const parsed = parseNumberList(raw);
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
  return { ok: true, values: uniqueSorted(out) };
}

/**
 * Validate test split values where each value is strictly between `0` and `1`.
 */
export function validateTestSizes(raw: string): ValidationResult<number> {
  const parsed = parseNumberList(raw);
  if (!parsed.ok) return parsed;

  const out: number[] = [];
  for (const value of parsed.values) {
    if (value <= 0 || value >= 1) {
      return { ok: false, error: `Test size must be > 0 and < 1: ${value}` };
    }
    out.push(value);
  }
  return { ok: true, values: uniqueSorted(out) };
}

/**
 * Validate learning-rate values in the range `(0, 1]`.
 */
export function validateLearningRates(raw: string): ValidationResult<number> {
  const parsed = parseNumberList(raw);
  if (!parsed.ok) return parsed;

  const out: number[] = [];
  for (const value of parsed.values) {
    if (value <= 0 || value > 1) {
      return { ok: false, error: `Learning rate must be > 0 and <= 1: ${value}` };
    }
    out.push(value);
  }
  return { ok: true, values: uniqueSorted(out) };
}

/**
 * Validate integer batch sizes in the inclusive range `[1, 200]`.
 */
export function validateBatchSizes(raw: string): ValidationResult<number> {
  const parsed = parseNumberList(raw);
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
  return { ok: true, values: uniqueSorted(out) };
}

export type SweepConfig = {
  epochs: number[];
  testSizes: number[];
  learningRates: number[];
  batchSizes: number[];
  hiddenDims: number[];
  numHiddenLayers: number[];
  dropouts: number[];
};

export type SweepCombination = {
  epochs: number;
  testSize: number;
  learningRate: number;
  batchSize: number;
  hiddenDim: number;
  numHiddenLayers: number;
  dropout: number;
};

/**
 * Validate integer hidden dimensions in the inclusive range `[8, 500]`.
 */
export function validateHiddenDims(raw: string): ValidationResult<number> {
  const parsed = parseNumberList(raw);
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
  return { ok: true, values: uniqueSorted(out) };
}

/**
 * Validate integer hidden-layer counts in the inclusive range `[1, 15]`.
 */
export function validateNumHiddenLayers(raw: string): ValidationResult<number> {
  const parsed = parseNumberList(raw);
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
  return { ok: true, values: uniqueSorted(out) };
}

/**
 * Validate dropout values in the range `[0, 0.9]`.
 */
export function validateDropouts(raw: string): ValidationResult<number> {
  const parsed = parseNumberList(raw);
  if (!parsed.ok) return parsed;

  const out: number[] = [];
  for (const value of parsed.values) {
    if (value < 0 || value > 0.9) {
      return { ok: false, error: `Dropout out of range (0-0.9): ${value}` };
    }
    out.push(value);
  }
  return { ok: true, values: uniqueSorted(out) };
}

/**
 * Build the Cartesian product of validated sweep dimensions.
 * Each returned entry corresponds to one planned training run.
 */
export function buildSweepCombinations(config: SweepConfig): SweepCombination[] {
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
