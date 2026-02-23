export type NumericInputSnapshot = {
  epochValuesInput: string;
  batchSizesInput: string;
  learningRatesInput: string;
  testSizesInput: string;
  hiddenDimsInput: string;
  numHiddenLayersInput: string;
  dropoutsInput: string;
};

type NumericInputSetters = {
  setEpochValuesInput: (value: string) => void;
  setBatchSizesInput: (value: string) => void;
  setLearningRatesInput: (value: string) => void;
  setTestSizesInput: (value: string) => void;
  setHiddenDimsInput: (value: string) => void;
  setNumHiddenLayersInput: (value: string) => void;
  setDropoutsInput: (value: string) => void;
};

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomFloat(min: number, max: number, decimals = 4): number {
  const value = Math.random() * (max - min) + min;
  return Number(value.toFixed(decimals));
}

export function buildRandomSweepInputs(): NumericInputSnapshot {
  const randomEpochs = [randomInt(40, 180), randomInt(181, 500)]
    .sort((a, b) => a - b)
    .join(",");
  const randomBatchSizes = [randomInt(8, 96), randomInt(97, 200)]
    .sort((a, b) => a - b)
    .join(",");
  const randomLearningRates = [randomFloat(0.0002, 0.003, 4), randomFloat(0.0031, 0.03, 4)]
    .sort((a, b) => a - b)
    .join(",");
  const randomTestSizes = [randomFloat(0.1, 0.3, 2), randomFloat(0.31, 0.45, 2)]
    .sort((a, b) => a - b)
    .join(",");
  const randomHiddenDims = [randomInt(32, 220), randomInt(221, 500)]
    .sort((a, b) => a - b)
    .join(",");
  const randomHiddenLayers = [randomInt(1, 6), randomInt(7, 15)]
    .sort((a, b) => a - b)
    .join(",");
  const randomDropouts = [randomFloat(0.05, 0.2, 2), randomFloat(0.21, 0.45, 2)]
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

export function applyNumericInputs(
  snapshot: NumericInputSnapshot,
  setters: NumericInputSetters
) {
  setters.setEpochValuesInput(snapshot.epochValuesInput);
  setters.setBatchSizesInput(snapshot.batchSizesInput);
  setters.setLearningRatesInput(snapshot.learningRatesInput);
  setters.setTestSizesInput(snapshot.testSizesInput);
  setters.setHiddenDimsInput(snapshot.hiddenDimsInput);
  setters.setNumHiddenLayersInput(snapshot.numHiddenLayersInput);
  setters.setDropoutsInput(snapshot.dropoutsInput);
}

export function parseNumericValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const normalized = trimmed.replace("x10^", "e");
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

export function metricHigherIsBetter(metricName: string): boolean {
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
