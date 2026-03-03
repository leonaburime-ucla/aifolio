export type ValidationResult<T> =
  | { ok: true; values: T[] }
  | { ok: false; error: string };

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
