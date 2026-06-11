export const defaultSweepCombination = {
  epochs: 60,
  testSize: 0.2,
  learningRate: 0.001,
  batchSize: 64,
  hiddenDim: 128,
  numHiddenLayers: 2,
  dropout: 0.1,
};

export const defaultTeacherConfig = {
  hidden: 128,
  layers: 2,
  dropout: 0.1,
  epochs: 60,
  batch: 64,
  learningRate: 0.001,
  testSize: 0.2,
};

export const defaultFormatters = {
  formatCompletedAt: () => "01/01/26 00:00:00",
  formatMetricNumber: ({ value }: { value: unknown }) => String(value ?? "n/a"),
};
