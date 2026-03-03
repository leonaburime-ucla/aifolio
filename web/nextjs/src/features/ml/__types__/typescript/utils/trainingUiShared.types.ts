import type { TrainingMetrics } from "@/features/ml/__types__/typescript/utils/trainingRuns.types";

export type NumericInputSnapshot = {
  epochValuesInput: string;
  batchSizesInput: string;
  learningRatesInput: string;
  testSizesInput: string;
  hiddenDimsInput: string;
  numHiddenLayersInput: string;
  dropoutsInput: string;
};

export type NumericInputSetters = {
  setEpochValuesInput: (value: string) => void;
  setBatchSizesInput: (value: string) => void;
  setLearningRatesInput: (value: string) => void;
  setTestSizesInput: (value: string) => void;
  setHiddenDimsInput: (value: string) => void;
  setNumHiddenLayersInput: (value: string) => void;
  setDropoutsInput: (value: string) => void;
};

export type RandomValueProvider = () => number;

export type DistilledSnapshot = {
  metrics: TrainingMetrics;
  modelId: string | null;
  modelPath: string | null;
};
