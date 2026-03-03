import type { HyperParams } from "@/features/ml/__types__/typescript/utils/bayesianOptimizer.types";
import type { NumericInputSnapshot } from "@/features/ml/__types__/typescript/utils/trainingUiShared.types";
import type {
  DistillComparison,
  TrainingMetrics,
  TrainingProgress,
} from "@/features/ml/__types__/typescript/utils/trainingRuns.types";
import type { MlTaskType } from "@/features/ml/__types__/typescript/config/datasetTrainingDefaults.types";

export type OptimalPrediction = {
  metricName: string;
  metricValue: number;
};

export type MlTrainingUiBaseState = {
  targetColumn: string;
  setTargetColumn: (value: string) => void;
  excludeColumnsInput: string | null;
  setExcludeColumnsInput: (value: string | null) => void;
  dateColumnsInput: string | null;
  setDateColumnsInput: (value: string | null) => void;
  task: MlTaskType;
  setTask: (value: MlTaskType) => void;
  epochValuesInput: string;
  setEpochValuesInput: (value: string) => void;
  testSizesInput: string;
  setTestSizesInput: (value: string) => void;
  learningRatesInput: string;
  setLearningRatesInput: (value: string) => void;
  batchSizesInput: string;
  setBatchSizesInput: (value: string) => void;
  hiddenDimsInput: string;
  setHiddenDimsInput: (value: string) => void;
  numHiddenLayersInput: string;
  setNumHiddenLayersInput: (value: string) => void;
  dropoutsInput: string;
  setDropoutsInput: (value: string) => void;
  runSweepEnabled: boolean;
  setRunSweepEnabled: (value: boolean) => void;
  savedNumericInputs: NumericInputSnapshot | null;
  setSavedNumericInputs: (value: NumericInputSnapshot | null) => void;
  savedSweepInputs: NumericInputSnapshot | null;
  setSavedSweepInputs: (value: NumericInputSnapshot | null) => void;
  isTraining: boolean;
  setIsTraining: (value: boolean) => void;
  isDistilling: boolean;
  setIsDistilling: (value: boolean) => void;
  autoDistillEnabled: boolean;
  setAutoDistillEnabled: (value: boolean) => void;
  trainingProgress: TrainingProgress;
  setTrainingProgress: (value: TrainingProgress) => void;
  trainingError: string | null;
  setTrainingError: (value: string | null) => void;
  copyRunsStatus: string | null;
  setCopyRunsStatus: (value: string | null) => void;
  optimizerStatus: string | null;
  setOptimizerStatus: (value: string | null) => void;
  distillStatus: string | null;
  setDistillStatus: (value: string | null) => void;
  saveDistilledModel: boolean;
  setSaveDistilledModel: (value: boolean) => void;
  isOptimalModalOpen: boolean;
  setIsOptimalModalOpen: (value: boolean) => void;
  pendingOptimalParams: HyperParams | null;
  setPendingOptimalParams: (value: HyperParams | null) => void;
  pendingOptimalPrediction: OptimalPrediction | null;
  setPendingOptimalPrediction: (value: OptimalPrediction | null) => void;
  isDistillMetricsModalOpen: boolean;
  setIsDistillMetricsModalOpen: (value: boolean) => void;
  distillMetrics: TrainingMetrics | null;
  setDistillMetrics: (value: TrainingMetrics | null) => void;
  distillModelId: string | null;
  setDistillModelId: (value: string | null) => void;
  distillModelPath: string | null;
  setDistillModelPath: (value: string | null) => void;
  distillComparison: DistillComparison | null;
  setDistillComparison: (value: DistillComparison | null) => void;
};
