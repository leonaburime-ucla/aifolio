import type { HyperParams } from "@/features/ml/__types__/typescript/utils/bayesianOptimizer.types";
import type { TrainingRunRow } from "@/features/ml/__types__/typescript/utils/trainingRuns.types";
import type {
  NumericInputSetters,
  NumericInputSnapshot,
} from "@/features/ml/__types__/typescript/utils/trainingUiShared.types";
import type { OptimalPrediction } from "@/features/ml/__types__/typescript/react/hooks/mlTrainingUiBase.types";

export type NumericInputState = NumericInputSnapshot &
  NumericInputSetters & {
    savedNumericInputs: NumericInputSnapshot | null;
    setSavedNumericInputs: (value: NumericInputSnapshot | null) => void;
    savedSweepInputs: NumericInputSnapshot | null;
    setSavedSweepInputs: (value: NumericInputSnapshot | null) => void;
    setRunSweepEnabled: (value: boolean) => void;
  };

export type OptimizerUiState = {
  pendingOptimalParams: HyperParams | null;
  setPendingOptimalParams: (value: HyperParams | null) => void;
  setPendingOptimalPrediction: (value: OptimalPrediction | null) => void;
  setIsOptimalModalOpen: (value: boolean) => void;
  setOptimizerStatus: (value: string | null) => void;
};

export type HandleFindOptimalParamsArgs = {
  trainingRuns: TrainingRunRow[];
  ui: OptimizerUiState;
};

export type TrainingSharedScheduler = (
  callback: () => void,
  delayMs: number
) => void;

export type TrainingSharedClipboardWriter = (text: string) => Promise<void>;

export type TrainingSharedRuntime = {
  schedule: TrainingSharedScheduler;
  writeClipboardText: TrainingSharedClipboardWriter;
};

export type HandleApplyOptimalParamsUi = OptimizerUiState &
  Pick<
    NumericInputSetters,
    | "setEpochValuesInput"
    | "setLearningRatesInput"
    | "setTestSizesInput"
    | "setBatchSizesInput"
    | "setHiddenDimsInput"
    | "setNumHiddenLayersInput"
    | "setDropoutsInput"
  > & {
    setRunSweepEnabled: (value: boolean) => void;
  };

export type HandleCopyTrainingRunsArgs = {
  trainingRuns: TrainingRunRow[];
  setCopyRunsStatus: (value: string | null) => void;
};
