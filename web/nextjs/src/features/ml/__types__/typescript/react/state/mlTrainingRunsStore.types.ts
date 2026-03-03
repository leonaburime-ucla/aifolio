import type { TrainingRunRow } from "@/features/ml/__types__/typescript/utils/trainingRuns.types";

export type MlTrainingRunsState = {
  trainingRuns: TrainingRunRow[];
};

export type MlTrainingRunsActions = {
  setTrainingRuns: (runs: TrainingRunRow[]) => void;
  prependTrainingRun: (run: TrainingRunRow) => void;
  clearTrainingRuns: () => void;
};

export type MlTrainingRunsStore = MlTrainingRunsState & MlTrainingRunsActions;
