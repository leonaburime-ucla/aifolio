import { create } from "zustand";
import type { TrainingRunRow } from "@/features/ml/utils/trainingRuns.util";

type MlTrainingRunsState = {
  trainingRuns: TrainingRunRow[];
};

type MlTrainingRunsActions = {
  setTrainingRuns: (runs: TrainingRunRow[]) => void;
  prependTrainingRun: (run: TrainingRunRow) => void;
  clearTrainingRuns: () => void;
};

type MlTrainingRunsStore = MlTrainingRunsState & MlTrainingRunsActions;

export const useMlTrainingRunsStore = create<MlTrainingRunsStore>()((set) => ({
  trainingRuns: [],
  setTrainingRuns: (runs) => set({ trainingRuns: runs }),
  prependTrainingRun: (run) =>
    set((state) => ({ trainingRuns: [run, ...state.trainingRuns] })),
  clearTrainingRuns: () => set({ trainingRuns: [] }),
}));
