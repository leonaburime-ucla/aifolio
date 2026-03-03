import { useMlTrainingRunsStore } from "@/features/ml/typescript/react/state/zustand/mlTrainingRunsStore";

/**
 * Adapter that exposes ML training-runs state/actions behind an injectable hook contract.
 */
export function useMlTrainingRunsAdapter() {
  const trainingRuns = useMlTrainingRunsStore((state) => state.trainingRuns);
  const prependTrainingRun = useMlTrainingRunsStore((state) => state.prependTrainingRun);
  const clearTrainingRuns = useMlTrainingRunsStore((state) => state.clearTrainingRuns);

  return {
    trainingRuns,
    prependTrainingRun,
    clearTrainingRuns,
  };
}
