import type { TrainingRunRow } from "@/features/ml/__types__/typescript/utils/trainingRuns.types";

/**
 * Shared training-runs store contract used by ML integration hooks.
 */
export type TrainingRunsState = {
  trainingRuns: TrainingRunRow[];
  prependTrainingRun: (row: TrainingRunRow) => void;
  clearTrainingRuns: () => void;
};

/**
 * Hook signature for pluggable training-run state providers.
 */
export type UseTrainingRunsState = () => TrainingRunsState;

/**
 * Arguments required to compose feature-level dataset + UI + logic integration.
 */
export type IntegrationComposeArgs<
  TDatasetState extends object,
  TUiState extends object,
  TLogic extends object,
> = {
  useDatasetState: () => TDatasetState;
  useUiState: () => TUiState;
  useLogic: (args: {
    dataset: TDatasetState;
    trainingRuns: TrainingRunRow[];
    prependTrainingRun: (row: TrainingRunRow) => void;
    ui: TUiState;
  }) => TLogic;
  useTrainingRunsState: UseTrainingRunsState;
};
