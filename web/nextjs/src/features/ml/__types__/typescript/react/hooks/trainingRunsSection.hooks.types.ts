import type { TrainingRunRow } from "@/features/ml/__types__/typescript/utils/trainingRuns.types";
import type { TrainingRunsSectionProps } from "@/features/ml/__types__/typescript/react/views/trainingRunsSection.types";

export type UseTrainingRunsSectionModelParams = Pick<
  TrainingRunsSectionProps,
  | "trainingRuns"
  | "onDistillFromRun"
  | "onSeeDistilledFromRun"
  | "isDistillationSupportedForRun"
  | "distillingTeacherKey"
  | "distilledByTeacher"
>;

export type UseTrainingRunsSectionModelResult = {
  trainingTableHeight: number;
  cellRenderers: {
    distill_action: (_value: unknown, row: TrainingRunRow) => unknown;
  };
};
