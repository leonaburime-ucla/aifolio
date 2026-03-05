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
  cellRenderers: TrainingRunsCellRenderers;
};

export type TrainingRunsCellRenderers = {
  distill_action: (_value: unknown, row: TrainingRunRow) => unknown;
};

export type UseTrainingRunsSectionModelDeps = {
  calcTrainingTableHeight: (args: { rowsCount: number }) => number;
  buildDistillActionModel: (args: {
    row: TrainingRunRow;
    isDistillationSupportedForRun?: (row: TrainingRunRow) => boolean;
    distillingTeacherKey: string | null;
    distilledByTeacher: Record<string, string>;
  }) => {
    kind: "student_model" | "not_available" | "show_distilled" | "distill";
    isDistillingThisRow: boolean;
  };
};
