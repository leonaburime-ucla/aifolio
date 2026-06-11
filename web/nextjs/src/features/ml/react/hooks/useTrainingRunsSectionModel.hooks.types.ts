import type { TrainingRunRow } from "@aifolio/contracts/entities/ml-training";
import type { ReactNode } from "react";
import type { TrainingRunsSectionProps } from "@/features/ml/react/views/components/TrainingRunsSection.types";

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
  distill_action: (value: string | number | null, row: TrainingRunRow) => ReactNode;
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
