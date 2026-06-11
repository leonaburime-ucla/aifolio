import type { TrainingRunRow } from "@aifolio/contracts/entities/ml-training";

export type TrainingRunsSectionProps = {
  trainingRuns: TrainingRunRow[];
  copyRunsStatus: string | null;
  isTraining?: boolean;
  isStopRequested?: boolean;
  onCopyTrainingRuns: () => void;
  onClearTrainingRuns: () => void;
  onStopTrainingRuns?: () => void;
  onDistillFromRun?: (run: TrainingRunRow) => void;
  onSeeDistilledFromRun?: (run: TrainingRunRow) => void;
  isDistillationSupportedForRun?: (run: TrainingRunRow) => boolean;
  distillingTeacherKey?: string | null;
  distilledByTeacher?: Record<string, string>;
};
