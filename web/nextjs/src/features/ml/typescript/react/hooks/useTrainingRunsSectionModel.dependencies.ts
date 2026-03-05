import { calcTrainingTableHeight } from "@/features/ml/typescript/utils/trainingRuns.util";
import { buildDistillActionModel } from "@/features/ml/typescript/logic/trainingRunsSection.logic";
import type { UseTrainingRunsSectionModelDeps } from "@/features/ml/__types__/typescript/react/hooks/trainingRunsSection.hooks.types";

/**
 * Default dependency wiring for `useTrainingRunsSectionModel`.
 */
export const DEFAULT_TRAINING_RUNS_SECTION_MODEL_DEPS: UseTrainingRunsSectionModelDeps = {
  calcTrainingTableHeight,
  buildDistillActionModel,
};
