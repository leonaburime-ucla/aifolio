import { calcTrainingTableHeight, buildDistillActionModel } from "@aifolio/frontend-core/ml-training";
import type { UseTrainingRunsSectionModelDeps } from "@/features/ml/react/hooks/useTrainingRunsSectionModel.hooks.types";

/**
 * Default dependency wiring for `useTrainingRunsSectionModel`.
 */
export const DEFAULT_TRAINING_RUNS_SECTION_MODEL_DEPS: UseTrainingRunsSectionModelDeps = {
  calcTrainingTableHeight,
  buildDistillActionModel,
};
