import { createDefaultTrainingRuntime } from "@/features/ml/logic/trainingRuntime.logic";
import {
  formatCompletedAt,
  formatMetricNumber,
} from "@aifolio/frontend-core/ml-training";
import { useMlTrainingUiBaseState } from "@/features/ml/react/hooks/ml.hooks.base";
import {
  getTrainingDefaults,
  buildSweepCombinations,
  validateBatchSizes,
  validateDropouts,
  validateEpochValues,
  validateHiddenDims,
  validateLearningRates,
  validateNumHiddenLayers,
  validateTestSizes,
  parseNumericValue,
  resolveTargetColumn,
  resolveTeacherRunKey,
  splitColumnInput,
  validateTrainingSetup,
  calculatePlannedRunCount,
  hasTeacherModelReference,
  isCompletedRunForMode,
  createReloadSweepValuesHandler,
  createToggleRunSweepHandler,
  handleApplyOptimalParams,
  handleCopyTrainingRuns,
  handleFindOptimalParams,
} from "@aifolio/frontend-core/ml-training";
import {
  buildDistillationComparison,
  buildEnrichedDistilledRun,
  resolveDistilledModalPayload,
} from "@aifolio/frontend-core/ml-training";
import type {
  TrainingFrameworkLogicDeps,
  TrainingFrameworkUiDeps,
} from "@/features/ml/react/hooks/trainingFramework.dependencies.types";

export const DEFAULT_TRAINING_FRAMEWORK_UI_DEPS: TrainingFrameworkUiDeps = {
  useBaseUiState: useMlTrainingUiBaseState,
};

export const DEFAULT_TRAINING_FRAMEWORK_LOGIC_DEPS: TrainingFrameworkLogicDeps = {
  getTrainingDefaults,
  createDefaultTrainingRuntime,
  buildSweepCombinations,
  validateBatchSizes,
  validateDropouts,
  validateEpochValues,
  validateHiddenDims,
  validateLearningRates,
  validateNumHiddenLayers,
  validateTestSizes,
  formatCompletedAt,
  formatMetricNumber,
  parseNumericValue,
  resolveTargetColumn,
  resolveTeacherRunKey,
  splitColumnInput,
  validateTrainingSetup,
  calculatePlannedRunCount,
  hasTeacherModelReference,
  isCompletedRunForMode,
  createReloadSweepValuesHandler,
  createToggleRunSweepHandler,
  handleApplyOptimalParams,
  handleCopyTrainingRuns,
  handleFindOptimalParams,
  buildDistillationComparison,
  buildEnrichedDistilledRun,
  resolveDistilledModalPayload,
};
