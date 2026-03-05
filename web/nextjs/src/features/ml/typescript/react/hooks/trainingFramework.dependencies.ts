import { getTrainingDefaults } from "@/features/ml/typescript/config/datasetTrainingDefaults";
import { createDefaultTrainingRuntime } from "@/features/ml/typescript/logic/trainingRuntime.logic";
import {
  buildSweepCombinations,
  validateBatchSizes,
  validateDropouts,
  validateEpochValues,
  validateHiddenDims,
  validateLearningRates,
  validateNumHiddenLayers,
  validateTestSizes,
} from "@/features/ml/typescript/validators/trainingSweep.validators";
import {
  formatCompletedAt,
  formatMetricNumber,
} from "@/features/ml/typescript/utils/trainingRuns.util";
import { parseNumericValue } from "@/features/ml/typescript/utils/trainingUiShared";
import {
  resolveTargetColumn,
  resolveTeacherRunKey,
  splitColumnInput,
  validateTrainingSetup,
} from "@/features/ml/typescript/logic/trainingInputValidation.logic";
import {
  calculatePlannedRunCount,
  hasTeacherModelReference,
  isCompletedRunForMode,
} from "@/features/ml/typescript/logic/trainingHookDecisions.logic";
import { useMlTrainingUiBaseState } from "@/features/ml/typescript/react/hooks/ml.hooks.base";
import {
  createReloadSweepValuesHandler,
  createToggleRunSweepHandler,
  handleApplyOptimalParams,
  handleCopyTrainingRuns,
  handleFindOptimalParams,
} from "@/features/ml/typescript/logic/trainingShared.logic";
import {
  buildDistillationComparison,
  buildEnrichedDistilledRun,
  resolveDistilledModalPayload,
} from "@/features/ml/typescript/logic/distillationView.logic";
import type {
  TrainingFrameworkLogicDeps,
  TrainingFrameworkUiDeps,
} from "@/features/ml/__types__/typescript/react/hooks/trainingFramework.dependencies.types";

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
