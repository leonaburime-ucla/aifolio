import type { useMlTrainingUiBaseState } from "@/features/ml/react/hooks/ml.hooks.base";
import type { createDefaultTrainingRuntime } from "@/features/ml/logic/trainingRuntime.logic";
import type { formatCompletedAt, formatMetricNumber } from "@aifolio/frontend-core/ml-training";
import type {
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
import type {
  buildDistillationComparison,
  buildEnrichedDistilledRun,
  resolveDistilledModalPayload,
} from "@aifolio/frontend-core/ml-training";

/**
 * Dependency contracts for `trainingFramework.hooks.ts` default wiring.
 */
export type TrainingFrameworkUiDeps = {
  useBaseUiState: typeof useMlTrainingUiBaseState;
};

export type TrainingFrameworkLogicDeps = {
  getTrainingDefaults: typeof getTrainingDefaults;
  createDefaultTrainingRuntime: typeof createDefaultTrainingRuntime;
  buildSweepCombinations: typeof buildSweepCombinations;
  validateBatchSizes: typeof validateBatchSizes;
  validateDropouts: typeof validateDropouts;
  validateEpochValues: typeof validateEpochValues;
  validateHiddenDims: typeof validateHiddenDims;
  validateLearningRates: typeof validateLearningRates;
  validateNumHiddenLayers: typeof validateNumHiddenLayers;
  validateTestSizes: typeof validateTestSizes;
  formatCompletedAt: typeof formatCompletedAt;
  formatMetricNumber: typeof formatMetricNumber;
  parseNumericValue: typeof parseNumericValue;
  resolveTargetColumn: typeof resolveTargetColumn;
  resolveTeacherRunKey: typeof resolveTeacherRunKey;
  splitColumnInput: typeof splitColumnInput;
  validateTrainingSetup: typeof validateTrainingSetup;
  calculatePlannedRunCount: typeof calculatePlannedRunCount;
  hasTeacherModelReference: typeof hasTeacherModelReference;
  isCompletedRunForMode: typeof isCompletedRunForMode;
  createReloadSweepValuesHandler: typeof createReloadSweepValuesHandler;
  createToggleRunSweepHandler: typeof createToggleRunSweepHandler;
  handleApplyOptimalParams: typeof handleApplyOptimalParams;
  handleCopyTrainingRuns: typeof handleCopyTrainingRuns;
  handleFindOptimalParams: typeof handleFindOptimalParams;
  buildDistillationComparison: typeof buildDistillationComparison;
  buildEnrichedDistilledRun: typeof buildEnrichedDistilledRun;
  resolveDistilledModalPayload: typeof resolveDistilledModalPayload;
};
