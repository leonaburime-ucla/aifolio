import type { useMlTrainingUiBaseState } from "@/features/ml/typescript/react/hooks/ml.hooks.base";
import type { getTrainingDefaults } from "@/features/ml/typescript/config/datasetTrainingDefaults";
import type { createDefaultTrainingRuntime } from "@/features/ml/typescript/logic/trainingRuntime.logic";
import type {
  buildSweepCombinations,
  validateBatchSizes,
  validateDropouts,
  validateEpochValues,
  validateHiddenDims,
  validateLearningRates,
  validateNumHiddenLayers,
  validateTestSizes,
} from "@/features/ml/typescript/validators/trainingSweep.validators";
import type { formatCompletedAt, formatMetricNumber } from "@/features/ml/typescript/utils/trainingRuns.util";
import type { parseNumericValue } from "@/features/ml/typescript/utils/trainingUiShared";
import type {
  resolveTargetColumn,
  resolveTeacherRunKey,
  splitColumnInput,
  validateTrainingSetup,
} from "@/features/ml/typescript/logic/trainingInputValidation.logic";
import type {
  calculatePlannedRunCount,
  hasTeacherModelReference,
  isCompletedRunForMode,
} from "@/features/ml/typescript/logic/trainingHookDecisions.logic";
import type {
  createReloadSweepValuesHandler,
  createToggleRunSweepHandler,
  handleApplyOptimalParams,
  handleCopyTrainingRuns,
  handleFindOptimalParams,
} from "@/features/ml/typescript/logic/trainingShared.logic";
import type {
  buildDistillationComparison,
  buildEnrichedDistilledRun,
  resolveDistilledModalPayload,
} from "@/features/ml/typescript/logic/distillationView.logic";

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
