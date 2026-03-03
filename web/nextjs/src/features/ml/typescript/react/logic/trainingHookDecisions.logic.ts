import { parseNumericValue } from "@/features/ml/typescript/utils/trainingUiShared";
import type {
  CalculatePlannedRunCountParams,
  HasTeacherModelReferenceParams,
  IsCompletedRunForModeParams,
} from "@/features/ml/__types__/typescript/react/logic/trainingHookDecisions.types";

/**
 * Returns whether all required sweep validations are usable for planning runs.
 * @param params - Required parameters.
 * @param params.isLinearBaselineMode - Whether hidden-layer validations should be ignored.
 * @param params.validations - Validation results for each sweep dimension.
 * @returns `true` when all required validations are valid.
 */
export function hasValidSweepInputs(
  { isLinearBaselineMode, validations }: CalculatePlannedRunCountParams,
  {}: Record<string, never> = {}
): boolean {
  if (!validations.epochsValidation.ok) return false;
  if (!validations.testSizesValidation.ok) return false;
  if (!validations.learningRatesValidation.ok) return false;
  if (!validations.batchSizesValidation.ok) return false;
  if (!isLinearBaselineMode && !validations.hiddenDimsValidation.ok) return false;
  if (!isLinearBaselineMode && !validations.numHiddenLayersValidation.ok) return false;
  if (!isLinearBaselineMode && !validations.dropoutsValidation.ok) return false;
  return true;
}

/**
 * Computes the number of planned runs from validated sweep dimensions.
 * @param params - Required parameters.
 * @returns Planned run count, or `0` when validations are not usable.
 */
export function calculatePlannedRunCount(
  { isLinearBaselineMode, validations }: CalculatePlannedRunCountParams,
  {}: Record<string, never> = {}
): number {
  if (!hasValidSweepInputs({ isLinearBaselineMode, validations })) {
    return 0;
  }

  const deepHiddenDimCount = validations.hiddenDimsValidation.ok
    ? validations.hiddenDimsValidation.values.length
    : 0;
  const deepLayerCount = validations.numHiddenLayersValidation.ok
    ? validations.numHiddenLayersValidation.values.length
    : 0;
  const deepDropoutCount = validations.dropoutsValidation.ok
    ? validations.dropoutsValidation.values.length
    : 0;

  return (
    validations.epochsValidation.values.length *
    validations.testSizesValidation.values.length *
    validations.learningRatesValidation.values.length *
    validations.batchSizesValidation.values.length *
    (isLinearBaselineMode ? 1 : deepHiddenDimCount) *
    (isLinearBaselineMode ? 1 : deepLayerCount) *
    (isLinearBaselineMode ? 1 : deepDropoutCount)
  );
}

/**
 * Checks whether a run is completed for the active mode and has numeric metric data.
 * @param params - Required parameters.
 * @returns `true` for completed runs suitable for optimization and distillation UI logic.
 */
export function isCompletedRunForMode(
  { run, mode }: IsCompletedRunForModeParams,
  {}: Record<string, never> = {}
): boolean {
  if (String(run.result ?? "") !== "completed") return false;
  if (String(run.training_mode ?? "") !== mode) return false;
  const metricName = String(run.metric_name ?? "").toLowerCase();
  if (metricName === "n/a") return false;
  return parseNumericValue({ value: run.metric_score }) !== null;
}

/**
 * Checks whether a run has at least one usable teacher model reference.
 * @param params - Required parameters.
 * @returns `true` when a run/model/path identifier exists and is not `n/a`.
 */
export function hasTeacherModelReference(
  { runId, modelId, modelPath }: HasTeacherModelReferenceParams,
  {}: Record<string, never> = {}
): boolean {
  return (
    (runId.length > 0 && runId !== "n/a") ||
    (modelId.length > 0 && modelId !== "n/a") ||
    (modelPath.length > 0 && modelPath !== "n/a")
  );
}

export type { TrainingSweepValidations } from "@/features/ml/__types__/typescript/react/logic/trainingHookDecisions.types";
