import type {
  ResolveTargetColumnParams,
  ResolveTeacherRunKeyParams,
  ValidateTrainingSetupParams,
} from "@/features/ml/__types__/typescript/logic/trainingInputValidation.types";

/**
 * Splits a comma-delimited column input into normalized tokens.
 * @param params - Required parameters.
 * @param params.value - Raw delimited value.
 * @returns Trimmed non-empty tokens.
 */
export function splitColumnInput({
  value,
}: {
  value: string;
}): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

/**
 * Resolves target column from explicit input, defaults, and fallback table columns.
 * @param params - Required parameters.
 * @returns Normalized target column.
 */
export function resolveTargetColumn({
  targetColumn,
  defaultTargetColumn,
  tableColumns,
}: ResolveTargetColumnParams): string {
  return targetColumn.trim() || defaultTargetColumn || tableColumns[0] || "";
}

/**
 * Validates high-level training setup before building sweep combinations.
 * @param params - Required parameters.
 * @returns User-safe validation error or `null` when valid.
 */
export function validateTrainingSetup({
  selectedDatasetId,
  resolvedTargetColumn,
  excludeColumns,
  dateColumns,
  isLinearBaselineMode,
  validations,
}: ValidateTrainingSetupParams): string | null {
  if (!selectedDatasetId) {
    return "Please select a dataset first.";
  }
  if (excludeColumns.includes(resolvedTargetColumn.trim())) {
    return "Target column cannot also be in excluded columns.";
  }
  if (dateColumns.includes(resolvedTargetColumn.trim())) {
    return "Target column cannot also be in date columns.";
  }
  const overlap = dateColumns.find((col) => excludeColumns.includes(col));
  if (overlap) {
    return `Column '${overlap}' cannot be in both excluded and date columns.`;
  }
  if (!resolvedTargetColumn.trim()) {
    return "Please provide a target column.";
  }
  if (!validations.epochsValidation.ok) {
    return validations.epochsValidation.error;
  }
  if (!validations.testSizesValidation.ok) {
    return validations.testSizesValidation.error;
  }
  if (!validations.learningRatesValidation.ok) {
    return validations.learningRatesValidation.error;
  }
  if (!validations.batchSizesValidation.ok) {
    return validations.batchSizesValidation.error;
  }
  if (!isLinearBaselineMode && !validations.hiddenDimsValidation.ok) {
    return validations.hiddenDimsValidation.error;
  }
  if (!isLinearBaselineMode && !validations.numHiddenLayersValidation.ok) {
    return validations.numHiddenLayersValidation.error;
  }
  if (!isLinearBaselineMode && !validations.dropoutsValidation.ok) {
    return validations.dropoutsValidation.error;
  }

  return null;
}

/**
 * Resolves a stable teacher key from run/model identifiers.
 * @param params - Required parameters.
 * @returns Key string or empty string when unavailable.
 */
export function resolveTeacherRunKey({
  run,
}: ResolveTeacherRunKeyParams): string {
  return (
    String(run.run_id ?? "") ||
    String(run.model_id ?? "") ||
    String(run.model_path ?? "") ||
    String(run.completed_at ?? "run")
  );
}
