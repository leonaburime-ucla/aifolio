import type { TrainingRunRow } from "@/features/ml/__types__/typescript/utils/trainingRuns.types";
import type { ValidationResult } from "@/features/ml/__types__/typescript/validators/trainingSweep.types";

export type TrainingInputValidations = {
  epochsValidation: ValidationResult<number>;
  testSizesValidation: ValidationResult<number>;
  learningRatesValidation: ValidationResult<number>;
  batchSizesValidation: ValidationResult<number>;
  hiddenDimsValidation: ValidationResult<number>;
  numHiddenLayersValidation: ValidationResult<number>;
  dropoutsValidation: ValidationResult<number>;
};

export type ResolveTargetColumnParams = {
  targetColumn: string;
  defaultTargetColumn: string;
  tableColumns: string[];
};

export type ValidateTrainingSetupParams = {
  selectedDatasetId: string | null;
  resolvedTargetColumn: string;
  excludeColumns: string[];
  dateColumns: string[];
  isLinearBaselineMode: boolean;
  validations: TrainingInputValidations;
};

export type ResolveTeacherRunKeyParams = {
  run: TrainingRunRow;
};
