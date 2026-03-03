import type { ValidationResult } from "@/features/ml/__types__/typescript/validators/trainingSweep.types";
import type { TrainingRunRow } from "@/features/ml/__types__/typescript/utils/trainingRuns.types";

export type TrainingSweepValidations = {
  epochsValidation: ValidationResult<number>;
  testSizesValidation: ValidationResult<number>;
  learningRatesValidation: ValidationResult<number>;
  batchSizesValidation: ValidationResult<number>;
  hiddenDimsValidation: ValidationResult<number>;
  numHiddenLayersValidation: ValidationResult<number>;
  dropoutsValidation: ValidationResult<number>;
};

export type CalculatePlannedRunCountParams = {
  isLinearBaselineMode: boolean;
  validations: TrainingSweepValidations;
};

export type IsCompletedRunForModeParams = {
  run: TrainingRunRow;
  mode: string;
};

export type HasTeacherModelReferenceParams = {
  runId: string;
  modelId: string;
  modelPath: string;
};
