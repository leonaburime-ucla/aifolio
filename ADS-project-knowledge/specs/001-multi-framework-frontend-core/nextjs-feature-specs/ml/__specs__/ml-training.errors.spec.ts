/**
 * Spec: ml-training.errors.spec.ts
 * Version: 1.1.0
 */
export const ML_TRAINING_ERRORS_SPEC_VERSION = "1.1.0";

export const mlTrainingErrorsSpec = {
  id: "ml-training.errors",
  version: ML_TRAINING_ERRORS_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  registry: [
    { code: "PYTORCH_TRAIN_FAILED", retryable: true, source: "trainPytorchModel" },
    { code: "PYTORCH_TRAIN_REQUEST_FAILED", retryable: true, source: "trainPytorchModel" },
    { code: "PYTORCH_DISTILL_FAILED", retryable: true, source: "distillPytorchModel" },
    { code: "PYTORCH_DISTILL_REQUEST_FAILED", retryable: true, source: "distillPytorchModel" },
    { code: "TENSORFLOW_TRAIN_FAILED", retryable: true, source: "trainTensorflowModel" },
    { code: "TENSORFLOW_TRAIN_REQUEST_FAILED", retryable: true, source: "trainTensorflowModel" },
    { code: "TENSORFLOW_DISTILL_FAILED", retryable: true, source: "distillTensorflowModel" },
    { code: "TENSORFLOW_DISTILL_REQUEST_FAILED", retryable: true, source: "distillTensorflowModel" },
  ],
  mappingRules: [
    "Train/distill API wrappers return status='error' objects with stable code values.",
    "Dataset orchestrator surfaces user-safe string errors via state when manifest/row fetch throws.",
    "Validation failures return explicit user-safe messages via ValidationResult.error.",
  ],
} as const;
