/**
 * AG-UI facade over ML-owned training tool adapters.
 *
 * Rationale:
 * - Framework training behavior is implemented in `features/ml`.
 * - AG-UI imports this facade so route-level orchestration stays stable.
 */

export {
  buildRandomPytorchFormPatch,
  buildRandomTensorflowFormPatch,
  handleRandomizePytorchFormFields,
  handleRandomizeTensorflowFormFields,
  handleSetPytorchFormFields,
  handleSetTensorflowFormFields,
  handleStartPytorchTrainingRuns,
  handleStartTensorflowTrainingRuns,
  handleTrainPytorchModel,
  handleTrainTensorflowModel,
} from "@/features/ml/typescript/ai/agUi/mlTrainingTools.adapter";
