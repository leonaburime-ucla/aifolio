/**
 * Spec: ml-training.ui.spec.ts
 * Version: 1.1.0
 */
export const ML_TRAINING_UI_SPEC_VERSION = "1.1.0";

export const mlTrainingUiSpec = {
  id: "ml-training.ui",
  version: ML_TRAINING_UI_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  surfaces: [
    "PyTorchPageConsumer",
    "TensorFlowPageConsumer",
    "TrainingRunsSection",
    "MlTrainingModals",
    "FieldHelp",
  ],
  controlledContracts: [
    {
      component: "TrainingRunsSection",
      requiredProps: ["trainingRuns", "copyRunsStatus", "onCopyTrainingRuns", "onClearTrainingRuns"],
      behavior: "Renders DataTable when runs exist; renders empty text when none.",
    },
    {
      component: "MlTrainingModals",
      requiredProps: ["isOpen", "onClose"],
      behavior: "Modals are presentation-only and render metrics/params from props.",
    },
  ],
  rules: [
    "Consumers receive orchestrator prop defaults to concrete integration orchestrators.",
    "Utility modules must not export JSX components.",
  ],
} as const;
