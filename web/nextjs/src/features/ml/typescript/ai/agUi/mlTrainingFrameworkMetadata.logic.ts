import type { CopilotActionParameter } from "@/features/ml/typescript/ai/agUi/mlTrainingToolActions.logic";

export type MlTrainingFrameworkId = "pytorch" | "tensorflow";

export type MlTrainingFrameworkMetadata = {
  id: MlTrainingFrameworkId;
  label: "PyTorch" | "TensorFlow";
  tab: MlTrainingFrameworkId;
  targetSelector: string;
  trainParameters: CopilotActionParameter[];
  setFieldsParameters: CopilotActionParameter[];
  randomizeParameters: CopilotActionParameter[];
};

const COMMON_TRAIN_PARAMETERS: CopilotActionParameter[] = [
  { name: "dataset_id", type: "string", required: true, description: "Dataset file id from /ml-data." },
  { name: "target_column", type: "string", required: true, description: "Target column name in the selected dataset." },
  { name: "task", type: "string", required: false, description: "Optional task hint: classification, regression, or auto." },
  { name: "epochs", type: "number", required: false, description: "Optional training epochs." },
  { name: "batch_size", type: "number", required: false, description: "Optional batch size." },
  { name: "learning_rate", type: "number", required: false, description: "Optional learning rate." },
];

const COMMON_RANDOMIZE_PARAMETERS: CopilotActionParameter[] = [
  { name: "style", type: "string", required: false, description: "Randomization profile: safe, balanced, or aggressive." },
  { name: "set_sweep_values", type: "boolean", required: false, description: "Preferred: explicitly enable/disable sweep values." },
  { name: "run_sweep", type: "boolean", required: false, description: "Legacy alias for set_sweep_values." },
  { name: "auto_distill", type: "boolean", required: false, description: "Optional explicit auto_distill setting." },
  { name: "lock_target_column", type: "boolean", required: false, description: "If true, preserve current target column." },
  { name: "randomize_model_fields", type: "boolean", required: false, description: "If true, also randomize training_mode, target_column, and task." },
];

function createSetFieldsParameters(frameworkId: MlTrainingFrameworkId): CopilotActionParameter[] {
  return [
    {
      name: "fields",
      type: "object",
      required: true,
      description:
        `Object patch for ${frameworkId}_* fields, e.g. {training_mode, target_column, task, epoch_values, batch_sizes, learning_rates, test_sizes, hidden_dims, num_hidden_layers, dropouts, exclude_columns, date_columns, set_sweep_values, auto_distill}.`,
    },
  ];
}

export const ML_TRAINING_FRAMEWORKS: Record<MlTrainingFrameworkId, MlTrainingFrameworkMetadata> = {
  pytorch: {
    id: "pytorch",
    label: "PyTorch",
    tab: "pytorch",
    targetSelector: '[data-ai-field="pytorch_target_column"]',
    trainParameters: COMMON_TRAIN_PARAMETERS,
    setFieldsParameters: createSetFieldsParameters("pytorch"),
    randomizeParameters: COMMON_RANDOMIZE_PARAMETERS,
  },
  tensorflow: {
    id: "tensorflow",
    label: "TensorFlow",
    tab: "tensorflow",
    targetSelector: '[data-ai-field="tensorflow_target_column"]',
    trainParameters: COMMON_TRAIN_PARAMETERS,
    setFieldsParameters: createSetFieldsParameters("tensorflow"),
    randomizeParameters: COMMON_RANDOMIZE_PARAMETERS,
  },
};
