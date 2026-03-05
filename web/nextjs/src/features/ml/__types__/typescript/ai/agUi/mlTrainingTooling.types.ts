/**
 * Shared ML training-tool contracts consumed by AG-UI orchestration.
 *
 * Ownership:
 * - This file lives in `features/ml` because it models framework training
 *   payloads/bridges rather than chat-route concerns.
 */

export type MlTask = "auto" | "classification" | "regression";
export type MlListField = string | number | number[];
export type MlMode = string;

export type PytorchTrainingMode =
  | "mlp_dense"
  | "linear_glm_baseline"
  | "tabresnet"
  | "imbalance_aware"
  | "calibrated_classifier"
  | "tree_teacher_distillation";

export type TensorflowTrainingMode =
  | "mlp_dense"
  | "linear_glm_baseline"
  | "wide_and_deep"
  | "imbalance_aware"
  | "quantile_regression"
  | "calibrated_classifier"
  | "entity_embeddings"
  | "autoencoder_head"
  | "multi_task_learning"
  | "time_aware_tabular";

export type MlFormPatch<TMode extends MlMode = MlMode> = {
  training_mode?: TMode;
  target_column?: string;
  task?: MlTask;
  epoch_values?: MlListField;
  batch_sizes?: MlListField;
  learning_rates?: MlListField;
  test_sizes?: MlListField;
  hidden_dims?: MlListField;
  num_hidden_layers?: MlListField;
  dropouts?: MlListField;
  exclude_columns?: string | string[];
  date_columns?: string | string[];
  set_sweep_values?: boolean;
  run_sweep?: boolean;
  auto_distill?: boolean;
};

export type PytorchFormPatch = MlFormPatch<PytorchTrainingMode>;
export type TensorflowFormPatch = MlFormPatch<TensorflowTrainingMode>;

export type MlFormRandomizeArgs = {
  style?: "safe" | "balanced" | "aggressive";
  set_sweep_values?: boolean;
  run_sweep?: boolean;
  auto_distill?: boolean;
  lock_target_column?: boolean;
  randomize_model_fields?: boolean;
};

export type PytorchRandomizeArgs = MlFormRandomizeArgs;
export type TensorflowRandomizeArgs = MlFormRandomizeArgs;

export type MlFormPatchResult = { applied: string[]; skipped: string[] };
export type MlFormStartResult =
  | { status: "ok" }
  | { status: "error"; reason: string };

export type MlFormBridge<TPatch extends MlFormPatch = MlFormPatch> = {
  applyPatch: (patch: TPatch) => MlFormPatchResult;
  startTrainingRuns?: () => Promise<MlFormStartResult>;
};

export type PytorchFormBridge = MlFormBridge<PytorchFormPatch>;
export type TensorflowFormBridge = MlFormBridge<TensorflowFormPatch>;

export type TrainPytorchModelArgs = {
  dataset_id: string;
  target_column: string;
  task?: MlTask;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
};

export type TrainTensorflowModelArgs = {
  dataset_id: string;
  target_column: string;
  task?: MlTask;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
};

export type MlFrameworkTab = "pytorch" | "tensorflow";

export type EnsureFrameworkTabArgs = {
  activeTab: string;
  setActiveTab: (tab: MlFrameworkTab) => void;
  pushRoute: (route: string) => void;
  frameworkTab: MlFrameworkTab;
  waitForFrameworkForm: () => Promise<boolean>;
};
