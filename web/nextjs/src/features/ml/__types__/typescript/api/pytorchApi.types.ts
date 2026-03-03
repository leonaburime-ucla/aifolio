export type PytorchTrainingMode =
  | "mlp_dense"
  | "linear_glm_baseline"
  | "tabresnet"
  | "imbalance_aware"
  | "calibrated_classifier"
  | "tree_teacher_distillation";

export type PytorchTrainRequest = {
  dataset_id: string;
  target_column: string;
  training_mode?: PytorchTrainingMode;
  save_model?: boolean;
  exclude_columns?: string[];
  date_columns?: string[];
  task?: "classification" | "regression" | "auto";
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  test_size?: number;
  hidden_dim?: number;
  num_hidden_layers?: number;
  dropout?: number;
};

export type PytorchTrainSuccess = {
  status: "ok";
  run_id?: string;
  model_id?: string;
  model_path?: string;
  metrics?: unknown;
  teacher_input_dim?: number | null;
  teacher_output_dim?: number | null;
  student_input_dim?: number | null;
  student_output_dim?: number | null;
  teacher_model_size_bytes?: number | null;
  student_model_size_bytes?: number | null;
  size_saved_bytes?: number | null;
  size_saved_percent?: number | null;
  teacher_param_count?: number | null;
  student_param_count?: number | null;
  param_saved_count?: number | null;
  param_saved_percent?: number | null;
};

export type PytorchTrainError = {
  status: "error";
  code: string;
  error: string;
};

export type PytorchDistillRequest = {
  dataset_id: string;
  target_column: string;
  training_mode?: PytorchTrainingMode;
  save_model?: boolean;
  teacher_run_id?: string;
  teacher_model_id?: string;
  teacher_model_path?: string;
  exclude_columns?: string[];
  date_columns?: string[];
  task?: "classification" | "regression" | "auto";
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  test_size?: number;
  temperature?: number;
  alpha?: number;
  student_hidden_dim?: number;
  student_num_hidden_layers?: number;
  student_dropout?: number;
};

export type PytorchApiRuntime = {
  fetchImpl: typeof fetch;
  resolveBaseUrl: () => string;
  scheduleTimeout: typeof setTimeout;
  clearScheduledTimeout: typeof clearTimeout;
};
