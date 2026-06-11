import type {
  MlDatasetOption,
  MlDatasetStatePort,
  SweepCombination,
  TrainingMetrics,
  TrainingRunRow,
  PytorchTrainingMode,
  TensorflowTrainingMode,
} from "../model/types";
import type { MlTaskType } from "../config/types";

export type { PytorchTrainingMode, TensorflowTrainingMode };

export type MlDataManifestEntry = {
  id: string;
  label?: string;
  format?: string;
};

export type MlDataManifestResponse = {
  status?: string;
  datasets?: MlDataManifestEntry[];
};

export type MlDatasetRowsResponse = {
  columns?: string[];
  rows?: Array<Record<string, string | number | null>>;
  rowCount?: number;
  totalRowCount?: number;
};

export type MlDatasetOptionsLoader = () => Promise<MlDatasetOption[]>;
export type MlDatasetRowsLoader = (
  params: { datasetId: string }
) => Promise<MlDatasetRowsResponse>;

export type MlDatasetOrchestratorDeps = {
  useDatasetState?: () => MlDatasetStatePort;
  loadDatasetOptions?: MlDatasetOptionsLoader;
  loadDatasetRows?: MlDatasetRowsLoader;
  autoLoad?: boolean;
};

export type MlDataApiRuntime = {
  fetchImpl: typeof fetch;
  resolveBaseUrl: () => string;
};

// --- PyTorch API ---

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

// --- TensorFlow API ---

export type TensorflowTrainRequest = {
  dataset_id: string;
  target_column: string;
  training_mode?: TensorflowTrainingMode;
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

export type TensorflowTrainSuccess = {
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

export type TensorflowTrainError = {
  status: "error";
  code: string;
  error: string;
};

export type TensorflowDistillRequest = {
  dataset_id: string;
  target_column: string;
  training_mode?: TensorflowTrainingMode;
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

export type TensorflowApiRuntime = {
  fetchImpl: typeof fetch;
  resolveBaseUrl: () => string;
  scheduleTimeout: typeof setTimeout;
  clearScheduledTimeout: typeof clearTimeout;
};

export type PytorchTrainCombo = SweepCombination;

export type RunPytorchTrainingProblem = {
  datasetId: string;
  targetColumn: string;
  task: MlTaskType;
  trainingMode: PytorchTrainingMode;
  isLinearBaselineMode: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  combinations: PytorchTrainCombo[];
};

export type PytorchTrainModelResult = {
  status: "ok" | "error";
  run_id?: string;
  model_id?: string;
  model_path?: string;
  metrics?: unknown;
  error?: string;
};

export type RunPytorchTrainingDeps = {
  trainModel: (payload: PytorchTrainRequest) => Promise<PytorchTrainModelResult>;
  prependTrainingRun: (row: TrainingRunRow) => void;
  onProgress: (current: number, total: number) => void;
  formatCompletedAt: (params: { date?: Date }, options?: Record<string, never>) => string;
  formatMetricNumber: (
    params: { value: unknown },
    options?: Record<string, never>
  ) => string;
  shouldContinue?: () => boolean;
};

export type RunPytorchTrainingResult = {
  stopped: boolean;
  completed: number;
  total: number;
  completedTeacherRuns: TrainingRunRow[];
  failedRuns: number;
  firstFailureMessage: string | null;
};

export type PytorchTeacherConfig = {
  hidden: number;
  layers: number;
  dropout: number;
  epochs: number;
  batch: number;
  learningRate: number;
  testSize: number;
  runId?: string;
  modelId?: string;
  modelPath?: string;
};

export type RunPytorchDistillationProblem = {
  datasetId: string;
  targetColumn: string;
  task: MlTaskType;
  trainingMode: PytorchTrainingMode;
  saveDistilledModel: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  teacher: PytorchTeacherConfig;
};

export type PytorchDistillModelResult = {
  status: "ok" | "error";
  model_id?: string;
  model_path?: string;
  run_id?: string;
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
  error?: string;
};

export type RunPytorchDistillationDeps = {
  distillModel: (payload: PytorchDistillRequest) => Promise<PytorchDistillModelResult>;
  formatCompletedAt: (params: { date?: Date }, options?: Record<string, never>) => string;
  formatMetricNumber: (
    params: { value: unknown },
    options?: Record<string, never>
  ) => string;
};

export type RunPytorchDistillationResult =
  | { status: "error"; error: string }
  | {
      status: "ok";
      metrics: TrainingMetrics;
      modelId: string | null;
      modelPath: string | null;
      runId: string | null;
      teacherModelSizeBytes: number | null;
      studentModelSizeBytes: number | null;
      teacherInputDim: number | null;
      teacherOutputDim: number | null;
      studentInputDim: number | null;
      studentOutputDim: number | null;
      sizeSavedBytes: number | null;
      sizeSavedPercent: number | null;
      teacherParamCount: number | null;
      studentParamCount: number | null;
      paramSavedCount: number | null;
      paramSavedPercent: number | null;
      distilledRun: TrainingRunRow;
    };

export type TensorflowTrainCombo = SweepCombination;

export type RunTensorflowTrainingProblem = {
  datasetId: string;
  targetColumn: string;
  task: MlTaskType;
  trainingMode: TensorflowTrainingMode;
  isLinearBaselineMode: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  combinations: TensorflowTrainCombo[];
};

export type TensorflowTrainModelResult = {
  status: "ok" | "error";
  run_id?: string;
  model_id?: string;
  model_path?: string;
  metrics?: unknown;
  error?: string;
};

export type RunTensorflowTrainingDeps = {
  trainModel: (payload: TensorflowTrainRequest) => Promise<TensorflowTrainModelResult>;
  prependTrainingRun: (row: TrainingRunRow) => void;
  onProgress: (current: number, total: number) => void;
  formatCompletedAt: (params: { date?: Date }, options?: Record<string, never>) => string;
  formatMetricNumber: (
    params: { value: unknown },
    options?: Record<string, never>
  ) => string;
  shouldContinue?: () => boolean;
};

export type RunTensorflowTrainingResult = {
  stopped: boolean;
  completed: number;
  total: number;
  completedTeacherRuns: TrainingRunRow[];
  failedRuns: number;
  firstFailureMessage: string | null;
};

export type TensorflowTeacherConfig = {
  hidden: number;
  layers: number;
  dropout: number;
  epochs: number;
  batch: number;
  learningRate: number;
  testSize: number;
  runId?: string;
  modelId?: string;
  modelPath?: string;
};

export type RunTensorflowDistillationProblem = {
  datasetId: string;
  targetColumn: string;
  task: MlTaskType;
  trainingMode: TensorflowTrainingMode;
  saveDistilledModel: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  teacher: TensorflowTeacherConfig;
};

export type TensorflowDistillModelResult = {
  status: "ok" | "error";
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
  error?: string;
};

export type RunTensorflowDistillationDeps = {
  distillModel: (payload: TensorflowDistillRequest) => Promise<TensorflowDistillModelResult>;
  formatCompletedAt: (params: { date?: Date }, options?: Record<string, never>) => string;
  formatMetricNumber: (
    params: { value: unknown },
    options?: Record<string, never>
  ) => string;
};

export type RunTensorflowDistillationResult =
  | { status: "error"; error: string }
  | {
      status: "ok";
      metrics: TrainingMetrics;
      modelId: string | null;
      modelPath: string | null;
      runId: string | null;
      teacherModelSizeBytes: number | null;
      studentModelSizeBytes: number | null;
      teacherInputDim: number | null;
      teacherOutputDim: number | null;
      studentInputDim: number | null;
      studentOutputDim: number | null;
      sizeSavedBytes: number | null;
      sizeSavedPercent: number | null;
      teacherParamCount: number | null;
      studentParamCount: number | null;
      paramSavedCount: number | null;
      paramSavedPercent: number | null;
      distilledRun: TrainingRunRow;
    };
