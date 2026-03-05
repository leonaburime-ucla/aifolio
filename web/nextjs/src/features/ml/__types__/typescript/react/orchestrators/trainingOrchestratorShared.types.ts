import type { MlTaskType } from "@/features/ml/__types__/typescript/config/datasetTrainingDefaults.types";
import type { SweepCombination } from "@/features/ml/__types__/typescript/validators/trainingSweep.types";
import type { TrainingMetrics, TrainingRunRow } from "@/features/ml/__types__/typescript/utils/trainingRuns.types";

export type TrainingProblemBase = {
  datasetId: string;
  targetColumn: string;
  task: MlTaskType;
  trainingMode: string;
  isLinearBaselineMode: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  combinations: SweepCombination[];
};

export type TrainModelRequestBase = {
  dataset_id: string;
  target_column: string;
  training_mode?: string;
  save_model?: boolean;
  exclude_columns?: string[];
  date_columns?: string[];
  task?: MlTaskType;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  test_size?: number;
  hidden_dim?: number;
  num_hidden_layers?: number;
  dropout?: number;
};

export type TrainModelResultBase = {
  status: "ok" | "error";
  run_id?: string;
  model_id?: string;
  model_path?: string;
  metrics?: unknown;
  error?: string;
};

export type TrainingDepsBase<
  TTrainRequest extends TrainModelRequestBase,
  TTrainResult extends TrainModelResultBase,
> = {
  trainModel: (payload: TTrainRequest) => Promise<TTrainResult>;
  prependTrainingRun: (row: TrainingRunRow) => void;
  onProgress: (current: number, total: number) => void;
  formatCompletedAt: (params: { date?: Date }, options?: Record<string, never>) => string;
  formatMetricNumber: (params: { value: unknown }, options?: Record<string, never>) => string;
  shouldContinue?: () => boolean;
};

export type DistillationProblemBase = {
  datasetId: string;
  targetColumn: string;
  task: MlTaskType;
  trainingMode: string;
  saveDistilledModel: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  teacher: {
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
};

export type DistillModelRequestBase = {
  dataset_id: string;
  target_column: string;
  training_mode?: string;
  save_model?: boolean;
  teacher_run_id?: string;
  teacher_model_id?: string;
  teacher_model_path?: string;
  exclude_columns?: string[];
  date_columns?: string[];
  task?: MlTaskType;
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

export type DistillModelResultBase = {
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

export type DistillationDepsBase<
  TDistillRequest extends DistillModelRequestBase,
  TDistillResult extends DistillModelResultBase,
> = {
  distillModel: (payload: TDistillRequest) => Promise<TDistillResult>;
  formatCompletedAt: (params: { date?: Date }, options?: Record<string, never>) => string;
  formatMetricNumber: (params: { value: unknown }, options?: Record<string, never>) => string;
};

export type TrainingSweepResult = {
  stopped: boolean;
  completed: number;
  total: number;
  completedTeacherRuns: TrainingRunRow[];
  failedRuns: number;
  firstFailureMessage: string | null;
};

export type DistillationSuccessResult = {
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

export type DistillationRunResult =
  | { status: "error"; error: string }
  | DistillationSuccessResult;
