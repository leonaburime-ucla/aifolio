import type {
  TensorflowDistillRequest,
  TensorflowTrainRequest,
} from "@/features/ml/__types__/typescript/api/tensorflowApi.types";
export type { TensorflowTrainingMode } from "@/features/ml/__types__/typescript/api/tensorflowApi.types";
import type { TensorflowTrainingMode } from "@/features/ml/__types__/typescript/api/tensorflowApi.types";
import type { MlTaskType } from "@/features/ml/__types__/typescript/config/datasetTrainingDefaults.types";
import type { SweepCombination } from "@/features/ml/__types__/typescript/validators/trainingSweep.types";
import type {
  TrainingMetrics,
  TrainingRunRow,
} from "@/features/ml/__types__/typescript/utils/trainingRuns.types";

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
