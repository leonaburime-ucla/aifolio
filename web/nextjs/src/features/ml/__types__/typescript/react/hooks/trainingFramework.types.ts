import type { MlTrainingUiBaseState } from "@/features/ml/__types__/typescript/react/hooks/mlTrainingUiBase.types";
import type { BaseTrainingRuntimeDeps } from "@/features/ml/__types__/typescript/logic/trainingRuntime.types";
import type { MlTaskType } from "@/features/ml/__types__/typescript/config/datasetTrainingDefaults.types";
import type { SweepCombination } from "@/features/ml/__types__/typescript/validators/trainingSweep.types";
import type { MlDatasetViewModel } from "@/features/ml/__types__/typescript/react/orchestrators/mlDatasetOrchestrator.types";
import type {
  DistillComparison,
  TrainingMetrics,
  TrainingRunRow,
} from "@/features/ml/__types__/typescript/utils/trainingRuns.types";

/**
 * Framework-agnostic ML training UI contract.
 */
export type FrameworkTrainingUiState<TMode extends string> = MlTrainingUiBaseState & {
  trainingMode: TMode;
  setTrainingMode: (value: TMode) => void;
};

export type CommonTrainingOutcome = {
  stopped: boolean;
  completed: number;
  total: number;
  completedTeacherRuns: TrainingRunRow[];
  failedRuns: number;
  firstFailureMessage: string | null;
};

export type CommonTeacherConfig = {
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

export type CommonDistillationOkResult = {
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

export type CommonDistillationResult =
  | { status: "error"; error: string }
  | CommonDistillationOkResult;

export type CommonTrainingProblem<TMode extends string> = {
  datasetId: string;
  targetColumn: string;
  task: MlTaskType;
  trainingMode: TMode;
  isLinearBaselineMode: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  combinations: SweepCombination[];
};

export type CommonTrainingDeps<TTrainPayload, TTrainResult> = {
  trainModel: (payload: TTrainPayload) => Promise<TTrainResult>;
  prependTrainingRun: (row: TrainingRunRow) => void;
  onProgress: (current: number, total: number) => void;
  formatCompletedAt: (params: { date?: Date }, options?: Record<string, never>) => string;
  formatMetricNumber: (params: { value: unknown }, options?: Record<string, never>) => string;
  shouldContinue?: () => boolean;
};

export type CommonDistillationProblem<TMode extends string> = {
  datasetId: string;
  targetColumn: string;
  task: MlTaskType;
  trainingMode: TMode;
  saveDistilledModel: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  teacher: CommonTeacherConfig;
};

export type CommonDistillationDeps<TDistillPayload, TDistillResult> = {
  distillModel: (payload: TDistillPayload) => Promise<TDistillResult>;
  formatCompletedAt: (params: { date?: Date }, options?: Record<string, never>) => string;
  formatMetricNumber: (params: { value: unknown }, options?: Record<string, never>) => string;
};

export type RunCommonTrainingFn<TMode extends string, TTrainPayload, TTrainResult> = (
  problem: CommonTrainingProblem<TMode>,
  deps: CommonTrainingDeps<TTrainPayload, TTrainResult>
) => Promise<CommonTrainingOutcome>;

export type RunCommonDistillationFn<TMode extends string, TDistillPayload, TDistillResult> = (
  problem: CommonDistillationProblem<TMode>,
  deps: CommonDistillationDeps<TDistillPayload, TDistillResult>
) => Promise<CommonDistillationResult>;

export type FrameworkTrainingAdapters<
  TMode extends string,
  TTrainPayload,
  TTrainResult,
  TDistillPayload,
  TDistillResult,
> = {
  isDistillationSupportedMode: (mode: string) => mode is TMode;
  trainModel: (payload: TTrainPayload) => Promise<TTrainResult>;
  distillModel: (payload: TDistillPayload) => Promise<TDistillResult>;
};

export type UseTrainingFrameworkLogicArgs<
  TMode extends string,
  TTrainPayload,
  TTrainResult,
  TDistillPayload,
  TDistillResult,
> = {
  dataset: MlDatasetViewModel;
  trainingRuns: TrainingRunRow[];
  prependTrainingRun: (row: TrainingRunRow) => void;
  ui: FrameworkTrainingUiState<TMode>;
  runTraining: RunCommonTrainingFn<TMode, TTrainPayload, TTrainResult>;
  runDistillation: RunCommonDistillationFn<TMode, TDistillPayload, TDistillResult>;
  runtime?: Partial<BaseTrainingRuntimeDeps>;
  framework: FrameworkTrainingAdapters<
    TMode,
    TTrainPayload,
    TTrainResult,
    TDistillPayload,
    TDistillResult
  >;
};

export type FrameworkDistilledSnapshot = Record<
  string,
  {
    metrics: TrainingMetrics;
    modelId: string | null;
    modelPath: string | null;
    comparison: DistillComparison;
  }
>;
