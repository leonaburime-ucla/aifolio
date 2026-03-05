import type { MlTrainingUiBaseState } from "@/features/ml/__types__/typescript/react/hooks/mlTrainingUiBase.types";
import type { MlDatasetViewModel } from "@/features/ml/__types__/typescript/react/orchestrators/mlDatasetOrchestrator.types";
import type { TensorflowTrainingMode } from "@/features/ml/__types__/typescript/api/tensorflowApi.types";
import type { useTrainingIntegration } from "@/features/ml/typescript/react/hooks/training.hooks";
import type {
  useTrainingFrameworkLogic,
  useTrainingFrameworkUiState,
} from "@/features/ml/typescript/react/hooks/trainingFramework.hooks";
import type {
  RunTensorflowDistillationDeps,
  RunTensorflowDistillationProblem,
  RunTensorflowDistillationResult,
  RunTensorflowTrainingDeps,
  RunTensorflowTrainingProblem,
  RunTensorflowTrainingResult,
} from "@/features/ml/__types__/typescript/react/orchestrators/tensorflowTrainingOrchestrator.types";
import type { UseTrainingRunsState } from "@/features/ml/__types__/typescript/react/hooks/trainingIntegration.types";
import type {
  DistillComparison,
  TrainingMetrics,
  TrainingRunRow,
} from "@/features/ml/__types__/typescript/utils/trainingRuns.types";

export type TensorflowUiState = MlTrainingUiBaseState & {
  trainingMode: TensorflowTrainingMode;
  setTrainingMode: (value: TensorflowTrainingMode) => void;
};

export type DistilledSnapshotByTeacher = Record<
  string,
  {
    metrics: TrainingMetrics;
    modelId: string | null;
    modelPath: string | null;
    comparison: DistillComparison;
  }
>;

export type TensorflowLogicArgs = {
  dataset: MlDatasetViewModel;
  trainingRuns: TrainingRunRow[];
  prependTrainingRun: (row: TrainingRunRow) => void;
  ui: TensorflowUiState;
  trainModel: RunTensorflowTrainingDeps["trainModel"];
  distillModel: RunTensorflowDistillationDeps["distillModel"];
  runTraining: (
    problem: RunTensorflowTrainingProblem,
    deps: RunTensorflowTrainingDeps
  ) => Promise<RunTensorflowTrainingResult>;
  runDistillation: (
    problem: RunTensorflowDistillationProblem,
    deps: RunTensorflowDistillationDeps
  ) => Promise<RunTensorflowDistillationResult>;
  runtime?: Partial<TensorflowRuntimeDeps>;
};

export type TensorflowRuntimeDeps = {
  notifySuccess: (message: string) => void;
  notifyError: (message: string) => void;
  schedule: (callback: () => void, delayMs: number) => void;
  writeClipboardText: (text: string) => Promise<void>;
};

export type TensorflowIntegrationArgs = {
  useDatasetState: () => MlDatasetViewModel;
  useTrainingRunsState: UseTrainingRunsState;
  trainModel: RunTensorflowTrainingDeps["trainModel"];
  distillModel: RunTensorflowDistillationDeps["distillModel"];
  runTraining: (
    problem: RunTensorflowTrainingProblem,
    deps: RunTensorflowTrainingDeps
  ) => Promise<RunTensorflowTrainingResult>;
  runDistillation: (
    problem: RunTensorflowDistillationProblem,
    deps: RunTensorflowDistillationDeps
  ) => Promise<RunTensorflowDistillationResult>;
  runtime?: Partial<TensorflowRuntimeDeps>;
};

export type TensorflowUiStateDeps = {
  useFrameworkUiState: typeof useTrainingFrameworkUiState;
};

export type TensorflowLogicDeps = {
  useFrameworkLogic: typeof useTrainingFrameworkLogic;
};

export type TensorflowIntegrationDeps = {
  useIntegration: typeof useTrainingIntegration;
};
