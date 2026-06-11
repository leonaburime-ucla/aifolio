import type { MlTrainingUiBaseState } from "@aifolio/contracts/entities/ml-training";
import type { MlDatasetViewModel } from "@aifolio/contracts/entities/ml-training";
import type { TensorflowTrainingMode } from "@aifolio/contracts/entities/ml-training";
import type { useTrainingIntegration } from "@/features/ml/react/hooks/training.hooks";
import type {
  useTrainingFrameworkLogic,
  useTrainingFrameworkUiState,
} from "@/features/ml/react/hooks/trainingFramework.hooks";
import type {
  RunTensorflowDistillationDeps,
  RunTensorflowDistillationProblem,
  RunTensorflowDistillationResult,
  RunTensorflowTrainingDeps,
  RunTensorflowTrainingProblem,
  RunTensorflowTrainingResult,
} from "@aifolio/contracts/entities/ml-training";
import type { UseTrainingRunsState } from "@aifolio/contracts/entities/ml-training";
import type {
  DistillComparison,
  TrainingMetrics,
  TrainingRunRow,
} from "@aifolio/contracts/entities/ml-training";

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
