import type { MlTrainingUiBaseState } from "@/features/ml/__types__/typescript/react/hooks/mlTrainingUiBase.types";
import type { MlDatasetViewModel } from "@/features/ml/__types__/typescript/react/orchestrators/mlDatasetOrchestrator.types";
import type { PytorchTrainingMode } from "@/features/ml/__types__/typescript/api/pytorchApi.types";
import type { useTrainingIntegration } from "@/features/ml/typescript/react/hooks/training.hooks";
import type {
  useTrainingFrameworkLogic,
  useTrainingFrameworkUiState,
} from "@/features/ml/typescript/react/hooks/trainingFramework.hooks";
import type {
  RunPytorchDistillationDeps,
  RunPytorchDistillationProblem,
  RunPytorchDistillationResult,
  RunPytorchTrainingDeps,
  RunPytorchTrainingProblem,
  RunPytorchTrainingResult,
} from "@/features/ml/__types__/typescript/react/orchestrators/pytorchTrainingOrchestrator.types";
import type { UseTrainingRunsState } from "@/features/ml/__types__/typescript/react/hooks/trainingIntegration.types";
import type {
  DistillComparison,
  TrainingMetrics,
  TrainingRunRow,
} from "@/features/ml/__types__/typescript/utils/trainingRuns.types";

export type PytorchUiState = MlTrainingUiBaseState & {
  trainingMode: PytorchTrainingMode;
  setTrainingMode: (value: PytorchTrainingMode) => void;
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

export type PytorchLogicArgs = {
  dataset: MlDatasetViewModel;
  trainingRuns: TrainingRunRow[];
  prependTrainingRun: (row: TrainingRunRow) => void;
  ui: PytorchUiState;
  trainModel: RunPytorchTrainingDeps["trainModel"];
  distillModel: RunPytorchDistillationDeps["distillModel"];
  runTraining: (
    problem: RunPytorchTrainingProblem,
    deps: RunPytorchTrainingDeps
  ) => Promise<RunPytorchTrainingResult>;
  runDistillation: (
    problem: RunPytorchDistillationProblem,
    deps: RunPytorchDistillationDeps
  ) => Promise<RunPytorchDistillationResult>;
  runtime?: Partial<PytorchRuntimeDeps>;
};

export type PytorchRuntimeDeps = {
  notifySuccess: (message: string) => void;
  notifyError: (message: string) => void;
  schedule: (callback: () => void, delayMs: number) => void;
  writeClipboardText: (text: string) => Promise<void>;
};

export type PytorchIntegrationArgs = {
  useDatasetState: () => MlDatasetViewModel;
  useTrainingRunsState: UseTrainingRunsState;
  trainModel: RunPytorchTrainingDeps["trainModel"];
  distillModel: RunPytorchDistillationDeps["distillModel"];
  runTraining: (
    problem: RunPytorchTrainingProblem,
    deps: RunPytorchTrainingDeps
  ) => Promise<RunPytorchTrainingResult>;
  runDistillation: (
    problem: RunPytorchDistillationProblem,
    deps: RunPytorchDistillationDeps
  ) => Promise<RunPytorchDistillationResult>;
  runtime?: Partial<PytorchRuntimeDeps>;
};

export type PytorchUiStateDeps = {
  useFrameworkUiState: typeof useTrainingFrameworkUiState;
};

export type PytorchLogicDeps = {
  useFrameworkLogic: typeof useTrainingFrameworkLogic;
};

export type PytorchIntegrationDeps = {
  useIntegration: typeof useTrainingIntegration;
};
